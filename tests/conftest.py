"""CPU tests can execute unmodified source bodies without a CUDA installation.

The isolated fixture does not claim to exercise full-package imports. All module
bindings are fixture-scoped; runtime/CUDA tests use the real installed package.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib.util
import logging
import os
import sys
import types
import uuid
from collections import Counter, defaultdict
from enum import Enum, auto
from pathlib import Path

import msgspec
import numpy as np
import pytest
import torch
from PIL import Image


def test_source_path():
    configured = os.environ.get("KIMI_K3_VIDEO_TEST_SRT")
    if configured:
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "python/sglang/srt"
        if candidate.is_dir():
            return candidate
    spec = importlib.util.find_spec("sglang")
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).parent / "srt"
    raise RuntimeError(
        "Set KIMI_K3_VIDEO_TEST_SRT to the pinned SGLang source directory"
    )


SRT = test_source_path()


@pytest.fixture
def config_values():
    return {
        "sample_fps": 8.0,
        "temporal_merge_kernel_size": 4,
        "max_num_frames_each_video": None,
        "patch_size": 14,
        "merge_kernel_size": 2,
        "in_patch_limit": 65536,
        "in_patch_limit_each_frame": 16384,
        "in_patch_limit_video": 655360,
        "patch_limit_on_one_side": 512,
        "fixed_output_tokens": None,
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
        "timestamp_mode": "hh:mm:ss.fff",
        "transparent_bg_config": None,
    }


class SourceModules:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch

    def module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        if "." in name:
            self.module(name.rsplit(".", 1)[0])
        module = types.ModuleType(name)
        module.__path__ = []
        self.monkeypatch.setitem(sys.modules, name, module)
        return module

    def full(self, name, path):
        self.module(name.rsplit(".", 1)[0])
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        self.monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    def nodes(self, name, path, names, **extra):
        module = self.module(name)
        namespace = {
            "dataclasses": dataclasses,
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "copy": copy,
            "hashlib": hashlib,
            "uuid": uuid,
            "Counter": Counter,
            "defaultdict": defaultdict,
            "Enum": Enum,
            "auto": auto,
            "msgspec": msgspec,
            "np": np,
            "torch": torch,
            "nn": torch.nn,
            "F": torch.nn.functional,
            "Image": Image,
            "logger": logging.getLogger(name),
            **extra,
        }
        module.__dict__.update(namespace)
        tree = ast.parse(path.read_text())
        selected = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in names:
                    selected.append(node)
            elif isinstance(node, ast.Assign):
                if any(
                    isinstance(target, ast.Name) and target.id in names
                    for target in node.targets
                ):
                    selected.append(node)
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        source = ast.fix_missing_locations(
            ast.Module(body=[future, *selected], type_ignores=[])
        )
        exec(compile(source, str(path), "exec"), module.__dict__)
        return module


@pytest.fixture
def source_modules(monkeypatch):
    # No changes to sys.modules at test module import time.
    loader = SourceModules(monkeypatch)
    loader.module("sglang")
    loader.module("sglang.srt")
    return loader


@pytest.fixture(autouse=True)
def compatibility_source_root(monkeypatch):
    from kimi_k3_video import compat

    monkeypatch.setattr(compat, "source_root", lambda: SRT.parent)


@pytest.fixture
def io_source(source_modules):
    from typing import Any, List, Optional, Union

    utils = source_modules.nodes(
        "sglang.srt.utils",
        SRT / "utils/common.py",
        {"VideoData", "ImageData"},
    )
    io = source_modules.nodes(
        "sglang.srt.managers.io_struct",
        SRT / "managers/io_struct.py",
        {"GenerateReqInput"},
        Any=Any,
        List=List,
        Optional=Optional,
        Union=Union,
        get_return_hidden_states_mode=lambda value: value,
    )
    return io, utils


@pytest.fixture
def req_source(source_modules):
    modality = source_modules.full(
        "sglang.srt.multimodal.modality", SRT / "multimodal/modality.py"
    )
    return source_modules.nodes(
        "sglang.srt.managers.schedule_batch",
        SRT / "managers/schedule_batch.py",
        {
            "MM_PAD_SHIFT_VALUE",
            "_MM_HASH_MASK",
            "_MM_IDENTITY_BYTES",
            "_MM_WIDE_PAD_TAG",
            "_MM_WIDE_PAD_MASK",
            "_compute_pad_values",
            "MultimodalDataItem",
            "MultimodalProcessorOutput",
            "MultimodalInputs",
        },
        Modality=modality.Modality,
        MultimodalInputFormat=modality.MultimodalInputFormat,
    )


@pytest.fixture
def registry_source(source_modules):
    return source_modules.full(
        "sglang.srt.plugins.hook_registry", SRT / "plugins/hook_registry.py"
    )


@pytest.fixture
def preprocess_source(source_modules):
    image = source_modules.nodes(
        "isolated_patchify",
        SRT.parent / "kernels/ops/mm/process/image.py",
        {"_normalize_and_patchify_torch", "normalize_and_patchify"},
    )
    source_modules.nodes(
        "sglang.srt.multimodal.kimi_k3_image_processing",
        SRT / "multimodal/kimi_k3_image_processing.py",
        {"to_chw_uint8", "normalization_tensors", "fill_transparent_bg"},
    )
    return source_modules.nodes(
        "sglang.srt.multimodal.processors.kimi_k25",
        SRT / "multimodal/processors/kimi_k25.py",
        {
            "navit_resize_config",
            "_grid_thw_from_resize_config",
            "_resize_bicubic_if_needed",
            "_process_single_image",
            "_resize_images_by_source_shape",
            "_gpu_preprocess_images",
            "_get_image_dimensions",
            "_to_cuda_chw",
            "_pil_to_cuda_chw",
            "_ensure_chw_rgb",
        },
        math=__import__("math"),
        normalize_and_patchify=image.normalize_and_patchify,
    )


class RecordingTokenizer:
    def __init__(self):
        self.encoded = []

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        # Disjoint from the structural placeholder ID used by tests.
        return [10_000 + ord(char) for char in text]


@pytest.fixture
def tokenizer():
    return RecordingTokenizer()
