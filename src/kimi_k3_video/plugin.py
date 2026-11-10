"""Native SGLang plugin entry point, also invoked in scheduler workers."""

from __future__ import annotations

import logging

from . import __version__
from .config import video_config_overrides
from .messages import apply_template, convert_request, prepare_messages
from .processor import process_mm_data
from .usage import compute_token_counts
from .vision import position_embeddings

CHAT = "sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat"
PROCESSOR = "sglang.srt.multimodal.processors.kimi_k3.KimiK3ImageProcessor"
POSITION = "sglang.srt.models.kimi_k3_vl.Learnable2DInterpPosEmbDividedFixed"
INPUTS = "sglang.srt.managers.schedule_batch.MultimodalInputs"

HOOKS = (
    (
        f"{CHAT}._convert_to_internal_request",
        convert_request,
        ("request", "raw_request"),
    ),
    (f"{CHAT}._prepare_kimi_k3_messages", prepare_messages, ("messages", "request")),
    (
        f"{CHAT}._apply_jinja_template",
        apply_template,
        ("request", "tools", "is_multimodal"),
    ),
    (
        f"{PROCESSOR}.process_mm_data_async",
        process_mm_data,
        ("image_data", "input_text", "request_obj"),
    ),
    (
        f"{POSITION}.position_embeddings",
        position_embeddings,
        ("grid_thws", "grid_thw_list"),
    ),
    (f"{INPUTS}.compute_mm_token_counts", compute_token_counts, ()),
)


def register():
    from .compat import check_version

    check_version()
    video_config_overrides()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    added = 0
    for target, hook, _ in HOOKS:
        # Registry ownership survives fork; it can also be reset by tests.
        if any(
            existing is hook for _, existing, _ in HookRegistry._hooks.get(target, ())
        ):
            continue
        HookRegistry.register(target, hook, HookType.AROUND)
        added += 1
    if added:
        logging.getLogger(__name__).info(
            "Kimi-K3 video plugin %s registered %d hooks in this process",
            __version__,
            added,
        )
