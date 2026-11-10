"""Video-only branching around the built-in image processor."""

from __future__ import annotations

import asyncio
import functools

from . import VIDEO_ITEM_KEY
from .config import VideoConfig
from .decode import load_video
from .identity import video_identity
from .messages import ordered_sources
from .preprocess import preprocess_visual_items, resize_config
from .prompt import encode_special, expand_prompt
from .sampling import VisualItem


def close_images(items):
    for item in items:
        if item.media_type == "image":
            close = getattr(item.frames[0], "close", None)
            if close is not None:
                close()


def load_sources(owner, sources, config):
    from sglang.srt.multimodal.modality import Modality
    from sglang.srt.multimodal.processors.kimi_common import KimiLoadedImage
    from sglang.srt.multimodal.processors.kimi_k25 import _get_image_dimensions

    items = []
    try:
        for source_index, (kind, source) in enumerate(sources):
            if kind == "video":
                items.extend(load_video(source.url, config, source_index))
                continue
            loaded = owner._load_single_item(
                source, Modality.IMAGE, discard_alpha_channel=False
            )
            image = loaded.image if isinstance(loaded, KimiLoadedImage) else loaded
            try:
                width, height = _get_image_dimensions(image)
                digest = owner.kimi_content_config_digest(loaded)
                items.append(
                    VisualItem(
                        "image",
                        source_index,
                        (image,),
                        width,
                        height,
                        config.image_patch_limit,
                        content_digest=digest,
                    )
                )
            except BaseException:
                close = getattr(image, "close", None)
                if close is not None:
                    close()
                raise
        return items
    except BaseException:
        close_images(items)
        raise


def prepare_output(
    owner, items, config, input_text, max_input_len=None, *, processor=None
):
    import torch
    from sglang.srt.environ import envs
    from sglang.srt.managers.schedule_batch import (
        MultimodalDataItem,
        MultimodalProcessorOutput,
    )
    from sglang.srt.multimodal.modality import Modality
    from sglang.srt.multimodal.processors.kimi_common import kimi_image_identity
    from sglang.srt.multimodal.processors.kimi_k25 import navit_resize_config
    from sglang.srt.multimodal.transport.cuda_ipc import (
        DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    )

    try:
        if processor is None:
            # The executor supplies a thread-local clone when available. Its
            # fallback can mean this remote processor cannot be deep-copied;
            # only read it here and keep normalization/configuration local.
            processor = owner._processor
        device = owner._fast_image_processor_device(processor)
        if (
            device is None
            or not str(device).startswith("cuda")
            or not torch.cuda.is_available()
        ):
            raise RuntimeError("Kimi-K3 video preprocessing requires a CUDA service")
        if envs.SGLANG_FORCE_CPU_IMAGE_PREPROCESSING.get():
            raise ValueError(
                "Kimi-K3 video does not support forced CPU image preprocessing"
            )
        configs = [resize_config(item, config, navit_resize_config) for item in items]
        original_ids = (
            encode_special(owner._tokenizer, input_text)
            if isinstance(input_text, str)
            else input_text
        )
        input_ids, offsets = expand_prompt(
            original_ids,
            owner.mm_tokens.image_token_id,
            items,
            configs,
            owner._tokenizer,
        )
        if max_input_len is not None and len(input_ids) > max_input_len:
            raise ValueError("Kimi-K3 video prompt exceeds the maximum input length")
        # Complete all input validation before GPU allocation/encoding.
        for source_index in {
            item.source_index for item in items if item.media_type == "video"
        }:
            count = sum(
                len(item.frames) * resize["num_tokens"] * config.merge_size**2
                for item, resize in zip(items, configs)
                if item.source_index == source_index
            )
            if (
                config.video_patch_limit is not None
                and count > config.video_patch_limit
            ):
                raise ValueError(
                    "Kimi-K3 padded video patches exceed the source-video budget"
                )
        features, grids = preprocess_visual_items(items, config, configs, device=device)
        mm_items = []
        for item, feature, grid, offset in zip(items, features, grids, offsets):
            if offset[1] - offset[0] + 1 != grid[1] * grid[2] // config.merge_size**2:
                raise ValueError(
                    "Kimi-K3 token span disagrees with the pooled temporal grid"
                )
            model_data = {"image_grid_thw": torch.tensor([grid], dtype=torch.int64)}
            if item.media_type == "video":
                model_data[VIDEO_ITEM_KEY] = True
            if owner.keep_mm_features_on_device:
                model_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = True
            mm_item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=feature,
                offsets=[offset],
                model_specific_data=model_data,
            )
            if item.media_type == "video":
                mm_item.set_identity(
                    video_identity(item, config, configs[len(mm_items)], grid)
                )
            elif not envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
                mm_item.set_identity(kimi_image_identity(item.content_digest, grid))
            mm_items.append(mm_item)
        # Do not use _finalize_mm_items: its image-only identity assignment
        # would overwrite video identities. Keep the existing transport stages.
        owner._precompute_hashes_before_cpu_transfer(mm_items)
        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_token_id=owner.mm_tokens.image_token_id,
        )
    finally:
        close_images(items)


async def _wait_for_loaded(future):
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        # The IO thread owns decoders until it finishes. Dispose any images it
        # produces after the caller has disconnected, without closing in-use data.
        def dispose(done):
            if not done.cancelled() and done.exception() is None:
                close_images(done.result())

        future.add_done_callback(dispose)
        raise


async def process_mm_data(
    original, self, image_data, input_text, request_obj, *args, **kwargs
):
    video_data = getattr(request_obj, "video_data", None)
    if not video_data:
        return await original(
            self, image_data, input_text, request_obj, *args, **kwargs
        )
    if self.hf_config.model_type != "kimi_k31":
        raise ValueError("This video plugin supports Kimi-K3.1-Pro (kimi_k31) only")
    if kwargs.get("audio_data") or getattr(request_obj, "audio_data", None):
        raise ValueError(
            "Kimi-K3 supports video without audio; separate audio is unsupported"
        )
    # serving_chat adds _question internally even when no public video_config
    # was supplied. It is metadata, not a per-request preprocessing strategy.
    request_config = getattr(request_obj, "video_config", None) or {}
    if any(key != "_question" for key in request_config) or getattr(
        request_obj, "use_audio_in_video", False
    ):
        raise ValueError("Kimi-K3 per-request video configuration is unsupported")
    self.reject_caller_image_identity(image_data, request_obj)
    config = VideoConfig.from_environment(
        self._processor.media_processor.media_proc_cfg,
    )
    config.validate_vision(self.hf_config.vision_config)
    sources = ordered_sources(list(image_data or []), list(video_data))
    loop = asyncio.get_running_loop()
    items = await _wait_for_loaded(
        loop.run_in_executor(
            self.io_executor,
            load_sources,
            self,
            sources,
            config,
        )
    )
    # Once scheduled, prepare_output owns loaded images and closes them even
    # after cancellation. Transport leases are allocated only after it returns.
    executor = self.mm_processor_executor
    max_input_len = kwargs.get("max_req_input_len")
    try:
        if executor is not None:
            work = asyncio.create_task(
                executor.run(
                    prepare_output, self, items, config, input_text, max_input_len
                )
            )
        else:
            work = loop.run_in_executor(
                self.io_executor,
                functools.partial(
                    prepare_output, self, items, config, input_text, max_input_len
                ),
            )
    except BaseException:
        close_images(items)
        raise
    # Also cover an executor failing before it invokes prepare_output, e.g. a
    # failed thread-local clone. Closing a fully materialized PIL image twice is safe.
    work.add_done_callback(lambda done: close_images(items))
    try:
        output = await asyncio.shield(work)
    except asyncio.CancelledError:
        # Retrieve background exceptions; prepare_output owns resource cleanup.
        work.add_done_callback(
            lambda done: done.exception() if not done.cancelled() else None
        )
        raise
    except BaseException:
        # The Future owns the exception, whose traceback retains this frame.
        # Drop the reverse reference so failed GPU preprocessing tensors do not
        # stay alive until cyclic GC runs, even when requests are serialized.
        work = None
        raise
    output.mm_items = self._prepare_mm_items_for_transport(output.mm_items)
    return output
