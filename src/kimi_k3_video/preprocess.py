"""Reuse SGLang's GPU image operators, retaining one grid per temporal item."""

from __future__ import annotations

from contextlib import nullcontext

from .config import VideoConfig


def resize_config(item, config: VideoConfig, resize_fn):
    result = resize_fn(
        item.width,
        item.height,
        config.patch_size,
        config.merge_size,
        item.patch_limit,
        config.side_patch_limit,
        config.fixed_output_tokens,
    )
    if item.media_type == "video":
        result = constrain_video_resize(result, item.patch_limit, config)
    h = (result["new_height"] + result["pad_height"]) // config.patch_size
    w = (result["new_width"] + result["pad_width"]) // config.patch_size
    if result["num_tokens"] != h * w // config.merge_size**2:
        raise ValueError("Kimi-K3 fixed token count disagrees with the spatial grid")
    return result


def constrain_video_resize(result, budget: int, config: VideoConfig):
    """Enforce the actual padded budget, including extreme aspect ratios.

    The reference NaViT estimate uses floor(width/patch) and pads afterward;
    its estimate can exceed the allocated video budget. Shrink only when needed.
    """
    factor = config.patch_size * config.merge_size

    def shape(width, height):
        pw, ph = (-width) % factor, (-height) % factor
        gw, gh = (width + pw) // config.patch_size, (height + ph) // config.patch_size
        return (
            {
                "new_width": width,
                "new_height": height,
                "pad_width": pw,
                "pad_height": ph,
                "num_tokens": gh * gw // config.merge_size**2,
            },
            gh,
            gw,
        )

    def fits(candidate):
        value, h, w = candidate
        return h * w <= budget and max(h, w) <= config.side_patch_limit

    width, height = result["new_width"], result["new_height"]
    if fits(shape(width, height)):
        return result
    if not fits(shape(1, 1)):
        raise ValueError("Video patch budget cannot fit the minimum spatial merge grid")
    low, high, edge = 1, max(width, height), max(width, height)
    best = shape(1, 1)[0]
    while low <= high:
        mid = (low + high) // 2
        candidate = shape(max(1, width * mid // edge), max(1, height * mid // edge))
        if fits(candidate):
            best = candidate[0]
            low = mid + 1
        else:
            high = mid - 1
    return best


def preprocess_visual_items(items, config, configs, *, device):
    import torch

    # Processor threads do not inherit another thread's CUDA current device.
    # Explicitly bind the owner device for Triton launches as well as tensors.
    scope = (
        torch.cuda.device(device)
        if torch.device(device).type == "cuda"
        else nullcontext()
    )
    with scope:
        return _preprocess_visual_items(items, config, configs, device=device)


def _preprocess_visual_items(items, config, configs, *, device):
    import torch
    from sglang.srt.multimodal.kimi_k3_image_processing import (
        fill_transparent_bg,
        normalization_tensors,
        to_chw_uint8,
    )
    from sglang.srt.multimodal.processors.kimi_k25 import _gpu_preprocess_images

    if not items or len(items) != len(configs):
        raise ValueError("Kimi-K3 requires matching visual items and resize configs")
    frames, frame_configs = [], []
    for item, resize in zip(items, configs):
        frames.extend(item.frames)
        frame_configs.extend([resize] * len(item.frames))
    scale, bias = normalization_tensors(config.image_mean, config.image_std, device)
    # No new kernel: use the checkpoint-compatible resize/alpha/norm/patchify.
    patches, frame_grids = _gpu_preprocess_images(
        frames,
        frame_configs,
        scale,
        bias,
        config.patch_size,
        to_chw=lambda frame: to_chw_uint8(frame, device=device),
        post_resize=lambda values: fill_transparent_bg(
            values, config.transparent_bg_config
        ),
    )
    grids, cursor = [], 0
    for item in items:
        count = len(item.frames)
        group = frame_grids[cursor : cursor + count]
        cursor += count
        if (
            count == 0
            or not torch.all(group[:, 0] == 1)
            or not torch.all(group[:, 1:] == group[0, 1:])
        ):
            raise ValueError(
                "All frames of a Kimi-K3 chunk must share one spatial grid"
            )
        grids.append((count, int(group[0, 1]), int(group[0, 2])))
    sizes = [t * h * w for t, h, w in grids]
    if sum(sizes) != patches.shape[0]:
        raise ValueError("Kimi-K3 patch tensor length does not match temporal grids")
    return list(patches.split(sizes)), grids
