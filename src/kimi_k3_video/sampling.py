"""Sampling and chunk prompts adapted from HF #172 / SGLang #35325."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import VideoConfig


@dataclass(frozen=True)
class VisualItem:
    media_type: str
    source_index: int
    frames: tuple[Any, ...]
    width: int
    height: int
    patch_limit: int
    chunk_index: int = 0
    frame_indices: tuple[int, ...] = ()
    timestamp: str = ""
    content_digest: str | None = None


def sampling_indices(total: int, fps: float, config: VideoConfig) -> list[int]:
    if total <= 0:
        raise ValueError("Video must contain at least one frame")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video must report a finite positive frame rate")
    # Python round uses nearest-even rounding for halfway sampling budgets.
    count = min(total, max(1, round(total * min(config.sample_fps, fps) / fps)))
    if config.frame_limit is not None:
        count = min(count, config.frame_limit)
    return np.linspace(0, total - 1, count).round().astype(int).tolist()


def format_timestamp(seconds: float, mode: str) -> str:
    # Match the pinned reference's truncation and time-of-day wrapping.
    total = max(0, int(seconds))
    millis = int((max(0.0, seconds) % 1) * 1000)
    second, minute = total % 60, (total // 60) % 60
    if mode == "hh:mm:ss.fff":
        return f"{(total // 3600) % 24:02d}:{minute:02d}:{second:02d}.{millis:03d}"
    if mode == "mm:ss.fff":
        return f"{minute:02d}:{second:02d}.{millis:03d}"
    if mode == "mm:ss":
        return f"{minute:02d}:{second:02d}"
    raise ValueError(f"Unsupported timestamp mode: {mode}")


def split_video(
    decoder: Any, config: VideoConfig, source_index: int
) -> list[VisualItem]:
    import torch

    fps = float(decoder.avg_fps or 0)
    indices = sampling_indices(len(decoder), fps, config)
    budget = config.per_frame_budget(len(indices))
    # Wrapper guarantees NHWC: use an explicit permutation even for tiny H=3/4.
    decoded = torch.as_tensor(decoder.get_frames_at(indices))
    if decoded.dtype != torch.uint8 or decoded.ndim != 4 or decoded.shape[-1] != 3:
        raise ValueError("Video decoder must return NHWC RGB uint8 frames")
    if decoded.shape[0] != len(indices) or min(decoded.shape[1:3]) <= 0:
        raise ValueError(
            "Video decoder returned an unexpected number or size of frames"
        )
    frames = decoded.permute(0, 3, 1, 2).contiguous()
    items = []
    for start in range(0, len(indices), config.temporal_merge):
        stop = start + config.temporal_merge
        # Clone chunks so views do not retain a whole decoded video allocation.
        chunk = frames[start:stop].clone()
        items.append(
            VisualItem(
                "video",
                source_index,
                tuple(chunk.unbind()),
                int(chunk.shape[3]),
                int(chunk.shape[2]),
                budget,
                len(items),
                tuple(indices[start:stop]),
                format_timestamp(indices[start] / fps, config.timestamp_mode),
            )
        )
    return items
