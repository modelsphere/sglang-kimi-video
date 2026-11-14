"""Plugin-owned configuration; no SGLang configuration is mutated."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

MAX_FRAMES_ENV = "KIMI_K3_VIDEO_MAX_FRAMES"
TIMESTAMP_MODES = ("hh:mm:ss.fff", "mm:ss.fff", "mm:ss")
VIDEO_ENV_VARS = {
    "KIMI_K3_VIDEO_SAMPLE_FPS": "sample_fps",
    MAX_FRAMES_ENV: "max_num_frames_each_video",
    "KIMI_K3_VIDEO_TEMPORAL_MERGE_KERNEL_SIZE": "temporal_merge_kernel_size",
    "KIMI_K3_VIDEO_IN_PATCH_LIMIT_EACH_FRAME": "in_patch_limit_each_frame",
    "KIMI_K3_VIDEO_IN_PATCH_LIMIT_VIDEO": "in_patch_limit_video",
    "KIMI_K3_VIDEO_PATCH_LIMIT_ON_ONE_SIDE": "patch_limit_on_one_side",
    "KIMI_K3_VIDEO_TIMESTAMP_MODE": "timestamp_mode",
}


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def plugin_frame_limit(environ: Mapping[str, str] | None = None) -> int | None:
    value = (os.environ if environ is None else environ).get(MAX_FRAMES_ENV)
    if value is None:
        return None
    return _positive_env_int(value, MAX_FRAMES_ENV)


def _positive_env_int(value: str, name: str) -> int:
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def video_config_overrides(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environ = os.environ if environ is None else environ
    overrides = {}
    for name, key in VIDEO_ENV_VARS.items():
        if name not in environ:
            continue
        value = environ[name]
        if key == "sample_fps":
            try:
                fps = float(value)
            except ValueError:
                raise ValueError(f"{name} must be finite and positive") from None
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"{name} must be finite and positive")
            overrides[key] = fps
        elif key == "timestamp_mode":
            if value not in TIMESTAMP_MODES:
                raise ValueError(f"{name} must be one of {', '.join(TIMESTAMP_MODES)}")
            overrides[key] = value
        elif key == "in_patch_limit_video" and value.lower() == "none":
            overrides[key] = None
        else:
            parsed = _positive_env_int(value, name)
            if key == "temporal_merge_kernel_size" and parsed > 4:
                raise ValueError(f"{name} must be between 1 and 4")
            overrides[key] = parsed
    return overrides


@dataclass(frozen=True)
class VideoConfig:
    sample_fps: float
    temporal_merge: int
    frame_limit: int | None
    patch_size: int
    merge_size: int
    image_patch_limit: int
    frame_patch_limit: int
    video_patch_limit: int | None
    side_patch_limit: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    timestamp_mode: str
    transparent_bg_config: dict | None
    fixed_output_tokens: int | None

    @classmethod
    def from_environment(
        cls,
        values: Mapping[str, Any],
        environ: Mapping[str, str] | None = None,
    ) -> VideoConfig:
        overrides = video_config_overrides(environ)
        # Preserve the model frame cap while applying overrides to a local copy.
        max_frames = overrides.pop("max_num_frames_each_video", None)
        return cls.from_mapping({**values, **overrides}, max_frames=max_frames)

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any], *, max_frames: int | None = None
    ) -> VideoConfig:
        required = (
            "sample_fps",
            "temporal_merge_kernel_size",
            "patch_size",
            "merge_kernel_size",
            "in_patch_limit",
            "patch_limit_on_one_side",
            "image_mean",
            "image_std",
            "timestamp_mode",
        )
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"Kimi-K3 media config is missing: {', '.join(missing)}")
        fps = values["sample_fps"]
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("sample_fps must be finite and positive")
        temporal = positive_int(
            values["temporal_merge_kernel_size"], "temporal_merge_kernel_size"
        )
        if temporal > 4:
            raise ValueError("temporal_merge_kernel_size must be between 1 and 4")
        model_limit = values.get("max_num_frames_each_video")
        limits = [
            positive_int(v, "max frames")
            for v in (model_limit, max_frames)
            if v is not None
        ]
        image_limit = positive_int(values["in_patch_limit"], "in_patch_limit")
        frame_limit = values.get("in_patch_limit_each_frame")
        frame_limit = (
            image_limit
            if frame_limit is None
            else positive_int(frame_limit, "in_patch_limit_each_frame")
        )
        video_limit = values.get("in_patch_limit_video")
        if video_limit is not None:
            positive_int(video_limit, "in_patch_limit_video")
        mean, std = (tuple(values[key]) for key in ("image_mean", "image_std"))
        if (
            len(mean) != 3
            or len(std) != 3
            or any(
                isinstance(x, bool)
                or not isinstance(x, (float, int))
                or not math.isfinite(x)
                for x in (*mean, *std)
            )
            or any(x <= 0 for x in std)
        ):
            raise ValueError(
                "image_mean/image_std require three finite values and positive std"
            )
        mode = values["timestamp_mode"]
        if mode not in TIMESTAMP_MODES:
            raise ValueError(f"Unsupported timestamp_mode: {mode}")
        fixed = values.get("fixed_output_tokens")
        if fixed is not None:
            positive_int(fixed, "fixed_output_tokens")
        if values.get("transparent_bg_fill_stage", "after_resize") != "after_resize":
            raise ValueError("Kimi-K3 requires transparent_bg_fill_stage=after_resize")
        return cls(
            float(fps),
            temporal,
            min(limits) if limits else None,
            positive_int(values["patch_size"], "patch_size"),
            positive_int(values["merge_kernel_size"], "merge_kernel_size"),
            image_limit,
            frame_limit,
            video_limit,
            positive_int(values["patch_limit_on_one_side"], "patch_limit_on_one_side"),
            mean,
            std,
            mode,
            deepcopy(values.get("transparent_bg_config")),
            fixed,
        )

    def validate_vision(self, vision_config: Any) -> None:
        if tuple(vision_config.merge_kernel_size) != (self.merge_size, self.merge_size):
            raise ValueError("Media and vision spatial merge configurations disagree")
        if self.patch_size != vision_config.patch_size:
            raise ValueError("Media and vision patch sizes disagree")
        if self.temporal_merge > vision_config.init_pos_emb_time:
            raise ValueError(
                "Temporal chunk exceeds the vision position encoding capacity"
            )

    def per_frame_budget(self, count: int) -> int:
        budget = self.frame_patch_limit
        if self.video_patch_limit is not None:
            budget = min(budget, max(1, round(self.video_patch_limit / count)))
        if budget < self.merge_size**2:
            raise ValueError(
                "Video patch budget cannot fit the minimum spatial merge grid"
            )
        return budget
