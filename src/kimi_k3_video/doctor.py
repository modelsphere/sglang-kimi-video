"""Installation and effective-hook checks, suitable for a startup wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .compat import check_decoder, check_sources, check_targets, check_version
from .config import VideoConfig, video_config_overrides


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-hooks",
        action="store_true",
        help="Load SGLang and inspect all effective hooks",
    )
    parser.add_argument(
        "--model-path", type=Path, help="Validate the deployed preprocessor_config.json"
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Decode a local smoke-test file through the Base64-only input path",
    )
    args = parser.parse_args(argv)
    try:
        overrides = video_config_overrides()
        result = {
            "plugin": __version__,
            "sglang": check_version(),
            "runtime_sources": check_sources(),
            "max_frames": overrides.get("max_num_frames_each_video"),
            "video_overrides": overrides,
        }
        result["decoder"] = check_decoder()
        config = None
        if args.model_path:
            model = json.loads((args.model_path / "config.json").read_text())
            if model.get("model_type") != "kimi_k31":
                raise ValueError("Video requires a Kimi-K3.1-Pro (kimi_k31) model")
            values = json.loads(
                (args.model_path / "preprocessor_config.json").read_text()
            )["media_proc_cfg"]
            config = VideoConfig.from_environment(values)
            from types import SimpleNamespace

            config.validate_vision(SimpleNamespace(**model["vision_config"]))
            result["model_config"] = "ok"
            result["video_config"] = asdict(config)
        if args.check_hooks:
            from sglang.srt.plugins import load_plugins

            check_targets()
            load_plugins()
            check_targets(applied=True)
            from .plugin import HOOKS

            result["hooks"] = f"{len(HOOKS)} active"
        if args.video:
            if config is None:
                raise ValueError("--video requires --model-path")
            import base64

            from .decode import load_video

            chunks = load_video(
                "data:video/mp4;base64,"
                + base64.b64encode(args.video.read_bytes()).decode(),
                config,
                0,
            )
            result["decoded_frames"] = sum(len(chunk.frames) for chunk in chunks)
            result["chunks"] = len(chunks)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"Kimi-K3 video check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
