"""Validate video geometry against the actual K3.1 Pro checkpoint config."""

import json
from pathlib import Path
from types import SimpleNamespace

from kimi_k3_video.config import VideoConfig

FIXTURE = Path(__file__).parent / "fixtures/kimi_k31"


def test_checkpoint_video_geometry_and_sparse_attention():
    model = json.loads((FIXTURE / "config.json").read_text())
    media = json.loads((FIXTURE / "preprocessor_config.json").read_text())
    config = VideoConfig.from_mapping(media["media_proc_cfg"])
    config.validate_vision(SimpleNamespace(**model["vision_config"]))
    assert model["model_type"] == "kimi_k31"
    assert model["architectures"] == ["KimiK31ForConditionalGeneration"]
    assert (config.patch_size, config.temporal_merge, config.merge_size) == (14, 4, 2)
    assert model["text_config"]["dsa_config"]["index_topk"] == 2048


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
