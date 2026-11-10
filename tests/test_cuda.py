"""Run on the supported CUDA SGLang image; never substitute CPU results."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kimi_k3_video.config import VideoConfig
from kimi_k3_video.preprocess import preprocess_visual_items, resize_config
from kimi_k3_video.sampling import VisualItem
from kimi_k3_video.vision import position_embeddings

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA is required"),
]


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_cuda_temporal_preprocess_positions_pool_and_projector(config_values, t):
    from sglang.srt.models.kimi_k3_vl import (
        KimiK3MultiModalProjector,
        Learnable2DInterpPosEmbDividedFixed,
    )
    from sglang.srt.models.kimi_vl_moonvit import tpool_patch_merger
    from sglang.srt.multimodal.processors.kimi_k25 import navit_resize_config

    config = VideoConfig.from_mapping(config_values)
    frames = tuple(
        torch.full((3, 28, 28), index * 40, dtype=torch.uint8) for index in range(t)
    )
    item = VisualItem("video", 0, frames, 28, 28, 16)
    features, grids = preprocess_visual_items(
        [item],
        config,
        [resize_config(item, config, navit_resize_config)],
        device="cuda",
    )
    assert features[0].is_cuda and grids == [(t, 2, 2)]
    for index in range(t):
        torch.testing.assert_close(
            features[0][index * 4 : (index + 1) * 4],
            torch.full((4, 3, 14, 14), index * 40 / 127.5 - 1, device="cuda"),
        )
    embed = Learnable2DInterpPosEmbDividedFixed(2, 2, 4, 8).cuda()
    # Explicitly leave time buffer on the host to cover the device correction.
    embed.time_weight = embed.time_weight.cpu()
    positions = position_embeddings(
        type(embed).position_embeddings, embed, torch.tensor(grids)
    )
    assert positions.is_cuda and positions.shape == (t * 4, 8)
    merged = tpool_patch_merger(
        torch.ones(t * 4, 8, device="cuda"), torch.tensor(grids)
    )
    assert merged[0].shape == (1, 4, 8)
    projector = KimiK3MultiModalProjector(
        SimpleNamespace(
            mm_hidden_size=8,
            merge_kernel_size=(2, 2),
            text_hidden_size=16,
            projector_ln_eps=1e-5,
        )
    ).cuda()
    assert projector(merged).shape == (1, 16)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
