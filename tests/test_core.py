from __future__ import annotations

import base64
import dataclasses
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kimi_k3_video.config import VideoConfig, plugin_frame_limit
from kimi_k3_video.decode import decode_data_url, load_video
from kimi_k3_video.identity import video_identity
from kimi_k3_video.preprocess import constrain_video_resize, resize_config
from kimi_k3_video.prompt import expand_prompt
from kimi_k3_video.sampling import VisualItem, sampling_indices, split_video


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/x.mp4",
        "/tmp/x.mp4",
        "file:///tmp/x.mp4",
        "YWJj",
        "data:image/png;base64,YWJj",
        "data:video/mp4;base64,",
        "data:video/mp4;base64,%%%%",
        "data:video/mp4;base64,AA A=",
        "data:video/mp4;base64,你好",
        None,
    ],
)
def test_reject_non_base64_video(value):
    with pytest.raises(ValueError):
        decode_data_url(value)


def test_decode_exact_bytes():
    payload = bytes(range(256))
    assert (
        decode_data_url("data:video/mp4;base64," + base64.b64encode(payload).decode())
        == payload
    )


@pytest.mark.parametrize("value", ["", "0", "-1", "+1", "1.5", "true", "４", " 4"])
def test_invalid_plugin_limit(value):
    with pytest.raises(ValueError):
        plugin_frame_limit({"KIMI_K3_VIDEO_MAX_FRAMES": value})


def test_config_and_limits(config_values):
    config_values["max_num_frames_each_video"] = 10
    assert VideoConfig.from_mapping(config_values, max_frames=4).frame_limit == 4
    assert VideoConfig.from_mapping(config_values, max_frames=20).frame_limit == 10
    config_values["max_num_frames_each_video"] = None
    assert VideoConfig.from_mapping(config_values).frame_limit is None
    assert plugin_frame_limit({}) is None
    assert plugin_frame_limit({"KIMI_K3_VIDEO_MAX_FRAMES": "4"}) == 4


@pytest.mark.parametrize(
    "key,value",
    [
        ("sample_fps", float("nan")),
        ("sample_fps", float("inf")),
        ("sample_fps", 0),
        ("temporal_merge_kernel_size", 5),
        ("patch_size", True),
        ("in_patch_limit_each_frame", 0),
        ("image_std", [0, 0.5, 0.5]),
        ("timestamp_mode", "invalid"),
        ("merge_kernel_size", 0),
    ],
)
def test_invalid_model_config(config_values, key, value):
    config_values[key] = value
    with pytest.raises(ValueError):
        VideoConfig.from_mapping(config_values)


def test_missing_and_vision_config(config_values):
    config = VideoConfig.from_mapping(config_values)
    config.validate_vision(
        SimpleNamespace(patch_size=14, merge_kernel_size=(2, 2), init_pos_emb_time=4)
    )
    with pytest.raises(ValueError):
        config.validate_vision(
            SimpleNamespace(
                patch_size=14, merge_kernel_size=(2, 2), init_pos_emb_time=2
            )
        )
    del config_values["sample_fps"]
    with pytest.raises(ValueError, match="missing"):
        VideoConfig.from_mapping(config_values)


@pytest.mark.parametrize(
    "total,fps,expected",
    [
        (1, 1, [0]),
        (3, 1, [0, 1, 2]),
        (10, 10, [0, 1, 3, 4, 5, 6, 8, 9]),
        (
            81,
            16,
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                21,
                23,
                25,
                27,
                29,
                31,
                33,
                35,
                37,
                39,
                41,
                43,
                45,
                47,
                49,
                51,
                53,
                55,
                57,
                59,
                62,
                64,
                66,
                68,
                70,
                72,
                74,
                76,
                78,
                80,
            ],
        ),
        (9, 30, [0, 8]),
        (12, 30, [0, 6, 11]),
        (5, 16, [0, 4]),
        (7, 16, [0, 2, 4, 6]),
        (1, 16, [0]),
        (1, 30, [0]),
        (8, 8, list(range(8))),
        (4, 16, [0, 3]),
    ],
)
def test_uniform_sampling(config_values, total, fps, expected):
    assert (
        sampling_indices(total, fps, VideoConfig.from_mapping(config_values))
        == expected
    )


@pytest.mark.parametrize(
    "model_limit,plugin_limit,expected_count",
    [(None, None, 40), (None, 4, 4), (12, None, 12), (12, 4, 4), (45, 50, 40)],
)
def test_rounded_sampling_respects_frame_limits(
    config_values, model_limit, plugin_limit, expected_count
):
    config_values["max_num_frames_each_video"] = model_limit
    config = VideoConfig.from_mapping(config_values, max_frames=plugin_limit)
    indices = sampling_indices(81, 16, config)
    assert len(indices) == expected_count
    assert indices[0] == 0 and indices[-1] == 80
    assert all(left < right for left, right in zip(indices, indices[1:]))


@pytest.mark.parametrize(
    "total,chunk_lengths,last_indices,last_timestamp",
    [
        (81, [4] * 10, (74, 76, 78, 80), "00:00:04.625"),
        (82, [4] * 10 + [1], (81,), "00:00:05.062"),
    ],
)
def test_fractional_sampling_chunks_after_rounding(
    config_values, total, chunk_lengths, last_indices, last_timestamp
):
    decoder = type(
        "Decoder",
        (),
        {
            "__len__": lambda self: total,
            "avg_fps": 16,
            "get_frames_at": lambda self, indices: np.zeros(
                (len(indices), 3, 4, 3), dtype=np.uint8
            ),
        },
    )()
    items = split_video(decoder, VideoConfig.from_mapping(config_values), 0)
    assert [len(item.frames) for item in items] == chunk_lengths
    indices = [index for item in items for index in item.frame_indices]
    assert len(indices) == sum(chunk_lengths)
    assert indices[0] == 0 and indices[-1] == total - 1
    assert all(left < right for left, right in zip(indices, indices[1:]))
    assert items[-1].frame_indices == last_indices
    assert items[-1].timestamp == last_timestamp


@pytest.mark.parametrize(
    "total,fps", [(0, 1), (1, 0), (1, float("inf")), (1, float("nan"))]
)
def test_bad_video_metadata(config_values, total, fps):
    with pytest.raises(ValueError):
        sampling_indices(total, fps, VideoConfig.from_mapping(config_values))


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6, 7, 8])
def test_tail_chunks_and_rgb_layout(config_values, count):
    frames = np.arange(count * 3 * 4 * 3, dtype=np.uint8).reshape(count, 3, 4, 3)
    decoder = SimpleNamespace(avg_fps=4, get_frames_at=lambda indices: frames[indices])
    decoder = type(
        "Decoder",
        (),
        {
            "__len__": lambda self: count,
            "avg_fps": 4,
            "get_frames_at": lambda self, indices: frames[indices],
        },
    )()
    items = split_video(decoder, VideoConfig.from_mapping(config_values), 2)
    assert [len(item.frames) for item in items] == [
        min(4, count - start) for start in range(0, count, 4)
    ]
    reconstructed = torch.stack([frame for item in items for frame in item.frames])
    assert torch.equal(reconstructed, torch.from_numpy(frames).permute(0, 3, 1, 2))
    assert items[0].timestamp == "00:00:00.000"
    if count > 4:
        assert items[1].timestamp == "00:00:01.000"


def test_limit_shared_by_all_source_frames(config_values):
    config_values.update(
        in_patch_limit_each_frame=20, in_patch_limit_video=80, sample_fps=4
    )
    config = VideoConfig.from_mapping(config_values)
    assert config.per_frame_budget(8) == 10
    with pytest.raises(ValueError, match="minimum"):
        config.per_frame_budget(100)


def test_decoder_closed_on_sampling_error(config_values):
    closed = []

    @contextmanager
    def factory(payload):
        assert payload == b"video"
        try:
            yield SimpleNamespace(avg_fps=0)
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="media index 2"):
        load_video(
            "data:video/mp4;base64,dmlkZW8=",
            VideoConfig.from_mapping(config_values),
            2,
            decoder_factory=factory,
        )
    assert closed == [True]


def test_prompt_expansion_order_and_offsets(tokenizer):
    items = [
        VisualItem("video", 0, (), 10, 20, 16, timestamp="00:00:00.000"),
        VisualItem("video", 0, (), 10, 20, 16, timestamp="00:00:01.000"),
        VisualItem("image", 1, (), 30, 40, 16),
        VisualItem("video", 2, (), 10, 20, 16, timestamp="00:00:00.000"),
    ]
    counts = [2, 3, 4, 1]
    output, offsets = expand_prompt(
        [1, 99, 2, 99, 3, 99, 4],
        99,
        items,
        [{"num_tokens": n} for n in counts],
        tokenizer,
    )
    assert [end - start + 1 for start, end in offsets] == counts
    assert all(
        output[start : end + 1] == [99] * n for (start, end), n in zip(offsets, counts)
    )
    assert [token for token in output if token < 99] == [1, 2, 3, 4]
    assert tokenizer.encoded[4] == "<|media_begin|>image 30x40<|media_content|>"
    with pytest.raises(ValueError, match="placeholder"):
        expand_prompt([99], 99, items, [{"num_tokens": n} for n in counts], tokenizer)


def test_video_identity_content_config_grid_and_frame_order(config_values):
    config = VideoConfig.from_mapping(config_values)
    frames = (
        torch.zeros(3, 4, 5, dtype=torch.uint8),
        torch.ones(3, 4, 5, dtype=torch.uint8),
    )
    item = VisualItem("video", 0, frames, 5, 4, 16)
    digest = video_identity(item, config, {}, [2, 2, 2])
    assert len(digest) == 32
    assert digest == video_identity(
        dataclasses.replace(item, source_index=5), config, {}, [2, 2, 2]
    )
    assert digest != video_identity(
        dataclasses.replace(item, frames=frames[::-1]), config, {}, [2, 2, 2]
    )
    assert digest != video_identity(
        item, dataclasses.replace(config, image_mean=(0, 0, 0)), {}, [2, 2, 2]
    )
    assert digest != video_identity(item, config, {}, [2, 4, 2])


@pytest.mark.parametrize(
    "width,height,budget",
    [(1920, 1080, 100), (10000, 1, 4), (1, 10000, 4), (57, 57, 4), (224, 224, 256)],
)
def test_padded_budget_is_enforced(
    source_modules, config_values, width, height, budget
):
    from conftest import SRT

    baseline = source_modules.nodes(
        "baseline_resize",
        SRT / "multimodal/processors/kimi_k25.py",
        {"navit_resize_config"},
        math=__import__("math"),
    )
    config = VideoConfig.from_mapping(config_values)
    item = VisualItem("video", 0, (), width, height, budget)
    result = resize_config(item, config, baseline.navit_resize_config)
    h = (result["new_height"] + result["pad_height"]) // 14
    w = (result["new_width"] + result["pad_width"]) // 14
    assert h * w <= budget
    assert h % 2 == w % 2 == 0
    assert result["num_tokens"] == h * w // 4
    assert result["new_width"] <= width and result["new_height"] <= height
    with pytest.raises(ValueError, match="minimum"):
        constrain_video_resize(result, 3, config)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
