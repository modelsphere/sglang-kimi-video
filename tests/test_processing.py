from __future__ import annotations

import asyncio
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import SRT
from PIL import Image

from kimi_k3_video import VIDEO_ITEM_KEY
from kimi_k3_video.config import VideoConfig
from kimi_k3_video.decode import load_video, open_decoder
from kimi_k3_video.preprocess import preprocess_visual_items, resize_config
from kimi_k3_video.processor import _wait_for_loaded, prepare_output
from kimi_k3_video.sampling import VisualItem


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_temporal_preprocessing_and_pooling(
    preprocess_source, source_modules, config_values, t
):
    config = VideoConfig.from_mapping(config_values)
    # Spatially constant frames give independent expected patch values.
    frames = tuple(
        torch.full((3, 28, 28), index * 40, dtype=torch.uint8) for index in range(t)
    )
    item = VisualItem("video", 0, frames, 28, 28, 16)
    configs = [resize_config(item, config, preprocess_source.navit_resize_config)]
    features, grids = preprocess_visual_items([item], config, configs, device="cpu")
    assert grids == [(t, 2, 2)]
    assert features[0].shape == (t * 4, 3, 14, 14)
    for index in range(t):
        torch.testing.assert_close(
            features[0][index * 4 : (index + 1) * 4],
            torch.full((4, 3, 14, 14), index * 40 / 127.5 - 1),
        )
    merger = source_modules.nodes(
        "isolated_merger", SRT / "models/kimi_vl_moonvit.py", {"tpool_patch_merger"}
    )
    x = torch.arange(t, dtype=torch.float32).repeat_interleave(4).unsqueeze(1)
    merged = merger.tpool_patch_merger(x, torch.tensor(grids))
    assert merged[0].shape == (1, 4, 1)
    assert torch.equal(merged[0], torch.full((1, 4, 1), (t - 1) / 2))


def test_mixed_rgba_image_uses_original_alpha_order(preprocess_source, config_values):
    config_values["transparent_bg_config"] = {"pattern": "white"}
    config = VideoConfig.from_mapping(config_values)
    rgba = Image.new("RGBA", (28, 28), (255, 0, 0, 0))
    items = [
        VisualItem("image", 0, (rgba,), 28, 28, 16),
        VisualItem(
            "video", 1, (torch.zeros(3, 28, 28, dtype=torch.uint8),), 28, 28, 16
        ),
    ]
    configs = [
        resize_config(item, config, preprocess_source.navit_resize_config)
        for item in items
    ]
    features, grids = preprocess_visual_items(items, config, configs, device="cpu")
    assert grids == [(1, 2, 2), (1, 2, 2)]
    torch.testing.assert_close(
        features[0], torch.ones_like(features[0]), rtol=0, atol=1e-6
    )
    assert torch.equal(features[1], -torch.ones_like(features[1]))
    rgba.close()


def test_prepare_composes_real_items_and_preserves_identity(
    preprocess_source,
    req_source,
    source_modules,
    config_values,
    tokenizer,
    monkeypatch,
):
    # Use the real CPU operator fallback to verify composition. This device
    # adapter is not CUDA evidence; test_cuda.py exercises that separately.
    from kimi_k3_video import processor

    fields = SimpleNamespace(
        SGLANG_FORCE_CPU_IMAGE_PREPROCESSING=SimpleNamespace(get=lambda: False),
        SGLANG_MM_SKIP_COMPUTE_HASH=SimpleNamespace(get=lambda: True),
    )
    source_modules.module("sglang.srt.environ").envs = fields
    source_modules.module(
        "sglang.srt.multimodal.processors.kimi_common"
    ).kimi_image_identity = lambda *args: None
    source_modules.module(
        "sglang.srt.multimodal.transport.cuda_ipc"
    ).DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY = "defer"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        processor,
        "preprocess_visual_items",
        lambda *args, **kwargs: preprocess_visual_items(*args, device="cpu"),
    )
    config = VideoConfig.from_mapping(config_values)
    item = VisualItem(
        "video", 0, (torch.zeros(3, 28, 28, dtype=torch.uint8),), 28, 28, 16
    )
    owner = SimpleNamespace(
        _fast_image_processor_device=lambda processor: "cuda:0",
        _tokenizer=tokenizer,
        mm_tokens=SimpleNamespace(image_token_id=99),
        keep_mm_features_on_device=False,
        _precompute_hashes_before_cpu_transfer=lambda items: None,
    )
    output = prepare_output(owner, [item], config, [1, 99, 2], processor=object())
    actual = output.mm_items[0]
    assert actual.modality is req_source.Modality.IMAGE
    assert actual.model_specific_data[VIDEO_ITEM_KEY] is True
    assert actual.model_specific_data["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert actual.feature.shape == (4, 3, 14, 14)
    assert len(actual.identity) == 32 and len(actual.pad_values) == 4
    start, end = actual.offsets[0]
    assert start == end and output.input_ids[start] == 99
    with pytest.raises(ValueError, match="maximum input"):
        prepare_output(owner, [item], config, [1, 99, 2], 2, processor=object())


def test_partial_decoder_initialization_cleans_temporary_file(source_modules, tmp_path):
    path = tmp_path / "created.mp4"
    baseline = source_modules.nodes(
        "isolated_decoder",
        SRT / "utils/video_decoder.py",
        {"VideoDecoderWrapper"},
        os=__import__("os"),
    )

    class FailingDecoder:
        close = baseline.VideoDecoderWrapper.close

        def __init__(self, source, **kwargs):
            self._tmp_path = str(path)
            path.write_bytes(source)
            raise RuntimeError("bad container")

    source_modules.module(
        "sglang.srt.utils.video_decoder"
    ).VideoDecoderWrapper = FailingDecoder
    with pytest.raises(RuntimeError, match="container"):
        with open_decoder(b"bad"):
            pytest.fail("Decoder should not open")
    assert not path.exists()


def test_cancellation_disposes_images_when_io_finishes():
    closed, started, finish = [], threading.Event(), threading.Event()
    image = SimpleNamespace(close=lambda: closed.append(True))
    item = VisualItem("image", 0, (image,), 1, 1, 4)

    def load():
        started.set()
        assert finish.wait(3)
        return [item]

    async def run(pool):
        future = asyncio.get_running_loop().run_in_executor(pool, load)
        task = asyncio.create_task(_wait_for_loaded(future))
        await asyncio.sleep(0)
        assert started.wait(3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        finish.set()
        await future
        await asyncio.sleep(0)

    with ThreadPoolExecutor(max_workers=1) as pool:
        asyncio.run(run(pool))
    assert closed == [True]


def test_real_mp4_sampling_through_test_decoder_adapter(config_values):
    av = pytest.importorskip("av", reason="Optional real-MP4 test adapter needs PyAV")
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for value in range(7):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), value * 30, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    closed = []

    @contextmanager
    def adapter(payload):
        # Test-only adapter: production exclusively uses VideoDecoderWrapper.
        with av.open(io.BytesIO(payload)) as container:
            fps = float(container.streams.video[0].average_rate)
            frames = np.stack(
                [
                    frame.to_ndarray(format="rgb24")
                    for frame in container.decode(video=0)
                ]
            )

            class Decoder:
                avg_fps = fps

                def __len__(self):
                    return len(frames)

                def get_frames_at(self, indices):
                    return frames[indices]

            yield Decoder()
        closed.append(True)

    chunks = load_video(
        "data:video/mp4;base64," + base64.b64encode(output.getvalue()).decode(),
        VideoConfig.from_mapping(config_values),
        0,
        decoder_factory=adapter,
    )
    assert [len(chunk.frames) for chunk in chunks] == [4, 3]
    assert [chunk.timestamp for chunk in chunks] == ["00:00:00.000", "00:00:01.000"]
    assert closed == [True]


@pytest.mark.parametrize(
    "overrides,expected_fps,expected_temporal",
    [
        ({}, 8, 4),
        (
            {
                "KIMI_K3_VIDEO_SAMPLE_FPS": "4",
                "KIMI_K3_VIDEO_TEMPORAL_MERGE_KERNEL_SIZE": "2",
            },
            4,
            2,
        ),
    ],
)
def test_async_video_accepts_internal_question_metadata(
    config_values, io_source, monkeypatch, overrides, expected_fps, expected_temporal
):
    from kimi_k3_video import processor
    from kimi_k3_video.config import VIDEO_ENV_VARS

    for name in VIDEO_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)

    _, utils = io_source
    chunks = [
        VisualItem("video", 0, (torch.zeros(3, 28, 28, dtype=torch.uint8),), 28, 28, 16)
    ]
    result = SimpleNamespace(mm_items=["feature"])
    observed = []

    def load_sources(owner, sources, config):
        observed.append(config)
        return chunks

    def prepare_output(owner, items, config, *args, **kwargs):
        observed.append(config)
        return result

    monkeypatch.setattr(processor, "load_sources", load_sources)
    monkeypatch.setattr(processor, "prepare_output", prepare_output)
    owner = SimpleNamespace(
        _processor=SimpleNamespace(
            media_processor=SimpleNamespace(media_proc_cfg=config_values)
        ),
        hf_config=SimpleNamespace(
            model_type="kimi_k31",
            vision_config=SimpleNamespace(
                patch_size=14, merge_kernel_size=(2, 2), init_pos_emb_time=4
            ),
        ),
        mm_processor_executor=None,
        reject_caller_image_identity=lambda *args: None,
        _prepare_mm_items_for_transport=lambda items: items,
    )
    from kimi_k3_video import SOURCE_INDEX_KEY

    obj = SimpleNamespace(
        video_data=[
            utils.VideoData("data:video/mp4;base64,dmlkZW8=", {SOURCE_INDEX_KEY: 0})
        ],
        video_config={"_question": "Describe the video"},
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        owner.io_executor = pool
        assert (
            asyncio.run(processor.process_mm_data(None, owner, [], [99], obj)) is result
        )
    assert len(observed) == 2 and observed[0] is observed[1]
    assert observed[0].sample_fps == expected_fps
    assert observed[0].temporal_merge == expected_temporal
    assert config_values["sample_fps"] == 8
    assert config_values["temporal_merge_kernel_size"] == 4
    obj.video_config["fps"] = 4
    with pytest.raises(ValueError, match="per-request"):
        asyncio.run(processor.process_mm_data(None, owner, [], [99], obj))


def test_later_video_failure_closes_previously_loaded_image(
    preprocess_source, req_source, source_modules, config_values
):
    from kimi_k3_video.processor import load_sources

    source_modules.nodes(
        "sglang.srt.multimodal.processors.kimi_common",
        SRT / "multimodal/processors/kimi_common.py",
        {"KimiLoadedImage"},
    )
    image = Image.new("RGBA", (28, 28))
    owner = SimpleNamespace(
        _load_single_item=lambda *args, **kwargs: image,
        kimi_content_config_digest=lambda source: "sha256:" + "00" * 32,
    )
    with pytest.raises(ValueError, match="requires data"):
        load_sources(
            owner,
            [("image", "fixture"), ("video", SimpleNamespace(url="not-a-data-url"))],
            VideoConfig.from_mapping(config_values),
        )
    with pytest.raises(ValueError):
        image.getpixel((0, 0))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
