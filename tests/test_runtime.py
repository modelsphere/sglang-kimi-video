"""Fresh HTTP worker import must activate hooks without parent state."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


def run_runtime(code, **env):
    try:
        importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Requires the installed SGLang service environment")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, **env},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.runtime
def test_fresh_http_worker_loads_plugin():
    run_runtime(
        "import kimi_k3_video.http_app; "
        "from kimi_k3_video.compat import check_targets; "
        "check_targets(applied=True)",
        SGLANG_PLUGINS="kimi_k3_video",
    )


@pytest.mark.runtime
def test_four_fresh_workers_load_hooks_independently():
    code = """
import os
import kimi_k3_video.http_app
from kimi_k3_video.compat import check_targets
from kimi_k3_video.plugin import HOOKS
from sglang.srt.plugins import load_plugins
from sglang.srt.plugins.hook_registry import HookRegistry
load_plugins()
check_targets(applied=True)
for target, hook, _ in HOOKS:
    assert sum(existing is hook for _, existing, _ in HookRegistry._hooks[target]) == 1
print('PID:' + str(os.getpid()))
"""
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: run_runtime(code, SGLANG_PLUGINS="kimi_k3_video"), range(4)
            )
        )
    pids = {result.stdout.strip().splitlines()[-1] for result in results}
    assert len(pids) == 4 and all(pid.startswith("PID:") for pid in pids)


@pytest.mark.runtime
def test_installed_sources_match_pinned_interfaces():
    run_runtime(
        "from kimi_k3_video.compat import check_sources, check_version; "
        "check_version(); assert check_sources()['checked_files'] == 19"
    )


@pytest.mark.runtime
def test_real_request_normalization_preserves_all_video_sources():
    run_runtime(
        """
import pickle
from kimi_k3_video import SOURCE_INDEX_KEY
from kimi_k3_video.messages import ordered_sources, convert_request
from types import SimpleNamespace
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import ImageData, VideoData

for kinds in [('video',), ('video', 'video'), ('video', 'image', 'video')]:
    for n in [1, 2, 3]:
        images = [ImageData(f'image-{i}') for i, k in enumerate(kinds) if k == 'image']
        videos = [VideoData(f'video-{i}', {SOURCE_INDEX_KEY: i}) for i, k in enumerate(kinds) if k == 'video']
        request = GenerateReqInput(input_ids=[1, 99, 2], image_data=images or None,
            video_data=videos, modalities=[], sampling_params={'n': n})
        external = SimpleNamespace(n=n, messages=[{'content':[{'type':'video_url'}]}])
        request, _ = convert_request(lambda *args: (request, external), SimpleNamespace(chat_encoding_spec='kimi_k3'), external)
        request.normalize_batch_and_arguments()
        rows = [request] if n == 1 else [request[i] for i in range(n)]
        for row in rows:
            row = pickle.loads(pickle.dumps(row))
            sources = ordered_sources(row.image_data or [], row.video_data)
            assert tuple(k for k, _ in sources) == kinds
            assert [v.url for _, v in sources] == [f'{k}-{i}' for i, k in enumerate(kinds)]
"""
    )


@pytest.mark.runtime
def test_real_msgpack_preserves_temporal_item_and_cache_identity():
    run_runtime(
        """
import hashlib
import msgspec
import torch
from kimi_k3_video import VIDEO_ITEM_KEY
from kimi_k3_video.usage import compute_token_counts
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem, MultimodalProcessorOutput, MultimodalInputs,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.utils.msgpack_utils import enc_hook, dec_hook, ext_hook

item = MultimodalDataItem(
    modality=Modality.IMAGE, offsets=[(2, 2)],
    feature=torch.arange(12, dtype=torch.float32).reshape(4, 3),
    model_specific_data={VIDEO_ITEM_KEY: True, 'image_grid_thw': torch.tensor([[1, 2, 2]])},
)
item.set_identity(hashlib.sha256(b'actual-video-frames').digest())
identity = item.identity
identity, pads = item.identity, item.pad_values
output = MultimodalProcessorOutput(mm_items=[item], input_ids=[1, 2, 99], im_token_id=99)
wire = msgspec.msgpack.encode(output, enc_hook=enc_hook)
decoded = msgspec.msgpack.decode(wire, type=MultimodalProcessorOutput, dec_hook=dec_hook, ext_hook=ext_hook)
received = decoded.mm_items[0]
assert received.identity == identity and received.pad_values == pads
assert received.model_specific_data[VIDEO_ITEM_KEY] is True
assert received.model_specific_data['image_grid_thw'].tolist() == [[1, 2, 2]]
assert torch.equal(received.feature, torch.arange(12, dtype=torch.float32).reshape(4, 3))
inputs = MultimodalInputs(mm_items=[received])
assert compute_token_counts(MultimodalInputs.compute_mm_token_counts, inputs) == (0, 0, 1)
""",
        SGLANG_USE_PICKLE_IPC="0",
    )


@pytest.mark.runtime
def test_native_cpu_decoder_and_doctor(tmp_path):
    av = pytest.importorskip("av")
    import numpy as np

    path = tmp_path / "seven-frames.mp4"
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for index in range(7):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), index * 30, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    from pathlib import Path

    model_path = Path(__file__).parent / "fixtures/kimi_k31"
    result = run_runtime(
        "from kimi_k3_video.doctor import main; "
        f"assert main(['--model-path', {str(model_path)!r}, '--video', {str(path)!r}]) == 0"
    )
    assert '"decoded_frames": 7' in result.stdout
    assert '"chunks": 2' in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
