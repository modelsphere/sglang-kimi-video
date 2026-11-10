"""Content-derived 256-bit identities for temporal visual items."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict


def video_identity(item, config, resize_config, grid) -> bytes:
    hasher = hashlib.sha256(b"sglang-kimi-k3-video-v1")
    metadata = json.dumps(
        {"config": asdict(config), "resize": resize_config, "grid": list(grid)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    _update(hasher, metadata)
    for frame in item.frames:
        # Decode takes place on CPU; avoid a GPU-to-CPU identity round trip.
        frame = frame.detach().cpu().contiguous()
        _update(hasher, str((str(frame.dtype), tuple(frame.shape))).encode())
        _update(hasher, frame.numpy().tobytes())
    return hasher.digest()


def _update(hasher, payload):
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)
