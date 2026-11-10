"""Validate video input before any SGLang URL or filesystem loader runs."""

from __future__ import annotations

import base64
import binascii
import re
from contextlib import contextmanager
from typing import Callable

from .config import VideoConfig
from .sampling import split_video

_DATA_URL = re.compile(r"data:video/[A-Za-z0-9!#$&^_.+\-]+;base64,", re.ASCII)


def decode_data_url(value: str) -> bytes:
    if not isinstance(value, str) or (match := _DATA_URL.match(value)) is None:
        raise ValueError("Video requires data:video/<format>;base64,<payload>")
    try:
        payload = base64.b64decode(value[match.end() :], validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Video contains invalid Base64") from None
    if not payload:
        raise ValueError("Video payload is empty")
    return payload


@contextmanager
def open_decoder(payload: bytes):
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    # Register ownership before __init__: a backend can create a temporary file
    # and then raise while probing corrupt bytes. __enter__ would be too late.
    decoder = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
    try:
        VideoDecoderWrapper.__init__(
            decoder, payload, device="cpu", num_decode_threads=1
        )
        yield decoder
    finally:
        decoder.close()


def load_video(
    url: str,
    config: VideoConfig,
    source_index: int,
    *,
    decoder_factory: Callable = open_decoder,
):
    payload = decode_data_url(url)
    try:
        with decoder_factory(payload) as decoder:
            return split_video(decoder, config, source_index)
    except (ImportError, ModuleNotFoundError):
        raise RuntimeError(
            "Kimi-K3 video requires working torchcodec or decord and FFmpeg"
        ) from None
    except Exception as error:
        # Decoder errors can contain input data; do not copy their text into logs.
        raise ValueError(
            f"Cannot decode video at media index {source_index} ({type(error).__name__})"
        ) from None
