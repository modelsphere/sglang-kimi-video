"""K3.1 request hooks. Original request content is never rewritten."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import SOURCE_INDEX_KEY
from .decode import decode_data_url


def media_order(request: Any) -> list[str]:
    order = []
    for message in request.messages:
        content = (
            message.content if hasattr(message, "content") else message.get("content")
        )
        if not isinstance(content, list):
            continue
        for part in content:
            kind = part.type if hasattr(part, "type") else part.get("type")
            if kind in ("image_url", "input_image"):
                order.append("image")
            elif kind == "video_url":
                order.append("video")
    return order


def is_video_request(serving: Any, request: Any) -> bool:
    return getattr(
        serving, "chat_encoding_spec", None
    ) == "kimi_k3" and "video" in media_order(request)


def validate_request(request: Any) -> None:
    if getattr(request, "mm_processor_kwargs", None):
        raise ValueError(
            "Kimi-K3 per-request multimodal processor options are not supported for video"
        )
    if getattr(request, "video_config", None) or getattr(
        request, "use_audio_in_video", False
    ):
        raise ValueError(
            "Kimi-K3 video_config and audio-track processing are not supported"
        )
    for message in request.messages:
        content = (
            message.content if hasattr(message, "content") else message.get("content")
        )
        if not isinstance(content, list):
            continue
        for part in content:
            data = (
                part.model_dump(exclude_none=True)
                if hasattr(part, "model_dump")
                else part
            )
            if data.get("type") != "video_url":
                continue
            video = data.get("video_url") or {}
            if isinstance(video, str):
                video = {"url": video}
            if any(value is not None for key, value in video.items() if key != "url"):
                raise ValueError(
                    "Kimi-K3 per-request video preprocessing options are not supported"
                )
            # Early validation gives invalid streaming requests a normal HTTP 400.
            decode_data_url(video.get("url"))


def validate_model(serving: Any) -> None:
    config = serving.tokenizer_manager.model_config.hf_config
    if config.model_type != "kimi_k31":
        raise ValueError("This video plugin supports Kimi-K3.1-Pro (kimi_k31) only")


def prepare_messages(original, self, messages, request):
    if "video" not in media_order(request):
        return original(self, messages, request)
    validate_model(self)
    validate_request(request)
    messages = deepcopy(messages)
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "video_url":
                    part["type"] = "image_url"
                    part["image_url"] = part.pop("video_url")
    return original(self, messages, request)


def apply_template(original, self, request, tools, is_multimodal):
    result = original(self, request, tools, is_multimodal)
    if not is_video_request(self, request):
        return result
    if not is_multimodal:
        raise ValueError("Video input requires a multimodal Kimi-K3 model")
    from sglang.srt.utils import VideoData

    order = media_order(request)
    images, videos = result.image_data or [], result.video_data or []
    if order.count("image") != len(images) or order.count("video") != len(videos):
        raise ValueError(
            "Kimi-K3 source media counts do not match the rendered request"
        )
    wrapped = []
    for source_index, video in zip(
        (i for i, kind in enumerate(order) if kind == "video"), videos
    ):
        url = video.url if isinstance(video, VideoData) else video
        wrapped.append(
            VideoData(url=url, preprocess_kwargs={SOURCE_INDEX_KEY: source_index})
        )
    result.video_data = wrapped
    # This field is indexed per request after n-way fan-out. A flat media-order
    # list breaks video-only n>1 requests (there is no image normalization to
    # replace it). The envelopes above preserve the full source order instead.
    result.modalities = []
    return result


def convert_request(original, self, request, raw_request=None):
    adapted, processed = original(self, request, raw_request)
    params = adapted.sampling_params
    beam_width = params.get("beam_width", 1) if isinstance(params, dict) else 1
    if (
        is_video_request(self, request)
        and getattr(request, "n", 1) > 1
        and (beam_width or 1) <= 1
    ):
        # _normalize_video_data copies the outer list. Group all source videos
        # as one request before n-way fan-out; do the same for mixed images.
        adapted.video_data = [adapted.video_data]
        if adapted.image_data:
            adapted.image_data = [adapted.image_data]
        # With no images, _normalize_image_data leaves this field untouched;
        # a flat media list would then be indexed as n request rows. Ordering
        # remains available in our video envelopes for every sub-request.
        adapted.modalities = None
        if adapted.mm_content_hashes is not None:
            adapted.mm_content_hashes = [adapted.mm_content_hashes]
    return adapted, processed


def ordered_sources(image_data, video_data):
    """Recover order even after batch normalization overwrites modalities."""
    count = len(image_data) + len(video_data)
    sources = [None] * count
    for video in video_data:
        metadata = getattr(video, "preprocess_kwargs", None) or {}
        index = metadata.get(SOURCE_INDEX_KEY)
        if (
            type(index) is not int
            or not 0 <= index < count
            or sources[index] is not None
        ):
            raise ValueError("Kimi-K3 video source order is missing or inconsistent")
        sources[index] = ("video", video)
    images = iter(image_data)
    return [
        source if source is not None else ("image", next(images)) for source in sources
    ]
