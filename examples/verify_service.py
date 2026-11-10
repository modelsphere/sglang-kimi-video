"""Protocol smoke checks against a real deployed model, without fake responses.

Uses actual supplied media. It checks protocol/usage/recovery, not semantic
video-understanding accuracy or the absence of wrong KV cache reuse.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path

from chat_video import media_part, send


def call(
    base_url,
    model,
    parts,
    *,
    stream=False,
    n=1,
    evidence_dir=None,
    case_name="request",
    messages=None,
):
    request = {
        "model": model,
        "messages": messages
        if messages is not None
        else [
            {
                "role": "user",
                "content": [{"type": "text", "text": "请简要描述输入内容。"}, *parts],
            }
        ],
        "max_tokens": 64,
        "stream": stream,
        "n": n,
    }
    if stream:
        request["stream_options"] = {"include_usage": True}
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / f"{case_name}.request.json").write_text(json.dumps(request))
    try:
        response = send(base_url, request)
    except urllib.error.HTTPError as error:
        if evidence_dir is not None:
            body = error.read()
            (evidence_dir / f"{case_name}.error.json").write_text(
                json.dumps({"status": error.code, "body": body.decode()}) + "\n"
            )
        raise
    with response:
        if not stream:
            result = json.load(response)
            if evidence_dir is not None:
                (evidence_dir / f"{case_name}.response.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
                )
            assert len(result["choices"]) == n, result
            return result["usage"]
        usage, done, events = None, False, []
        completed_choices = set()
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                continue
            item = json.loads(line[6:])
            events.append(item)
            if evidence_dir is not None:
                # Keep the partial stream if a later event fails validation.
                (evidence_dir / f"{case_name}.response.json").write_text(
                    json.dumps(events, ensure_ascii=False, indent=2) + "\n"
                )
            assert "error" not in item, item
            for choice in item.get("choices", []):
                if choice.get("finish_reason") is not None:
                    completed_choices.add(choice["index"])
            if item.get("usage"):
                usage = item["usage"]
        assert done and usage is not None, "Missing final stream usage or DONE"
        assert completed_choices == set(range(n)), completed_choices
        return usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--model", default="kimi-k3.1-pro")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--second-video", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--tail-video",
        type=Path,
        action="append",
        default=[],
        help="Additional short videos exercising actual tail lengths",
    )
    parser.add_argument("--output", type=Path, default=Path("service-validation.json"))
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="Save replayable requests and complete responses",
    )
    args = parser.parse_args()
    video, second, image = (
        media_part("video", args.video),
        media_part("video", args.second_video),
        media_part("image", args.image),
    )
    results = []
    cases = [
        ("text", [], False, 1),
        ("image", [image], False, 1),
        ("video", [video], False, 1),
        ("video_repeat", [video], False, 1),
        ("video_stream", [video], True, 1),
        ("video_n2", [video], False, 2),
        ("video_stream_n2", [video], True, 2),
        ("second_video", [second], False, 1),
        ("multiple_videos", [video, second], False, 1),
        ("mixed", [video, image, second], False, 1),
        ("mixed_stream", [video, image, second], True, 1),
        ("mixed_n2", [video, image, second], False, 2),
        ("mixed_stream_n2", [video, image, second], True, 2),
    ]
    cases.extend(
        (f"tail_{index + 1}", [media_part("video", path)], False, 1)
        for index, path in enumerate(args.tail_video)
    )
    mixed_details = video_details = None
    for label, parts, stream, n in cases:
        usage = call(
            args.base_url,
            args.model,
            parts,
            stream=stream,
            n=n,
            evidence_dir=args.evidence_dir,
            case_name=label,
        )
        details = usage.get("prompt_tokens_details") or {}
        if any(part["type"] == "video_url" for part in parts):
            assert details.get("video_tokens", 0) > 0, usage
        if any(part["type"] == "image_url" for part in parts):
            assert details.get("image_tokens", 0) > 0, usage
        if parts and all(part["type"] == "video_url" for part in parts):
            assert details.get("image_tokens", 0) == 0, usage
        if label == "video":
            video_details = details
        elif label.startswith("video_"):
            assert details.get("video_tokens", 0) == video_details.get(
                "video_tokens", 0
            )
        if label == "mixed":
            mixed_details = details
        elif label.startswith("mixed_"):
            assert mixed_details is not None
            for key in ("image_tokens", "video_tokens"):
                assert details.get(key, 0) == mixed_details.get(key, 0), (label, usage)
        results.append({"case": label, "usage": usage})
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(f"PASS {label}", flush=True)
    sparse_usage = call(
        args.base_url,
        args.model,
        [video],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Observe the video carefully. " * 1024},
                    video,
                    {"type": "text", "text": "Describe its temporal changes briefly."},
                ],
            }
        ],
        evidence_dir=args.evidence_dir,
        case_name="video_sparse_context",
    )
    assert sparse_usage["prompt_tokens"] > 2048, sparse_usage
    assert sparse_usage["prompt_tokens_details"]["video_tokens"] > 0, sparse_usage
    results.append({"case": "video_sparse_context", "usage": sparse_usage})
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("PASS video_sparse_context", flush=True)
    cross_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Remember this video."}, video],
        },
        {"role": "assistant", "content": "Understood."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe all media in order."},
                image,
                second,
            ],
        },
    ]
    cross_usage = call(
        args.base_url,
        args.model,
        [video, image, second],
        messages=cross_messages,
        evidence_dir=args.evidence_dir,
        case_name="cross_message",
    )
    cross_details = cross_usage.get("prompt_tokens_details") or {}
    assert (
        cross_details.get("video_tokens", 0) > 0
        and cross_details.get("image_tokens", 0) > 0
    ), cross_usage
    results.append({"case": "cross_message", "usage": cross_usage})
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("PASS cross_message", flush=True)
    # Three 12-frame fixtures produce nine temporal chunks across eight ranks.
    # A distinct leading question prevents reuse of the earlier KV prefix.
    uneven_usage = call(
        args.base_url,
        args.model,
        [video, second, video],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these three videos in order."},
                    video,
                    second,
                    video,
                ],
            }
        ],
        evidence_dir=args.evidence_dir,
        case_name="uneven_video_chunks",
    )
    uneven_details = uneven_usage.get("prompt_tokens_details") or {}
    assert (
        uneven_details.get("video_tokens", 0) > 0
        and uneven_details.get("image_tokens", 0) == 0
    ), uneven_usage
    results.append({"case": "uneven_video_chunks", "usage": uneven_usage})
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("PASS uneven_video_chunks", flush=True)
    for index, invalid in enumerate(
        [
            "https://example.invalid/video.mp4",
            "file:///tmp/video.mp4",
            "/tmp/video.mp4",
            "dmlkZW8=",
            "data:video/mp4;base64,%%%%",
            "data:video/mp4;base64,bm90LWEtdmlkZW8=",
        ]
    ):
        try:
            call(
                args.base_url,
                args.model,
                [{"type": "video_url", "video_url": {"url": invalid}}],
                evidence_dir=args.evidence_dir,
                case_name=f"invalid-{index}",
            )
        except urllib.error.HTTPError as error:
            if (
                args.evidence_dir is not None
                and not (args.evidence_dir / f"invalid-{index}.error.json").exists()
            ):
                (args.evidence_dir / f"invalid-{index}.error.json").write_text(
                    json.dumps(
                        {"status": error.code, "body": error.read().decode()},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
            assert error.code == 400, error.code
            results.append({"case": "invalid_input", "status": error.code})
        else:
            raise AssertionError("Invalid video input was accepted")
    call(
        args.base_url,
        args.model,
        [video],
        evidence_dir=args.evidence_dir,
        case_name="healthy-after-errors",
    )
    results.append({"case": "healthy_after_errors", "status": "passed"})
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
