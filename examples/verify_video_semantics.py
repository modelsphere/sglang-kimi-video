"""Check temporal answers and cache isolation against actual model responses."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
from pathlib import Path

from chat_video import media_part, send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31023")
    parser.add_argument("--model", default="kimi-k3.1-pro")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = json.loads((args.fixtures / "expected.json").read_text())
    results = []
    # Cross the hybrid cache checkpoint interval while keeping the first video
    # inside the cacheable prefix. Short prompts can answer correctly without
    # exercising any KV reuse at all.
    video_question = "Look at colors. " * 75 + (
        "This video has three consecutive solid background colors. List those three colors in chronological order. Output exactly three English words chosen from RED, GREEN, BLUE, separated by spaces. Ignore the small moving white square."
    )
    cases = [
        (
            name,
            video_question,
            [media_part("video", args.fixtures / name)],
            expected[name],
        )
        for name in ["forward.mp4", "forward.mp4", "reverse.mp4", "forward.mp4"]
    ]
    cases.extend(
        [
            (
                "image",
                "What color is the triangle? Output just its English color name.",
                [media_part("image", args.fixtures / "yellow-triangle.png")],
                ["YELLOW"],
            ),
            (
                "mixed",
                "There are three media inputs in this message. For each in order, list only its colors: the three video background colors chronologically, the triangle color, then the three colors of the last video chronologically. Output exactly seven English color words, separated by spaces. Ignore the small white squares.",
                [
                    media_part("video", args.fixtures / "forward.mp4"),
                    media_part("image", args.fixtures / "yellow-triangle.png"),
                    media_part("video", args.fixtures / "reverse.mp4"),
                ],
                ["RED", "GREEN", "BLUE", "YELLOW", "BLUE", "GREEN", "RED"],
            ),
            ("text", "What is 2 + 2? Reply with just the digit.", [], ["4"]),
        ]
    )
    for index, (name, question, parts, wanted) in enumerate(cases):
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                        *parts,
                    ],
                }
            ],
            "max_tokens": 128,
            "reasoning_effort": "none",
            "temperature": 0,
            # Observe cache hits without changing the deployment's CLI flags.
            "return_meta_info": True,
        }
        # Preserve the replayable request and complete response before asserting.
        (args.output / f"{index}-{name}.request.json").write_text(json.dumps(payload))
        started = time.monotonic()
        try:
            with send(args.base_url, payload) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            (args.output / f"{index}-{name}.error.json").write_text(
                json.dumps({"status": error.code, "body": error.read().decode()}) + "\n"
            )
            raise
        elapsed = time.monotonic() - started
        (args.output / f"{index}-{name}.response.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        answer = result["choices"][0]["message"].get("content") or ""
        pattern = r"\b4\b" if name == "text" else r"\b(?:RED|GREEN|BLUE|YELLOW)\b"
        colors = re.findall(pattern, answer.upper())
        details = result["usage"].get("prompt_tokens_details") or {}
        meta = result["choices"][0].get("meta_info") or {}
        cached = meta.get("cached_tokens", details.get("cached_tokens", 0))
        cache_required = index in (1, 3)
        # Ensure the media itself is within the checkpointed prefix, rather
        # than proving reuse of a long text-only prefix.
        if index < 4:
            details = result["usage"].get("prompt_tokens_details") or {}
            text_tokens = result["usage"]["prompt_tokens"] - details["video_tokens"]
        else:
            text_tokens = None
        passed = colors == wanted and (not cache_required or cached > text_tokens)
        if index == 2:
            # The reversed video is new under this question. Reusing the text
            # prefix is valid; borrowing visual KV from the forward video is not.
            passed = passed and cached <= text_tokens
        results.append(
            {
                "video": name,
                "answer": answer,
                "expected": wanted,
                "colors": colors,
                "passed": passed,
                "cache_required": cache_required,
                "cache_isolation_required": index == 2,
                "non_video_prompt_tokens": text_tokens,
                "cached_tokens": cached,
                "seconds": elapsed,
                "usage": result["usage"],
            }
        )
        (args.output / "summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            f"{'PASS' if passed else 'FAIL'} {name}: {answer!r} ({elapsed:.2f}s)",
            flush=True,
        )
        if not passed:
            raise AssertionError(f"Temporal video answer differs from fixture: {name}")


if __name__ == "__main__":
    main()
