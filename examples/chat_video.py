"""Send ordered image/video inputs to a running service using only stdlib."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path


class AppendMedia(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        media = list(getattr(namespace, self.dest, None) or [])
        media.append(("video" if option_string == "--video" else "image", Path(values)))
        setattr(namespace, self.dest, media)


def media_part(kind, path):
    mime = mimetypes.guess_type(path.name)[0]
    if mime is None or not mime.startswith(kind + "/"):
        raise ValueError(f"Cannot determine {kind} MIME type: {path.name}")
    return {
        "type": f"{kind}_url",
        f"{kind}_url": {
            "url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
        },
    }


def send(base_url, payload):
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    return urllib.request.urlopen(req, timeout=600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--model", default="kimi-k3.1-pro")
    parser.add_argument(
        "--prompt", default="请按顺序描述这些图片和视频，并说明视频中的动作变化。"
    )
    parser.add_argument("--video", dest="media", action=AppendMedia)
    parser.add_argument("--image", dest="media", action=AppendMedia)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if not args.media:
        parser.error("Provide at least one --video or --image")
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    *[media_part(kind, path) for kind, path in args.media],
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "n": args.n,
        "stream": args.stream,
    }
    if args.stream:
        payload["stream_options"] = {"include_usage": True}
    try:
        with send(args.base_url, payload) as response:
            if args.stream:
                for line in response:
                    print(line.decode().rstrip(), flush=True)
            else:
                print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
    except urllib.error.HTTPError as error:
        parser.exit(1, f"HTTP {error.code}: {error.read().decode()}\n")


if __name__ == "__main__":
    main()
