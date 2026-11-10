"""Generate small deterministic videos for temporal-order and cache checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

COLORS = {"RED": (230, 20, 20), "GREEN": (20, 210, 20), "BLUE": (20, 20, 230)}


def write_video(path, sequence, frames_each=4):
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 224, 224, "yuv420p"
        for name in sequence:
            for index in range(frames_each):
                image = Image.new("RGB", (224, 224), COLORS[name])
                draw = ImageDraw.Draw(image)
                x = 20 + index * 40
                draw.rectangle((x, 150, x + 20, 170), fill="white")
                frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = {
        "forward.mp4": ["RED", "GREEN", "BLUE"],
        "reverse.mp4": ["BLUE", "GREEN", "RED"],
    }
    for name, sequence in sequences.items():
        write_video(args.output / name, sequence)
    for count in (1, 2, 3, 4):
        write_video(args.output / f"tail-{count}.mp4", ["RED"], frames_each=count)
    image = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(image).polygon(
        [(112, 30), (30, 190), (194, 190)], fill=(245, 210, 20)
    )
    image.save(args.output / "yellow-triangle.png")
    (args.output / "expected.json").write_text(json.dumps(sequences, indent=2) + "\n")
    print(f"Wrote deterministic fixtures to {args.output}")


if __name__ == "__main__":
    main()
