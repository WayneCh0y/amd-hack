"""Stitch the rendered slide scenes into one continuous MP4.

`manim-slides convert` only targets HTML / PDF / PPTX — there is no MP4 writer —
so a flat video has to be assembled from the per-scene files that `manim` emits.

manim-slides ships `utils.concatenate_video_files`, but it feeds ffmpeg's concat
demuxer a Windows path with backslashes and dies with a spurious FileNotFoundError.
This does the same remux with POSIX paths, via the PyAV that manim already bundles
(no standalone ffmpeg needed).

Usage (after `manim -qh --media_dir presentation/media presentation/slides.py <scenes>`):

    python presentation/build_video.py               # every scene, deck order
    python presentation/build_video.py Title Results # or an explicit subset
"""

from __future__ import annotations

import sys
from pathlib import Path

import av

HERE = Path(__file__).parent
RENDER_DIR = HERE / "media" / "videos" / "slides" / "1080p60"
DEST = HERE / "final.mp4"

# The deck, in presentation order. Keep in sync with the scenes in slides.py.
DECK = ["Title", "Objective", "Architecture", "Iteration", "Results", "Closing"]


def concatenate(files: list[Path], dest: Path) -> None:
    listing = dest.parent / "concat_list.txt"
    listing.write_text("".join(f"file '{f.resolve().as_posix()}'\n" for f in files))

    with (
        av.open(str(listing.as_posix()), format="concat", options={"safe": "0"}) as src,
        av.open(str(dest), mode="w") as out,
    ):
        stream = out.add_stream(template=src.streams.video[0])
        for packet in src.demux():
            if packet.dts is None:      # flush packet, nothing to write
                continue
            packet.stream = stream
            out.mux(packet)

    listing.unlink()


def main() -> int:
    scenes = sys.argv[1:] or DECK
    files = [RENDER_DIR / f"{s}.mp4" for s in scenes]

    missing = [f for f in files if not f.exists()]
    if missing:
        print("Missing rendered scenes (render them first):", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        return 1

    concatenate(files, DEST)
    print(f"Wrote {DEST} ({DEST.stat().st_size / 1e6:.1f} MB) from: {', '.join(scenes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
