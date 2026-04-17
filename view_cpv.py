"""
View / convert a .cpv NVR export file.

Format detected in NVR_ch1_main_*.cpv:
  - 1064-byte proprietary "ADIT" header
  - Raw H.265 (HEVC) Annex-B bitstream follows
  - VPS (NAL 32) / SPS (33) / PPS (34) / IDR (19) / slices (1)

Usage:
    python view_cpv.py <file.cpv>                 # show info
    python view_cpv.py <file.cpv> --extract       # write raw .h265
    python view_cpv.py <file.cpv> --mp4           # remux to .mp4
    python view_cpv.py <file.cpv> --play          # play with ffplay
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ADIT_MAGIC = b"ADIT"
HEADER_SIZE = 1064  # observed offset where first real NAL (VPS) begins


def find_stream_offset(path: Path) -> int:
    """Locate the first real H.265 NAL unit (skip header padding)."""
    with path.open("rb") as f:
        data = f.read(4096)
    # real NAL = 0x00000001 followed by a non-zero byte
    m = re.search(rb"\x00\x00\x00\x01[^\x00]", data)
    if not m:
        raise RuntimeError("No H.265 start code found in first 4KB")
    return m.start()


def show_info(path: Path) -> None:
    size = path.stat().st_size
    with path.open("rb") as f:
        magic = f.read(4)
    offset = find_stream_offset(path)

    print(f"File        : {path.name}")
    print(f"Size        : {size:,} bytes ({size / 1024 / 1024:.1f} MiB)")
    print(f"Magic       : {magic!r}  {'(ADIT NVR container)' if magic == ADIT_MAGIC else '(unknown)'}")
    print(f"Header bytes: {offset}")
    print(f"Codec       : H.265 / HEVC (Annex-B)")
    print()
    # Ask ffprobe to parse the raw stream starting after the header
    print("--- ffprobe on raw stream ---")
    cmd = [
        "ffprobe", "-v", "error",
        "-f", "hevc",
        "-show_streams", "-show_format",
        "-i", "pipe:0",
    ]
    with path.open("rb") as f:
        f.seek(offset)
        subprocess.run(cmd, stdin=f, check=False)


def extract_raw(path: Path, out: Path) -> None:
    offset = find_stream_offset(path)
    with path.open("rb") as fin, out.open("wb") as fout:
        fin.seek(offset)
        while chunk := fin.read(1 << 20):
            fout.write(chunk)
    print(f"Wrote raw HEVC -> {out} ({out.stat().st_size:,} bytes)")


def remux_mp4(path: Path, out: Path, fps: int = 25) -> None:
    """Wrap the raw HEVC in an MP4 container so any player can open it."""
    offset = find_stream_offset(path)
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),            # assumed fps (NVR streams are usually 25)
        "-f", "hevc",
        "-i", "pipe:0",
        "-c:v", "copy",
        str(out),
    ]
    with path.open("rb") as f:
        f.seek(offset)
        subprocess.run(cmd, stdin=f, check=True)
    print(f"Wrote MP4 -> {out}")


def play(path: Path) -> None:
    offset = find_stream_offset(path)
    cmd = ["ffplay", "-f", "hevc", "-i", "pipe:0"]
    with path.open("rb") as f:
        f.seek(offset)
        subprocess.run(cmd, stdin=f, check=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--extract", action="store_true", help="write raw .h265 next to input")
    g.add_argument("--mp4", action="store_true", help="remux to .mp4 next to input")
    g.add_argument("--play", action="store_true", help="play via ffplay")
    p.add_argument("--fps", type=int, default=25, help="assumed fps for --mp4 (default 25)")
    args = p.parse_args()

    if not args.file.exists():
        print(f"not found: {args.file}", file=sys.stderr)
        return 1

    if args.extract:
        extract_raw(args.file, args.file.with_suffix(".h265"))
    elif args.mp4:
        remux_mp4(args.file, args.file.with_suffix(".mp4"), fps=args.fps)
    elif args.play:
        play(args.file)
    else:
        show_info(args.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
