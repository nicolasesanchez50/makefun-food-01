#!/usr/bin/env python3
"""make.fun food-picture-01 entry: one picture of food, written to /output/entry.png.

The picture itself is `art/entry.png`, a deterministic procedural render of a
bowl of tonkotsu ramen drawn by `tools/draw_food.py` (numpy + Pillow, fixed
seed, no network, no external assets).

This program is the entry point the harness runs: it re-validates the shipped
PNG (signature, every chunk CRC, the 1024x1024 cap and the 1 MiB cap) and then
writes those exact bytes to the output directory. Copying, rather than drawing
again, keeps the two harness runs byte-identical: the same file in, the same
file out, twice.
"""
from __future__ import annotations

import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "art", "entry.png")

PNG_SIG = b"\x89PNG\r\n\x1a\n"
MAX_BYTES = 1024 * 1024
MAX_DIM = 1024


class EntryError(Exception):
    """The shipped picture fails a rule the competition checks."""


def validate_png(data: bytes) -> tuple[int, int]:
    """Return (width, height) after checking the signature and every chunk CRC."""
    if not data.startswith(PNG_SIG):
        raise EntryError("art/entry.png is not a PNG (bad signature)")
    if len(data) > MAX_BYTES:
        raise EntryError(f"art/entry.png is {len(data)} bytes, over the {MAX_BYTES} cap")

    pos = len(PNG_SIG)
    width = height = None
    first = True
    while pos < len(data):
        if pos + 8 > len(data):
            raise EntryError("truncated chunk header at the end of the file")
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        body_start = pos + 8
        body_end = body_start + length
        if body_end + 4 > len(data):
            raise EntryError(f"chunk {ctype!r} runs past the end of the file")
        body = data[body_start:body_end]
        stored = struct.unpack(">I", data[body_end:body_end + 4])[0]
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != stored:
            raise EntryError(f"chunk {ctype!r} fails its CRC check")
        if first:
            if ctype != b"IHDR":
                raise EntryError("the first chunk is not IHDR")
            width, height = struct.unpack(">II", body[:8])
            first = False
        pos = body_end + 4
        if ctype == b"IEND":
            break

    if width is None or height is None:
        raise EntryError("no IHDR chunk: cannot read the picture's size")
    if width > MAX_DIM or height > MAX_DIM:
        raise EntryError(f"art/entry.png is {width}x{height}, over {MAX_DIM}x{MAX_DIM}")
    if width <= 0 or height <= 0:
        raise EntryError(f"art/entry.png has an empty size ({width}x{height})")
    return width, height


def output_dir() -> str:
    """The harness mounts /output; MAKEFUN_OUTPUT_DIR stands in when run by hand."""
    return os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"


def write_entry(out_dir: str, data: bytes) -> str:
    """Write the bytes, then flush and fsync so the harness never reads a short file."""
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "entry.png")
    with open(target, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.getsize(target) != len(data):
        raise EntryError(f"{target} is {os.path.getsize(target)} bytes, expected {len(data)}")
    return target


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    try:
        with open(ART, "rb") as fh:
            data = fh.read()
        width, height = validate_png(data)
        if check_only:
            print(f"ok: art/entry.png is a valid {width}x{height} PNG, {len(data)} bytes")
            return 0
        target = write_entry(output_dir(), data)
    except (OSError, EntryError) as exc:
        print(f"entry invalid: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {target} ({len(data)} bytes, {width}x{height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
