"""
tiling.py — CLI entry point for triangle-tilings.

Draws the tiling of the plane, disk, or sphere by the triangle group Δ(p,q,r)
and exports {word: vertex coordinates} as JSON alongside the image.

Example
-------
python tiling.py --pqr 2 3 7 --depth 12 --out tiling.svg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from geometry import fundamental_triangle
from group import generate_tiles, tiles_to_json
from render import COLOR_SCHEMES, render


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Draw triangle tilings for Δ(p,q,r).")
    ap.add_argument("--pqr", nargs=3, type=int, required=True, metavar=("P", "Q", "R"),
                    help="angles of the fundamental triangle are π/p, π/q, π/r")
    ap.add_argument("--depth", type=int, default=12, help="max word length (default 12)")
    ap.add_argument("--color", choices=COLOR_SCHEMES, default="alternate",
                    help="tile coloring (default alternate = word-length parity)")
    ap.add_argument("--labels", action="store_true", help="print group words on tiles")
    ap.add_argument("--format", choices=("svg", "png"), default=None,
                    help="output format (default: from --out suffix, else svg)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output image path (default tiling_P_Q_R.<format>)")
    ap.add_argument("--json", type=Path, default=None,
                    help="word → vertices JSON path (default: --out with .json suffix)")
    ap.add_argument("--min-size", type=float, default=1e-3,
                    help="skip tiles with Euclidean diameter below this (default 1e-3)")
    return ap.parse_args(argv)


def write_json(data: dict, path: Path) -> None:
    """One word per line — diff-friendly and still valid JSON."""
    lines = [f"  {json.dumps(w)}: {json.dumps(v)}" for w, v in data.items()]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    p, q, r = args.pqr
    if min(p, q, r) < 2:
        print("error: p, q, r must all be ≥ 2", file=sys.stderr)
        return 2

    fmt = args.format
    if fmt is None:
        suffix = args.out.suffix.lstrip(".").lower() if args.out else ""
        fmt = suffix if suffix in ("svg", "png") else "svg"
    out = args.out or Path(f"tiling_{p}_{q}_{r}.{fmt}")
    if out.suffix.lstrip(".").lower() != fmt:
        out = out.with_suffix(f".{fmt}")
    json_path = args.json or out.with_suffix(".json")

    t0 = time.perf_counter()
    tri = fundamental_triangle(p, q, r)
    tiles = generate_tiles(tri, args.depth, args.min_size)
    render(tiles, tri, out, color=args.color, labels=args.labels, fmt=fmt, depth=args.depth)
    write_json(tiles_to_json(tiles), json_path)

    drawn = sum(t.drawable for t in tiles)
    print(f"Δ({p},{q},{r}) [{tri.kind}]  depth={args.depth}  "
          f"tiles={len(tiles)} (drawn {drawn})  {time.perf_counter() - t0:.2f}s")
    print(f"  image → {out}")
    print(f"  json  → {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
