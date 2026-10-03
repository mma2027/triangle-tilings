"""
tiling.py — CLI entry point for triangle-tilings.

Draws the tiling of the plane, disk, or sphere by the triangle group Δ(p,q,r)
and exports {word: vertex coordinates} as JSON alongside the image.

Examples
--------
python tiling.py --pqr 2 3 7                    # → data/2_3_7/d40_m0.001.png + .json
python tiling.py --pqr 2 3 7 --svg              # … + .svg
python tiling.py --pqr 2 3 7 --depth 50 --min-size 1e-4 --html   # → data/2_3_7/d50_m0.0001.*
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from geometry import fmt_order, fundamental_triangle
from group import generate_tiles, tiles_to_json
from render import COLOR_SCHEMES, render
from viewer import render_html

EXTRA_FORMATS = ("svg", "html")
DATA_DIR = Path(__file__).resolve().parent / "data"   # always inside the project


def order(value: str) -> int:
    """argparse type for p, q, r: an integer ≥ 2, or 0 / inf / ∞ for ∞."""
    if value.lower() in ("0", "inf", "∞"):
        return 0
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}")
    if n < 2:
        raise argparse.ArgumentTypeError(f"must be ≥ 2, or 0 for ∞ (got {n})")
    return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Draw triangle tilings for Δ(p,q,r).")
    ap.add_argument("--pqr", nargs=3, type=order, required=True, metavar=("P", "Q", "R"),
                    help="angles of the fundamental triangle are π/p, π/q, π/r; 0 = ∞")
    ap.add_argument("--depth", type=int, default=40, help="max word length (default 40)")
    ap.add_argument("--color", choices=COLOR_SCHEMES, default="alternate",
                    help="tile coloring (default alternate = word-length parity)")
    ap.add_argument("--labels", action="store_true", help="print group words on tiles")
    ap.add_argument("--svg", action="store_true", help="also write an SVG")
    ap.add_argument("--html", action="store_true", help="also write a zoomable HTML viewer")
    ap.add_argument("--out", type=Path, default=None,
                    help="base output path, overriding the default "
                         "data/<group>/d<depth>_m<min-size>[...]; a PNG is always written, "
                         "--svg / --html add files with the same name")
    ap.add_argument("--json", type=Path, default=None,
                    help="word → vertices JSON path (default: same name as the PNG, .json)")
    ap.add_argument("--min-size", type=float, default=1e-3,
                    help="skip tiles with Euclidean diameter below this (default 1e-3)")
    return ap.parse_args(argv)


def write_json(data: dict, path: Path) -> None:
    """One word per line — diff-friendly and still valid JSON."""
    lines = [f"  {json.dumps(w)}: {json.dumps(v)}" for w, v in data.items()]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def display(path: Path) -> str:
    """Relative path when under the current directory, else absolute."""
    rel = os.path.relpath(path)
    return str(path.resolve()) if rel.startswith("..") else rel


def with_ext(base: Path, ext: str) -> Path:
    # Not Path.with_suffix: names like "d40_m0.001" already contain a dot.
    return base.parent / f"{base.name}.{ext}"


def output_base(args: argparse.Namespace) -> Path:
    """
    Path (without extension) shared by every output of this run.

    Default: data/<group>/<settings>, e.g. data/2_3_7/d40_m0.001 or
    data/2_3_inf/d30_m0.0005_wordlength_labels. --out overrides it; a known
    suffix on --out is dropped (and .svg / .html switch that format on).
    """
    if args.out is not None:
        ext = args.out.suffix.lstrip(".").lower()
        return args.out.with_suffix("") if ext in ("png", "json", *EXTRA_FORMATS) else args.out
    group = "_".join("inf" if n == 0 else str(n) for n in args.pqr)
    parts = [f"d{args.depth}", f"m{args.min_size:g}"]
    if args.color != "alternate":
        parts.append(args.color)
    if args.labels:
        parts.append("labels")
    return DATA_DIR / group / "_".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    p, q, r = args.pqr
    name = ",".join(fmt_order(n) for n in (p, q, r))

    base = output_base(args)
    ext = args.out.suffix.lstrip(".").lower() if args.out else ""
    extras = [f for f in EXTRA_FORMATS if getattr(args, f) or ext == f]
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.json or with_ext(base, "json")

    t0 = time.perf_counter()
    tri = fundamental_triangle(p, q, r)
    tiles = generate_tiles(tri, args.depth, args.min_size)
    style = dict(color=args.color, labels=args.labels, depth=args.depth)
    outputs = [(f, with_ext(base, f)) for f in ("png", *extras)]
    for kind, path in outputs:
        if kind == "html":
            render_html(tiles, tri, path, **style)
        else:
            render(tiles, tri, path, fmt=kind, **style)
    write_json(tiles_to_json(tiles), json_path)

    drawn = sum(t.drawable for t in tiles)
    print(f"Δ({name}) [{tri.kind}]  depth={args.depth}  "
          f"tiles={len(tiles)} (drawn {drawn})  {time.perf_counter() - t0:.2f}s")
    for kind, path in outputs:
        print(f"  {kind:<5} → {display(path)}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  json  → {display(json_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
