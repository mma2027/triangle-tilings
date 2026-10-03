"""
viewer.py — Export tiles as a self-contained, zoomable HTML viewer.

The page draws on a canvas and, on every redraw, skips tiles that are off
screen or smaller than about a pixel, so a tiling with hundreds of thousands
of tiles stays responsive: zooming in reveals the small tiles near the
boundary. Hovering a tile shows its word; clicking copies it.

Data layout (base64 little-endian arrays, one entry per drawable tile)
---------------------------------------------------------------------
centroid  float64 × 2   tile centroid
offsets   float32 × 6   vertices (V_p, V_q, V_r) relative to the centroid
bulge     float32 × 3   per edge (c, a, b): signed sagitta / chord length,
                        0 for a straight edge — the arc midpoint is
                        (u + v)/2 + bulge · i(v − u)
parent    int32         index of the tile one letter shorter (-1 for root)
letter    uint8         last letter of the word (0 = a, 1 = b, 2 = c)

Words are rebuilt in the browser by walking parent links; tiles whose parent
wasn't drawn (spherical tiles next to ∞) carry their word explicitly.

Functions
---------
render_html(tiles, tri, out, color="alternate", labels=False, depth=None)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgb

from geometry import Triangle, arc_through, fmt_order
from group import Tile
import render

_TEMPLATE = Path(__file__).with_name("viewer_template.html")


def _bulge(u: complex, m: complex, v: complex) -> float:
    arc = arc_through(u, m, v)
    if arc is None:
        return 0.0
    c, r, tu, sweep = arc
    apex = c + r * complex(np.exp(1j * (tu + sweep / 2)))
    return ((apex - (u + v) / 2) / (1j * (v - u))).real


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()).decode()


def _palette(scheme: str, max_len: int) -> tuple[list[str], str]:
    """(colors indexed by color class, color for unresolved sub-pixel tiles)."""
    if scheme == "alternate":
        even, odd = render._EVEN, render._ODD
        return [even, odd], to_hex((np.array(to_rgb(even)) + np.array(to_rgb(odd))) / 2)
    if scheme == "wordlength":
        cmap = colormaps["viridis"]
        top = max(max_len, 1)
        colors = [to_hex(cmap(k / top)) for k in range(max_len + 1)]
        return colors, colors[-1]
    return [render._NONE_FILL], render._NONE_FILL


def render_html(
    tiles: list[Tile],
    tri: Triangle,
    out: str | Path,
    color: str = "alternate",
    labels: bool = False,
    depth: int | None = None,
) -> None:
    drawn = [t for t in tiles if t.drawable]
    n = len(drawn)
    index = {t.word: i for i, t in enumerate(drawn)}

    centroid = np.empty((n, 2), dtype=np.float64)
    offsets = np.empty((n, 6), dtype=np.float32)
    bulge = np.empty((n, 3), dtype=np.float32)
    parent = np.full(n, -1, dtype=np.int32)
    letter = np.zeros(n, dtype=np.uint8)
    explicit: dict[int, str] = {}

    for i, t in enumerate(drawn):
        c = t.centroid
        centroid[i] = (c.real, c.imag)
        rel = t.vertices - c
        offsets[i] = np.column_stack([rel.real, rel.imag]).ravel()
        bulge[i] = [_bulge(*e) for e in t.edges()]
        if t.word:
            letter[i] = "abc".index(t.word[-1])
            p = index.get(t.word[:-1])
            if p is None:
                explicit[i] = t.word
            else:
                parent[i] = p

    max_len = max((t.length for t in drawn), default=0)
    colors, unresolved = _palette(color, max_len)
    name = ",".join(fmt_order(k) for k in (tri.p, tri.q, tri.r))
    meta = {
        "title": f"Δ({name})",
        "kind": tri.kind,
        "count": n,
        "scheme": color,
        "colors": colors,
        "unresolved": unresolved,
        "edge": render._EDGE,
        "labels": labels,
        "viewRadius": render._view_radius(drawn, tri, depth),
        "boundary": 1.0 if tri.kind == "hyperbolic" else None,
        "explicit": {str(k): v for k, v in explicit.items()},
    }
    data = {
        "centroid": _b64(centroid),
        "offsets": _b64(offsets),
        "bulge": _b64(bulge),
        "parent": _b64(parent),
        "letter": _b64(letter),
    }
    html = _TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__TITLE__", f"{meta['title']} tiling")
    html = html.replace("/*__META__*/null", json.dumps(meta, ensure_ascii=False))
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    Path(out).write_text(html, encoding="utf-8")
