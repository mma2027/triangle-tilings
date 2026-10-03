"""
render.py — Draw a list of tiles to SVG/PNG with matplotlib.

Every tile edge is drawn as a true circular arc (or a straight segment when
the geodesic is a line). In the Poincaré disk those arcs are orthogonal to the
unit circle; in the spherical case they are stereographic images of great
circles.

Functions
---------
render(tiles, tri, out, color="alternate", labels=False, fmt="svg", depth=None)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Circle, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from geometry import EUCLIDEAN, HYPERBOLIC, SPHERICAL, Triangle, arc_through, fmt_order  # noqa: E402
from group import Tile  # noqa: E402

COLOR_SCHEMES = ("alternate", "wordlength", "none")

_EVEN = "#1f2a44"
_ODD = "#f4efe6"
_EDGE = "#11151f"
_NONE_FILL = "#ffffff"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _edge_segments(u: complex, m: complex, v: complex) -> tuple[list, list]:
    """Path vertices/codes from u (exclusive) to v (inclusive) along the arc u → m → v."""
    arc = arc_through(u, m, v)
    if arc is None:
        return [(v.real, v.imag)], [MplPath.LINETO]
    c, r, tu, sweep = arc
    t0 = math.degrees(tu if sweep > 0 else tu + sweep)
    unit = MplPath.arc(t0, t0 + math.degrees(abs(sweep)))
    pts = [(c.real + r * x, c.imag + r * y) for x, y in unit.vertices]
    if sweep < 0:
        pts.reverse()
    return pts[1:], [MplPath.CURVE4] * (len(pts) - 1)


def tile_path(tile: Tile) -> MplPath:
    v0 = tile.vertices[0]
    verts, codes = [(v0.real, v0.imag)], [MplPath.MOVETO]
    for edge in tile.edges():
        pts, cs = _edge_segments(*edge)
        verts += pts
        codes += cs
    verts.append(verts[0])
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(verts, codes)


# ---------------------------------------------------------------------------
# Colors / framing
# ---------------------------------------------------------------------------

def _face_colors(tiles: list[Tile], scheme: str) -> list:
    if scheme == "alternate":
        return [_EVEN if t.length % 2 == 0 else _ODD for t in tiles]
    if scheme == "wordlength":
        cmap = plt.get_cmap("viridis")
        top = max(t.length for t in tiles) or 1
        return [cmap(t.length / top) for t in tiles]
    return [_NONE_FILL] * len(tiles)


def _view_radius(tiles: list[Tile], tri: Triangle, depth: int | None) -> float:
    if tri.kind != EUCLIDEAN:
        return tri.view_radius
    # Largest origin-centered disk fully covered by the BFS ball.
    max_len = depth if depth is not None else max(t.length for t in tiles)
    frontier = [abs(t.centroid) - t.size for t in tiles if t.length >= max_len]
    if not frontier:
        return max(abs(z) for t in tiles for z in t.vertices)
    return max(min(frontier), 1.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render(
    tiles: list[Tile],
    tri: Triangle,
    out: str | Path,
    color: str = "alternate",
    labels: bool = False,
    fmt: str = "svg",
    depth: int | None = None,
    dpi: int = 200,
) -> None:
    drawn = [t for t in tiles if t.drawable]
    radius = _view_radius(drawn, tri, depth)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 1.03 * radius
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    clip = Circle((0, 0), radius, transform=ax.transData)

    # Thin outlines that shrink with the tile so the disk boundary stays clean.
    widths = [min(0.6, max(0.05, 4.0 * t.size / radius)) for t in drawn]
    patches = [PathPatch(tile_path(t)) for t in drawn]
    coll = PatchCollection(
        patches,
        facecolors=_face_colors(drawn, color),
        edgecolors=_EDGE,
        linewidths=widths,
        joinstyle="round",
    )
    coll.set_clip_path(clip)
    ax.add_collection(coll)

    # Unit circle in the disk model; the clip circle in the other two.
    boundary_r = 1.0 if tri.kind == HYPERBOLIC else radius
    ax.add_patch(Circle((0, 0), boundary_r, fill=False, lw=0.8, ec=_EDGE))
    if tri.kind == SPHERICAL and radius > 1:
        # Equator of the sphere = unit circle under stereographic projection.
        ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=0.6, ec="#c0392b", ls="--"))

    if labels:
        for t in drawn:
            c = t.centroid
            fs = min(9.0, 150 * t.size / radius / max(1, len(t.label)) ** 0.5)
            if fs < 2.5 or abs(c) > radius:
                continue
            txt_color = _ODD if (color == "alternate" and t.length % 2 == 0) else _EDGE
            ax.text(c.real, c.imag, t.label, ha="center", va="center",
                    fontsize=fs, color=txt_color, family="monospace")

    name = ",".join(fmt_order(n) for n in (tri.p, tri.q, tri.r))
    ax.set_title(f"Δ({name}) — {tri.kind}, {len(drawn)} tiles",
                 fontsize=11, color=_EDGE)

    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
