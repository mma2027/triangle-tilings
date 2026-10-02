"""
group.py — Enumerate tiles of Δ(p,q,r) by BFS over reduced words in a, b, c.

Word convention
---------------
The word "abc" is the group element g = a∘b∘c, and its tile is g(T). The tile
across side s of g(T) is (g s)(T), so BFS extends words on the right. Words
never repeat a letter back-to-back (s² = 1). Each tile is kept once, under the
first (shortest) word that reaches it; duplicates are detected by rounding the
tile's centroid.

Functions
---------
generate_tiles(tri, depth, min_size=1e-3)  → list[Tile]
tiles_to_json(tiles)                       → dict[str, list[[x, y] | None]]
"""

from __future__ import annotations

import cmath
from collections import deque
from dataclasses import dataclass

import numpy as np

from geometry import SPHERICAL, AntiMobius, Triangle

IDENTITY_WORD = "e"
_KEY_DECIMALS = 8


@dataclass
class Tile:
    word: str                # "" for the identity
    g: AntiMobius
    vertices: np.ndarray     # images of (V_p, V_q, V_r)
    midpoints: np.ndarray    # images of the side midpoints (c, a, b order)
    drawable: bool = True    # False if the tile touches ∞ (spherical only)

    @property
    def length(self) -> int:
        return len(self.word)

    @property
    def label(self) -> str:
        return self.word or IDENTITY_WORD

    @property
    def centroid(self) -> complex:
        return complex(self.vertices.mean())

    @property
    def size(self) -> float:
        """Euclidean diameter of the vertex set."""
        v = self.vertices
        return float(max(abs(v[0] - v[1]), abs(v[1] - v[2]), abs(v[2] - v[0])))

    def edges(self) -> list[tuple[complex, complex, complex]]:
        v, m = self.vertices, self.midpoints
        return [(v[0], m[0], v[1]), (v[1], m[1], v[2]), (v[2], m[2], v[0])]


def _make_tile(tri: Triangle, word: str, g: AntiMobius) -> Tile:
    vertices = g(np.array(tri.vertices))
    midpoints = g(np.array([tri.side_midpoints[s] for s in "cab"]))
    drawable = bool(np.all(np.isfinite(vertices)) and np.all(np.isfinite(midpoints)))
    if drawable and tri.kind == SPHERICAL:
        # g(T) contains ∞ iff g⁻¹(∞) ∈ T; such a tile can't be drawn as a bounded region.
        drawable = not tri.contains(g.preimage_of_infinity())
    return Tile(word, g, vertices, midpoints, drawable)


def _to_sphere(z: complex) -> np.ndarray:
    """Inverse stereographic projection onto the unit sphere (∞ ↦ north pole)."""
    if not cmath.isfinite(z) or abs(z) > 1e12:
        return np.array([0.0, 0.0, 1.0])
    n = abs(z) ** 2
    return np.array([2 * z.real, 2 * z.imag, n - 1]) / (n + 1)


def _key(tri: Triangle, tile: Tile) -> tuple:
    if tri.kind == SPHERICAL:
        # Plane coordinates blow up near ∞, so take the centroid on the sphere.
        c = sum(_to_sphere(complex(z)) for z in tile.vertices) / 3
        return tuple(round(float(x), _KEY_DECIMALS) for x in c)
    c = tile.centroid
    return (round(c.real, _KEY_DECIMALS), round(c.imag, _KEY_DECIMALS))


def generate_tiles(tri: Triangle, depth: int, min_size: float = 1e-3) -> list[Tile]:
    """
    BFS over reduced words up to length `depth`.

    Tiles whose Euclidean size falls below `min_size` are dropped and not
    expanded (in the disk they only get smaller toward the boundary).
    """
    start = _make_tile(tri, "", AntiMobius.identity())
    tiles = [start]
    seen = {_key(tri, start)}
    queue = deque([start])

    while queue:
        tile = queue.popleft()
        if tile.length >= depth:
            continue
        for s in "abc":
            if tile.word.endswith(s):
                continue
            g = tile.g.compose(tri.reflections[s]).normalized()
            child = _make_tile(tri, tile.word + s, g)
            key = _key(tri, child)
            if key in seen:
                continue
            seen.add(key)
            if child.drawable and child.size < min_size:
                continue
            tiles.append(child)
            queue.append(child)
    return tiles


def tiles_to_json(tiles: list[Tile]) -> dict[str, list]:
    """{word: [[x, y], [x, y], [x, y]]} with vertices in (V_p, V_q, V_r) order; ∞ → null."""
    out = {}
    for t in tiles:
        out[t.label] = [
            [float(z.real), float(z.imag)] if cmath.isfinite(z) else None
            for z in t.vertices
        ]
    return out
