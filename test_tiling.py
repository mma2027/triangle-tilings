"""
test_tiling.py — Numerical checks for the geometry and group generation.

Run with either:
    python -m unittest -v
    python -m pytest -q
"""

from __future__ import annotations

import base64
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geometry import (
    EUCLIDEAN,
    HYPERBOLIC,
    SPHERICAL,
    AntiMobius,
    _measured_vertex_angles,
    classify,
    fundamental_triangle,
)
from group import generate_tiles
from viewer import render_html

GROUPS = [(2, 3, 7), (3, 3, 4), (2, 4, 4), (2, 3, 5), (4, 5, 6), (3, 3, 3), (2, 3, 6), (2, 3, 4)]
# 0 = ∞. Covers ∞ in every position (exercises the relabelling) and the ideal triangle.
INF_GROUPS = [(2, 3, 0), (0, 2, 3), (3, 0, 2), (2, 0, 0), (0, 0, 4), (0, 0, 0)]
ALL_GROUPS = GROUPS + INF_GROUPS

_rng = np.random.default_rng(0)
SAMPLES = 0.4 * (_rng.uniform(-1, 1, 20) + 1j * _rng.uniform(-1, 1, 20))


def power(g: AntiMobius, n: int) -> AntiMobius:
    out = AntiMobius.identity()
    for _ in range(n):
        out = out.compose(g).normalized()
    return out


class TestClassify(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(classify(2, 3, 7), HYPERBOLIC)
        self.assertEqual(classify(3, 3, 4), HYPERBOLIC)
        self.assertEqual(classify(2, 4, 4), EUCLIDEAN)
        self.assertEqual(classify(3, 3, 3), EUCLIDEAN)
        self.assertEqual(classify(2, 3, 5), SPHERICAL)
        self.assertEqual(classify(2, 3, 0), HYPERBOLIC)
        self.assertEqual(classify(0, 0, 0), HYPERBOLIC)


class TestReflections(unittest.TestCase):
    def test_involutions(self):
        for pqr in ALL_GROUPS:
            tri = fundamental_triangle(*pqr)
            for name, s in tri.reflections.items():
                with self.subTest(pqr=pqr, s=name):
                    ss = s.compose(s)
                    self.assertTrue(ss.is_identity())
                    np.testing.assert_allclose(ss(SAMPLES), SAMPLES, atol=1e-10)

    def test_reflections_fix_their_side(self):
        for pqr in ALL_GROUPS:
            tri = fundamental_triangle(*pqr)
            vp, vq, vr = tri.vertices
            on_side = {"a": [vq, vr, tri.side_midpoints["a"]],
                       "b": [vp, vr, tri.side_midpoints["b"]],
                       "c": [vp, vq, tri.side_midpoints["c"]]}
            for name, pts in on_side.items():
                with self.subTest(pqr=pqr, s=name):
                    pts = np.array(pts)
                    np.testing.assert_allclose(tri.reflections[name](pts), pts, atol=1e-10)

    def test_coxeter_relations(self):
        """(ab)^r = (bc)^p = (ca)^q = identity."""
        for p, q, r in GROUPS:
            tri = fundamental_triangle(p, q, r)
            a, b, c = (tri.reflections[s] for s in "abc")
            for label, pair, n in (("ab", a.compose(b), r),
                                   ("bc", b.compose(c), p),
                                   ("ca", c.compose(a), q)):
                with self.subTest(pqr=(p, q, r), rel=f"({label})^{n}"):
                    g = power(pair, n)
                    self.assertTrue(g.is_identity(tol=1e-9))
                    np.testing.assert_allclose(g(SAMPLES), SAMPLES, atol=1e-9)
                    # ...and the order is exactly n, not a proper divisor.
                    for k in range(1, n):
                        self.assertFalse(power(pair, k).is_identity(tol=1e-6))


    def test_infinite_order_pairs_are_parabolic(self):
        """For an ∞ vertex the product of its two reflections is parabolic."""
        for p, q, r in INF_GROUPS:
            tri = fundamental_triangle(p, q, r)
            a, b, c = (tri.reflections[s] for s in "abc")
            for label, pair, n in (("ab", a.compose(b), r),
                                   ("bc", b.compose(c), p),
                                   ("ca", c.compose(a), q)):
                with self.subTest(pqr=(p, q, r), pair=label):
                    g = pair.normalized()
                    self.assertFalse(g.conj)
                    if n == 0:
                        self.assertAlmostEqual(abs(np.trace(g.m)) ** 2, 4.0, places=9)
                        for k in range(1, 30):
                            self.assertFalse(power(pair, k).is_identity(tol=1e-6))
                    else:
                        self.assertTrue(power(pair, n).is_identity(tol=1e-9))


class TestFundamentalTriangle(unittest.TestCase):
    def test_angles(self):
        for pqr in ALL_GROUPS:
            tri = fundamental_triangle(*pqr)
            with self.subTest(pqr=pqr):
                np.testing.assert_allclose(_measured_vertex_angles(tri), tri.angles, atol=1e-7)

    def test_hyperbolic_side_is_orthogonal_to_unit_circle(self):
        for pqr in ALL_GROUPS:
            tri = fundamental_triangle(*pqr)
            if tri.kind != HYPERBOLIC:
                continue
            with self.subTest(pqr=pqr):
                # Lines through the origin are diameters; every circle side must
                # be orthogonal to the boundary.
                for side in tri.sides.values():
                    if side.kind == "circle":
                        self.assertAlmostEqual(abs(side.center) ** 2, side.radius ** 2 + 1, places=10)
                    else:
                        self.assertAlmostEqual(side.signed(0j), 0.0, places=12)
                for n, v in zip(pqr, tri.vertices):
                    if n == 0:
                        self.assertAlmostEqual(abs(v), 1.0, places=12)   # ideal
                    else:
                        self.assertLess(abs(v), 1.0)

    def test_spherical_side_is_great_circle(self):
        a = fundamental_triangle(2, 3, 5).sides["a"]
        self.assertAlmostEqual(abs(a.center) ** 2, a.radius ** 2 - 1, places=10)


class TestGeneration(unittest.TestCase):
    def test_spherical_tile_count(self):
        """Δ(2,3,5) is the full icosahedral group of order 120."""
        tri = fundamental_triangle(2, 3, 5)
        tiles = generate_tiles(tri, depth=30, min_size=0)
        self.assertEqual(len(tiles), 120)
        self.assertEqual(max(t.length for t in tiles), 15)   # longest element

    def test_words_unique_and_reduced(self):
        tri = fundamental_triangle(2, 3, 7)
        tiles = generate_tiles(tri, depth=10)
        words = [t.word for t in tiles]
        self.assertEqual(len(words), len(set(words)))
        for w in words:
            self.assertFalse(any(x == y for x, y in zip(w, w[1:])))

    def test_cusp_groups_generate(self):
        for pqr in INF_GROUPS:
            tri = fundamental_triangle(*pqr)
            tiles = generate_tiles(tri, depth=10)
            with self.subTest(pqr=pqr):
                self.assertGreater(len(tiles), 50)
                for t in tiles:
                    self.assertTrue(np.all(np.abs(t.vertices) <= 1 + 1e-9))

    def test_tiles_match_their_words(self):
        """Recomposing each word from scratch reproduces the stored tile."""
        tri = fundamental_triangle(3, 3, 4)
        for t in generate_tiles(tri, depth=7):
            g = AntiMobius.identity()
            for s in t.word:
                g = g.compose(tri.reflections[s])
            np.testing.assert_allclose(g(np.array(tri.vertices)), t.vertices, atol=1e-9)

    def test_neighbors_share_an_edge(self):
        tri = fundamental_triangle(2, 4, 4)
        by_word = {t.word: t for t in generate_tiles(tri, depth=6)}
        for w, t in by_word.items():
            if not w:
                continue
            parent = by_word.get(w[:-1])
            if parent is None:
                continue
            shared = sum(np.isclose(t.vertices, z).any() for z in parent.vertices)
            self.assertEqual(shared, 2, msg=w)


class TestViewer(unittest.TestCase):
    def _export(self, pqr, depth):
        tri = fundamental_triangle(*pqr)
        tiles = [t for t in generate_tiles(tri, depth) if t.drawable]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "v.html"
            render_html(tiles, tri, out)
            html = out.read_text(encoding="utf-8")
        meta = json.loads(re.search(r"const META = (.*);", html).group(1))
        data = json.loads(re.search(r"const DATA = (.*);", html).group(1))
        arr = {k: np.frombuffer(base64.b64decode(v), dtype=dt) for (k, v), dt in zip(
            data.items(), ("<f8", "<f4", "<f4", "<i4", "u1"))}
        return tiles, meta, arr

    def test_embedded_data_round_trips(self):
        """Words rebuilt from parent links and vertices from centroid + offsets match."""
        for pqr, depth in (((2, 3, 7), 12), ((2, 3, 0), 10), ((2, 3, 5), 15)):
            tiles, meta, arr = self._export(pqr, depth)
            with self.subTest(pqr=pqr):
                self.assertEqual(meta["count"], len(tiles))
                words = []
                for i in range(len(tiles)):
                    if str(i) in meta["explicit"]:
                        words.append(meta["explicit"][str(i)])
                    elif arr["parent"][i] < 0:
                        words.append("")
                    else:
                        words.append(words[arr["parent"][i]] + "abc"[arr["letter"][i]])
                self.assertEqual(words, [t.word for t in tiles])
                cen = arr["centroid"].reshape(-1, 2)
                off = arr["offsets"].reshape(-1, 3, 2)
                verts = (cen[:, None, 0] + off[:, :, 0]) + 1j * (cen[:, None, 1] + off[:, :, 1])
                expected = np.array([t.vertices for t in tiles])
                np.testing.assert_allclose(verts, expected, atol=1e-6)

    def test_bulge_reproduces_arc_midpoint(self):
        """The arc apex rebuilt from the stored bulge lies on the true geodesic circle."""
        from geometry import arc_through
        tiles, _, arr = self._export((3, 3, 4), 6)
        bulge = arr["bulge"].reshape(-1, 3)
        for t, bs in zip(tiles, bulge):
            for (u, m, v), b in zip(t.edges(), bs):
                apex = (u + v) / 2 + float(b) * 1j * (v - u)
                arc = arc_through(u, m, v)
                if arc is None:
                    self.assertAlmostEqual(float(b), 0.0, places=6)
                else:
                    c, r, _, _ = arc
                    self.assertAlmostEqual(abs(apex - c), r, delta=1e-6 * max(1.0, r))


if __name__ == "__main__":
    unittest.main()
