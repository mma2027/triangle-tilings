"""
test_tiling.py — Numerical checks for the geometry and group generation.

Run with either:
    python -m unittest -v
    python -m pytest -q
"""

from __future__ import annotations

import math
import unittest

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

GROUPS = [(2, 3, 7), (3, 3, 4), (2, 4, 4), (2, 3, 5), (4, 5, 6), (3, 3, 3), (2, 3, 6), (2, 3, 4)]

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


class TestReflections(unittest.TestCase):
    def test_involutions(self):
        for pqr in GROUPS:
            tri = fundamental_triangle(*pqr)
            for name, s in tri.reflections.items():
                with self.subTest(pqr=pqr, s=name):
                    ss = s.compose(s)
                    self.assertTrue(ss.is_identity())
                    np.testing.assert_allclose(ss(SAMPLES), SAMPLES, atol=1e-10)

    def test_reflections_fix_their_side(self):
        for pqr in GROUPS:
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


class TestFundamentalTriangle(unittest.TestCase):
    def test_angles(self):
        for pqr in GROUPS:
            tri = fundamental_triangle(*pqr)
            with self.subTest(pqr=pqr):
                np.testing.assert_allclose(_measured_vertex_angles(tri), tri.angles, atol=1e-9)

    def test_hyperbolic_side_is_orthogonal_to_unit_circle(self):
        for pqr in GROUPS:
            tri = fundamental_triangle(*pqr)
            if tri.kind != HYPERBOLIC:
                continue
            a = tri.sides["a"]
            with self.subTest(pqr=pqr):
                self.assertAlmostEqual(abs(a.center) ** 2, a.radius ** 2 + 1, places=10)
                self.assertTrue(all(abs(v) < 1 for v in tri.vertices))

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


if __name__ == "__main__":
    unittest.main()
