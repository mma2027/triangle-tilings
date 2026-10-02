"""
geometry.py — Möbius / anti-Möbius maps and the fundamental triangle of Δ(p,q,r).

Conventions
-----------
- Everything lives in the complex plane. Hyperbolic tilings use the Poincaré
  disk, spherical tilings use stereographic projection (the unit circle is the
  equator), Euclidean tilings use the plane directly.
- A group element is an AntiMobius map  z ↦ M·z  or  z ↦ M·conj(z),
  where M is a 2×2 complex matrix acting by linear fractional transformation.
- Fundamental triangle T has vertices
      V_p = 0            (angle π/p, between sides b and c)
      V_q on the real axis (angle π/q, between sides c and a)
      V_r on the ray at angle π/p (angle π/r, between sides a and b)
  so the three reflections satisfy (ab)^r = (bc)^p = (ca)^q = 1.
- Side c is the real axis, side b is the line at angle π/p, side a is the
  geodesic through V_q and V_r (a circle orthogonal to the unit circle in the
  hyperbolic case, a great circle in the spherical case, a line otherwise).

Functions
---------
classify(p, q, r)                 → "hyperbolic" | "euclidean" | "spherical"
fundamental_triangle(p, q, r)     → Triangle
circle_through(z1, z2, z3)        → (center, radius) | None
arc_through(u, m, v)              → (center, radius, theta_u, sweep) | None
arc_tangent(u, m, v)              → unit tangent at u of the arc u → m → v
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import numpy as np

HYPERBOLIC = "hyperbolic"
EUCLIDEAN = "euclidean"
SPHERICAL = "spherical"

_EPS = 1e-12


# ---------------------------------------------------------------------------
# (Anti-)Möbius transformations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AntiMobius:
    """z ↦ (a w + b) / (c w + d) with w = conj(z) if `conj` else z."""

    m: np.ndarray
    conj: bool = False

    @staticmethod
    def identity() -> "AntiMobius":
        return AntiMobius(np.eye(2, dtype=complex), False)

    def __call__(self, z):
        w = np.conj(z) if self.conj else np.asarray(z, dtype=complex)
        (a, b), (c, d) = self.m
        with np.errstate(divide="ignore", invalid="ignore"):
            return (a * w + b) / (c * w + d)

    def compose(self, other: "AntiMobius") -> "AntiMobius":
        """Return self ∘ other (apply `other` first)."""
        rhs = np.conj(other.m) if self.conj else other.m
        return AntiMobius(self.m @ rhs, self.conj ^ other.conj)

    def inverse(self) -> "AntiMobius":
        inv = np.linalg.inv(self.m)
        return AntiMobius(np.conj(inv) if self.conj else inv, self.conj)

    def normalized(self) -> "AntiMobius":
        """Scale M to determinant 1 so long compositions don't overflow."""
        return AntiMobius(self.m / cmath.sqrt(np.linalg.det(self.m)), self.conj)

    def preimage_of_infinity(self) -> complex:
        """g⁻¹(∞), or complex('inf') if g fixes ∞."""
        (a, b), (c, d) = self.m
        if abs(c) < _EPS * max(abs(a), abs(d), 1.0):
            return complex("inf")
        z = -d / c
        return z.conjugate() if self.conj else z

    def is_identity(self, tol: float = 1e-9) -> bool:
        """True if this is the identity map (M a scalar multiple of I)."""
        if self.conj:
            return False
        m = self.m / self.m[0, 0]
        return bool(np.allclose(m, np.eye(2), atol=tol))


# ---------------------------------------------------------------------------
# Geodesics (lines and circles) and arcs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Geodesic:
    """A generalized circle: either a line (point + unit direction) or a circle."""

    kind: str                       # "line" | "circle"
    point: complex = 0j             # line: a point on it
    direction: complex = 1 + 0j     # line: unit direction
    center: complex = 0j            # circle
    radius: float = 0.0             # circle

    @staticmethod
    def line(point: complex, direction: complex) -> "Geodesic":
        return Geodesic("line", point=point, direction=direction / abs(direction))

    @staticmethod
    def circle(center: complex, radius: float) -> "Geodesic":
        return Geodesic("circle", center=center, radius=radius)

    def reflection(self) -> AntiMobius:
        if self.kind == "line":
            e2 = self.direction ** 2
            p = self.point
            m = np.array([[e2, p - e2 * p.conjugate()], [0, 1]], dtype=complex)
        else:
            c, r = self.center, self.radius
            m = np.array([[c, r * r - abs(c) ** 2], [1, -c.conjugate()]], dtype=complex)
        return AntiMobius(m, True)

    def signed(self, z: complex) -> float:
        """Signed 'which side' function; zero on the geodesic."""
        if self.kind == "line":
            return ((z - self.point) * self.direction.conjugate()).imag
        return abs(z - self.center) ** 2 - self.radius ** 2


def circle_through(z1: complex, z2: complex, z3: complex) -> Optional[tuple[complex, float]]:
    """Circle through three points, or None if they are (numerically) collinear."""
    w = (z3 - z1) / (z2 - z1)
    if abs(w.imag) < 1e-9 * max(1.0, abs(w)):
        return None
    center = (z2 - z1) * (w - abs(w) ** 2) / (2j * w.imag) + z1
    return center, abs(z1 - center)


def arc_through(u: complex, m: complex, v: complex) -> Optional[tuple[complex, float, float, float]]:
    """
    Arc from u through m to v.

    Returns (center, radius, theta_u, sweep) with `sweep` signed (positive =
    counter-clockwise), or None if the three points are collinear — in that
    case the "arc" is the straight segment u → v.
    """
    circ = circle_through(u, m, v)
    if circ is None:
        return None
    c, r = circ
    if r > 1e6 * abs(v - u):
        return None
    tu, tm, tv = (cmath.phase(z - c) for z in (u, m, v))
    dm = (tm - tu) % (2 * math.pi)
    dv = (tv - tu) % (2 * math.pi)
    sweep = dv if dm < dv else dv - 2 * math.pi
    return c, r, tu, sweep


def arc_tangent(u: complex, m: complex, v: complex) -> complex:
    """Unit tangent vector at u of the arc u → m → v."""
    arc = arc_through(u, m, v)
    if arc is None:
        return (v - u) / abs(v - u)
    c, r, _, sweep = arc
    t = 1j * (u - c) / r
    return t if sweep > 0 else -t


def arc_contains(u: complex, m: complex, v: complex, z: complex) -> bool:
    """True if z lies on the arc u → m → v (assumes z is on the circle)."""
    arc = arc_through(u, m, v)
    if arc is None:
        return abs(abs(z - u) + abs(v - z) - abs(v - u)) < 1e-9
    c, _, tu, sweep = arc
    dz = (cmath.phase(z - c) - tu) % (2 * math.pi)
    return dz <= sweep if sweep > 0 else dz >= 2 * math.pi + sweep


# ---------------------------------------------------------------------------
# Fundamental triangle
# ---------------------------------------------------------------------------

def classify(p: int, q: int, r: int) -> str:
    s = Fraction(1, p) + Fraction(1, q) + Fraction(1, r)
    if s < 1:
        return HYPERBOLIC
    if s == 1:
        return EUCLIDEAN
    return SPHERICAL


@dataclass
class Triangle:
    p: int
    q: int
    r: int
    kind: str
    vertices: tuple[complex, complex, complex]           # (V_p, V_q, V_r)
    sides: dict[str, Geodesic]                           # "a", "b", "c"
    side_midpoints: dict[str, complex]                   # a point inside each side
    reflections: dict[str, AntiMobius] = field(default_factory=dict)
    side_lengths: tuple[float, float, float] = (0.0, 0.0, 0.0)  # |V_pV_q|, |V_qV_r|, |V_rV_p|
    view_radius: float = 1.0

    @property
    def angles(self) -> tuple[float, float, float]:
        return (math.pi / self.p, math.pi / self.q, math.pi / self.r)

    def edges(self) -> list[tuple[complex, complex, complex]]:
        """Boundary as (start, interior point, end) triples: c, a, b in order."""
        vp, vq, vr = self.vertices
        mid = self.side_midpoints
        return [(vp, mid["c"], vq), (vq, mid["a"], vr), (vr, mid["b"], vp)]

    def interior_point(self) -> complex:
        return sum(self.vertices) / 3

    def contains(self, z: complex, tol: float = 1e-9) -> bool:
        """Closed-triangle membership test (finite z only)."""
        if not cmath.isfinite(z):
            return False
        inside = self.interior_point()
        for side in self.sides.values():
            s = side.signed(z)
            if abs(s) > tol and (s > 0) != (side.signed(inside) > 0):
                return False
        return True


def _measured_vertex_angles(tri: Triangle) -> tuple[float, float, float]:
    """Interior angles measured from arc tangents (used by tests)."""
    (vp, mc, vq), (_, ma, vr), (_, mb, _) = tri.edges()

    def angle(t1: complex, t2: complex) -> float:
        return math.acos(max(-1.0, min(1.0, (t1 * t2.conjugate()).real)))

    return (
        angle(arc_tangent(vp, mc, vq), arc_tangent(vp, mb, vr)),
        angle(arc_tangent(vq, ma, vr), arc_tangent(vq, mc, vp)),
        angle(arc_tangent(vr, mb, vp), arc_tangent(vr, ma, vq)),
    )


def fundamental_triangle(p: int, q: int, r: int) -> Triangle:
    """Build the fundamental triangle with angles π/p, π/q, π/r at V_p, V_q, V_r."""
    if min(p, q, r) < 2:
        raise ValueError("p, q, r must all be ≥ 2")
    kind = classify(p, q, r)
    al, be, ga = math.pi / p, math.pi / q, math.pi / r
    rot = cmath.exp(1j * al)

    if kind == EUCLIDEAN:
        # Scale so |V_pV_q| = 1; law of sines gives |V_pV_r|.
        len_pq, len_pr = 1.0, math.sin(be) / math.sin(ga)
        vq, vr = complex(len_pq), len_pr * rot
        len_qr = abs(vr - vq)
        side_a = Geodesic.line(vq, vr - vq)
        mid_a = (vq + vr) / 2
        view = 1.0  # set later from the generated tiles
    else:
        # Dual law of cosines (same formula in H² with cosh, in S² with cos).
        def opposite(c_ang: float, a_ang: float, b_ang: float) -> float:
            return (math.cos(c_ang) + math.cos(a_ang) * math.cos(b_ang)) / (
                math.sin(a_ang) * math.sin(b_ang))

        if kind == HYPERBOLIC:
            len_pq = math.acosh(opposite(ga, al, be))
            len_pr = math.acosh(opposite(be, al, ga))
            len_qr = math.acosh(opposite(al, be, ga))
            to_plane = lambda d: math.tanh(d / 2)       # Poincaré disk radius
        else:
            len_pq = math.acos(opposite(ga, al, be))
            len_pr = math.acos(opposite(be, al, ga))
            len_qr = math.acos(opposite(al, be, ga))
            to_plane = lambda d: math.tan(d / 2)        # stereographic radius

        vq, vr = complex(to_plane(len_pq)), to_plane(len_pr) * rot
        # Third point on side a's circle: the inverse of V_q in the unit circle
        # (hyperbolic) or the antipode of V_q (spherical).
        partner = 1 / vq.conjugate() if kind == HYPERBOLIC else -1 / vq.conjugate()
        center, radius = circle_through(vq, vr, partner)
        side_a = Geodesic.circle(center, radius)
        # Pick the arc midpoint that does NOT pass through `partner`.
        n = ((vq + vr) / 2 - center)
        n /= abs(n)
        mid_a = center + radius * n
        if arc_contains(vq, mid_a, vr, partner):
            mid_a = center - radius * n

        if kind == HYPERBOLIC:
            view = 1.0
        else:
            # Hide the tiles touching ∞: they lie within the triangle's diameter
            # of the north pole, so stay inside that cap's complement.
            diam = max(len_pq, len_pr, len_qr)
            view = min(3.0, 0.97 * math.tan((math.pi - diam) / 2))

    sides = {
        "a": side_a,
        "b": Geodesic.line(0j, rot),
        "c": Geodesic.line(0j, 1 + 0j),
    }
    tri = Triangle(
        p=p, q=q, r=r, kind=kind,
        vertices=(0j, vq, vr),
        sides=sides,
        side_midpoints={"a": mid_a, "b": vr / 2, "c": vq / 2},
        side_lengths=(len_pq, len_qr, len_pr),
        view_radius=view,
    )
    tri.reflections = {name: side.reflection() for name, side in sides.items()}
    return tri
