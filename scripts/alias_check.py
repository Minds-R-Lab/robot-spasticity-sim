"""Check the identifiability claim numerically.

Claim.  Let the reflex fire when q(t-d) + mu q'(t-d) >= lam.  On a stretch at constant speed v, q(t) = q0 + v t, so firing is seen at
q(t) = lam + (d - mu) v.  Two pairs (mu, d) with the same mu - d give the same onset at every speed, so only mu - d is identifiable
from constant-speed stretches.  Any slope fit to onsets against speed reads mu - d, not mu.

The check: for a patient (lam, mu, d), scan (mu', d') on a grid and compute the misfit of the reflex shape, first on constant-speed
ramps and then on real stretches, which accelerate and brake.  The first must be flat along mu' - d' = mu - d; the second need not be.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elbow.assessors import trapezoid          # noqa: E402
from elbow.device import DT                    # noqa: E402
from elbow.model import reflex_shape           # noqa: E402

lam, mu, d = 0.6, 0.12, 10                     # d in ticks of 5 ms: 50 ms
mus = np.round(np.arange(0.02, 0.32, 0.01), 2)
ds = np.array([4, 6, 8, 10, 12, 14, 16, 18, 20, 24])
speeds = (1.0, 1.5, 2.0, 3.2)


def ramp(v, end=1.2):
    n = int(end / v / DT)
    t = np.arange(n) * DT
    return v * t - 0.3, np.full(n, v)          # a ramp already at speed well before the threshold


def shapes(moves, m, dd):
    return np.concatenate([reflex_shape(q, qd, np.array([lam]), np.array([m]), np.array([8.0]), np.array([dd]))[:, 0] for q, qd in moves])


for name, moves in (("constant speed", [ramp(v) for v in speeds]), ("real stretches", [trapezoid(1.2, v) for v in speeds])):
    y = shapes(moves, mu, d)
    print(name)
    best = {}
    for dd in ds:
        e = [np.sqrt(np.mean((shapes(moves, m, dd) - y) ** 2)) for m in mus]
        i = int(np.argmin(e))
        best[dd] = (mus[i], e[i])
        print(f"  d {dd * DT * 1000:4.0f} ms: best mu {mus[i]:.2f}, mu - d {mus[i] - dd * DT:+.3f}, misfit {e[i]:.5f}")
print(f"truth: mu {mu}, d {d * DT}, mu - d {mu - d * DT:.3f}")
