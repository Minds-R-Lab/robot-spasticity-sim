"""Compare simulated fatigue with the Patterson et al. 2022 data: 20 stretches at 120 deg/s, holds of 10 s, net torque read at the same angle at
constant speed, as a ratio to the first stretch.  Development seeds only.  usage: fatigue_check.py world scenario first last out"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from elbow.device2 import Session2         # noqa: E402
from run2 import patient                   # noqa: E402

V = 2.09


def series(seed, world, scenario):
    p, n, Arm, *_ = patient(seed, world, scenario)
    if not n.reflex or n.guard or n.effort_rate > 0 or n.lam - n.mu * V > 0.6 * p["q_s"]:
        return None
    rng = np.random.default_rng([seed, 9])
    ses = Session2(Arm(p, n, rng, 0.0), rng)
    ses.rest(1.0)
    q_end, q_read = 0.9 * p["q_s"], 0.75 * p["q_s"]
    tau = []
    for i in range(20):
        r = ses.move(q_end, V, label="fast", record=False)["rows"]
        k = int(np.argmin(np.abs(r[:, 1] - q_read)))
        tau.append(float(np.mean(r[max(k - 5, 0):k + 5, 3])))
        ses.rest(10.0)
        ses.move(0.0, V, label="back", record=False)
        ses.rest(10.0)
    return tau


if __name__ == "__main__":
    world, sc, a, b, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    with open(out, "a") as f:
        for s in range(a, b):
            x = series(s, world, sc)
            if x is not None:
                f.write(json.dumps(dict(seed=s, tau=x)) + "\n")
