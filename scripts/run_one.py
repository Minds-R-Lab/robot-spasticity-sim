"""One simulated patient, one assessor: appends one JSON line.  The patient depends only on the seed, so every assessor
meets the same person, with the same notes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elbow.assessors import METHODS                         # noqa: E402
from elbow.device import Session                            # noqa: E402
from elbow.patient import sample_patient                    # noqa: E402


def notes(p, seed):
    """What the clinic knows before the session: a weight estimate and a remembered range, each with an error."""
    r = np.random.default_rng([seed, 7])
    return dict(G_hat=float(p.G * (1 + 0.15 * r.standard_normal())),
                rom_hat=float(p.q_s + 0.2 * r.standard_normal()))


def truth(p):
    catch_at_top = p.lam - p.mu * 3.2
    return dict(q_s=p.q_s, k80=p.k_ref, reflex=bool(p.reflex), lam=p.lam, mu=p.mu, reach=bool(p.reflex and catch_at_top
                < 0.8 * p.q_s), phenotype=p.phenotype, guard=bool(p.guard), active=p.effort_rate > 0,
                gain=p.gain, tau_max=p.tau_max, delay=p.delay, Ta=p.Ta)


def run(seed, method, stress):
    p = sample_patient(np.random.default_rng([seed, 1]), stress)
    rng = np.random.default_rng([seed, 2, zlib.crc32(method.encode())])
    ses = Session(p, rng)
    info = notes(p, seed)
    if "ELBOW_ORACLE" in os.environ:                      # development diagnostic: 1 tells the fit the true delay, else this delay
        v = float(os.environ["ELBOW_ORACLE"])
        info["oracle_delay"] = p.delay if v == 1 else v
    t0 = time.time()
    rep = METHODS[method](ses, info, np.random.default_rng([seed, 3]))
    return dict(seed=seed, method=method, stress=stress, report=rep, truth=truth(p), info=info,
                time_s=ses.t, stretches=ses.n_fast, cutouts=ses.n_cutouts, safety=ses.result_truth(),
                wall=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--stress", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.seed, a.method, bool(a.stress))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
