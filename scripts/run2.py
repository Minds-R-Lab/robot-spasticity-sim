"""One simulated patient in the second-generation world, one assessor, one session.  Appends one JSON line.  A patient depends only on
(seed, world, scenario), so every assessor meets the same person, with the same notes and the same day-to-day drift.  --session 2 is
a retest on another day."""
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
from elbow.assessors import METHODS                                         # noqa: E402
from elbow.device2 import Session2                                          # noqa: E402
from elbow.real import Robot                                                # noqa: E402
from elbow.world import (LitArm, MyoArm, k_ref_lit, myo_gravity, sample_lit, sample_myo)    # noqa: E402


def patient(seed, world, scenario):
    tag = zlib.crc32(f"{world}{scenario}".encode())
    rng = np.random.default_rng([seed, tag])
    if world == "lit":
        p, n = sample_lit(rng, scenario)
        return p, n, LitArm, p["G"], p["phi"], k_ref_lit(p)
    p, n = sample_myo(rng, scenario)
    g, phi = myo_gravity(p)
    return p, n, MyoArm, g, phi, p["k80"]


def run(seed, world, scenario, method, session):
    p, n, Arm, g, phi, k = patient(seed, world, scenario)
    r = np.random.default_rng([seed, 7])
    info = dict(G_hat=float(g * (1 + 0.15 * r.standard_normal())), phi_hat=float(phi + 0.1 * r.standard_normal()),
                rom_hat=float(p["q_s"] + 0.2 * r.standard_normal()))
    if "ELBOW_ORACLE" in os.environ:                      # an ablation: 1 tells the fit the true delay, else this delay
        v = float(os.environ["ELBOW_ORACLE"])
        info["oracle_delay"] = n.delay if v == 1 else v
    day = np.random.default_rng([seed, 8]).standard_normal(3)[session] * n.day_sd
    rng = np.random.default_rng([seed, 2, session, zlib.crc32(method.encode())])
    if p.get("position"):                                 # test 7: this session's shoulder angle, the same for every assessor
        p["position"]["shoulder"] = float(np.random.default_rng([seed, 9]).uniform(-0.15, 0.15, 3)[session])
        p["position"]["seed"] = [seed, 12, session]
    arm = Arm(p, n, rng, day)
    if p.get("device"):                                   # test 7: the robot as built, with its own sensor noise
        arm = Robot(arm, p["device"], np.random.default_rng([seed, 11, session, zlib.crc32(method.encode())]))
    ses = Session2(arm, rng)
    t0 = time.time()
    rep = METHODS[method](ses, info, np.random.default_rng([seed, 3, session]))
    catch = n.lam - n.mu * 3.2
    truth = dict(q_s=p["q_s"], k80=k, reflex=bool(n.reflex), lam=n.lam, mu=n.mu, reach=bool(n.reflex and catch < 0.8 * p["q_s"]),
                 phenotype=n.phenotype, guard=bool(n.guard), active=n.effort_rate > 0, gain=n.gain, tau_max=n.tau_max, delay=n.delay,
                 day=float(day))
    return dict(seed=seed, world=world, scenario=scenario, method=method, session=session, report=rep, truth=truth, info=info,
                time_s=ses.t, stretches=ses.n_fast, cutouts=ses.n_cutouts, safety=ses.result_truth(), wall=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--world", default="lit")
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--method", required=True)
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.seed, a.world, a.scenario, a.method, a.session)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
