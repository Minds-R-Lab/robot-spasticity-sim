"""Record one TSRT-4 session (short_e) on the Hill body in the realistic world of test 7, every 5 ms, for the second video.

Development seeds only (5000 and up), never a test seed.  The same patient is also run with the soft cuff turned off
(`real-device`, as in explore_t7_loo.py), for the comparison at the end of the video.
Saves results/video/real_<seed>.npz.
Usage: python scripts/record_real.py [seed]     (no seed: search 5000-5399 for a patient with no reflex whom the robot
calls spastic in the realistic world and not with a rigid cuff)
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import explore_t7_loo                                 # noqa: E402,F401  (adds the real-X scenarios)
import run2                                           # noqa: E402
from elbow.assessors import METHODS                   # noqa: E402
from elbow.real import Robot                          # noqa: E402
from record_session import Recorder as Base           # noqa: E402

OUT = ROOT / "results" / "video"
METHOD = "short_e"


class Recorder(Base):
    """Also keeps the true forearm angle (behind the soft cuff), the shoulder angle and the muscle activations."""

    def _tick(self, q_r, v_r, rows=None):
        out = super(Base, self)._tick(q_r, v_r, rows)
        a = self.arm
        arm = a.arm if isinstance(a, Robot) else a
        sh = float(arm.d.qpos[0]) if arm.j == 1 else 0.0
        self.ticks.append((a.t, a.q, arm.q, out[0], a.emg, arm.nv.a, sh, *arm.d.act[:6]))
        return out


def session(seed, scenario):
    p, n, Arm, g, phi, k = run2.patient(seed, "myo", scenario)
    r = np.random.default_rng([seed, 7])
    info = dict(G_hat=float(g * (1 + 0.15 * r.standard_normal())), phi_hat=float(phi + 0.1 * r.standard_normal()),
                rom_hat=float(p["q_s"] + 0.2 * r.standard_normal()))
    day = np.random.default_rng([seed, 8]).standard_normal(3)[1] * n.day_sd
    rng = np.random.default_rng([seed, 2, 1, zlib.crc32(METHOD.encode())])
    if p.get("position"):
        p["position"]["shoulder"] = float(np.random.default_rng([seed, 9]).uniform(-0.15, 0.15, 3)[1])
        p["position"]["seed"] = [seed, 12, 1]
    arm = Arm(p, n, rng, day)
    if p.get("device"):
        arm = Robot(arm, p["device"], np.random.default_rng([seed, 11, 1, zlib.crc32(METHOD.encode())]))
    ses = Recorder(arm, rng)
    rep = METHODS[METHOD](ses, info, np.random.default_rng([seed, 3, 1]))
    return p, n, ses, rep


def summary(p, n, ses, rep):
    return dict(reflex=rep["reflex"], lam_hat=rep.get("lam"), time=float(ses.t), stretches=ses.n_fast)


def save(seed, p, n, ses, rep, rigid):
    OUT.mkdir(parents=True, exist_ok=True)
    dev = p.get("device") or {}
    meta = dict(seed=seed, method=METHOD, phenotype=n.phenotype, reflex_true=bool(n.reflex), lam=float(n.lam), q_s=float(p["q_s"]),
                cuff_k=dev.get("k"), cuff_zeta=dev.get("zeta"), mis=dev.get("mis"),
                shoulder=(p.get("position") or {}).get("shoulder"), marks=[[float(t), s, v] for t, s, v in ses.marks],
                rigid=rigid, catches=[[None if x is None else float(x) for x in c] for c in rep.get("catches", [])],
                **summary(p, n, ses, rep))
    np.savez_compressed(OUT / f"real_{seed}.npz", ticks=np.array(ses.ticks), meta=json.dumps(meta))
    return meta


if __name__ == "__main__":
    seeds = [int(sys.argv[1])] if len(sys.argv) > 1 else range(5000, 5400)
    for s in seeds:
        p, n, *_ = run2.patient(s, "myo", "real")
        if len(seeds) > 1 and (n.reflex or n.guard or n.effort_rate > 0):
            continue
        p, n, ses, rep = session(s, "real")
        print(s, n.phenotype, "real:", rep["reflex"], rep.get("lam"), flush=True)
        if rep["reflex"] != "yes" and len(seeds) > 1:
            continue
        p2, n2, ses2, rep2 = session(s, "real-device")
        print("   rigid cuff:", rep2["reflex"], rep2.get("lam"), flush=True)
        if rep2["reflex"] == "no" or len(seeds) == 1:
            print(json.dumps({k: v for k, v in save(s, p, n, ses, rep, summary(p2, n2, ses2, rep2)).items() if k not in ("marks",)}))
            break
