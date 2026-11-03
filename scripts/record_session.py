"""Record one Bayesian session on the Hill (MyoSuite) body, every 5 ms, for the video and the setup figure.

The patient comes from the development seeds (5000 and up), never from a test.  Saves results/video/session_<seed>.npz.
Usage: python scripts/record_session.py [seed [method]]   (no seed: search 5000-5199 for a clear moderate case with l2a_e)
With a method other than l2a_e (pub_e, short_e), saves results/video/session_<seed>_<method>.npz, for the protocol timeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from elbow.assessors import METHODS                   # noqa: E402
from elbow.device2 import Session2                    # noqa: E402
from run2 import patient                              # noqa: E402

OUT = ROOT / "results" / "video"                      # recordings; the rendered video is in video/


class Recorder(Session2):
    """Session2 that also keeps every tick: time, angle, motor torque, EMG, reflex torque (truth) and muscle activations."""

    def __init__(self, arm, rng):
        super().__init__(arm, rng)
        self.phase, self.ticks, self.marks = "rest", [], []

    def _tick(self, q_r, v_r, rows=None):
        out = super()._tick(q_r, v_r, rows)
        a = self.arm
        self.ticks.append((a.t, a.q, a.qd, out[0], a.emg, a.nv.a, *a.d.act[:6]))
        return out

    def rest(self, seconds, gravity_hold=True):
        self.phase = "rest"
        self.marks.append((self.arm.t, "rest", 0.0))
        return super().rest(seconds, gravity_hold)

    def move(self, target, speed, label="", stop=None, governor=None, record=True):
        self.marks.append((self.arm.t, label or "move", float(speed)))
        return super().move(target, speed, label, stop, governor, record)


def session(seed, method="l2a_e"):
    p, n, Arm, g, phi, k = patient(seed, "myo", "base")
    r = np.random.default_rng([seed, 7])
    info = dict(G_hat=float(g * (1 + 0.15 * r.standard_normal())), phi_hat=float(phi + 0.1 * r.standard_normal()),
                rom_hat=float(p["q_s"] + 0.2 * r.standard_normal()))
    rng = np.random.default_rng([seed, 2, 0, 1])
    ses = Recorder(Arm(p, n, rng, 0.0), rng)
    rep = METHODS[method](ses, info, np.random.default_rng([seed, 3, 0]))
    return p, n, ses, rep


def save(seed, p, n, ses, rep, method="l2a_e"):
    OUT.mkdir(parents=True, exist_ok=True)
    if method != "l2a_e":
        meta = dict(seed=seed, method=method, lam=float(n.lam), time=float(ses.t), reflex=rep["reflex"], lam_hat=rep.get("lam"),
                    marks=[[float(t), s, v] for t, s, v in ses.marks])
        np.savez_compressed(OUT / f"session_{seed}_{method}.npz", ticks=np.array(ses.ticks)[:, :5], meta=json.dumps(meta))
        return meta
    tr = rep.get("trace", [])
    meta = dict(seed=seed, phenotype=n.phenotype, lam=float(n.lam), mu=float(n.mu), delay=float(n.delay), q_s=float(p["q_s"]),
                reflex=rep["reflex"], lam_hat=rep.get("lam"), lam_sd=rep.get("lam_sd"), speeds=[float(v) for v in rep["speeds"]],
                trace=[[float(t[0]), None if t[1] is None else float(t[1]), None if t[2] is None else float(t[2]), float(t[5])] for t in tr],
                marks=[[float(t), s, v] for t, s, v in ses.marks], time=float(ses.t),
                catches=[[float(v), None if a is None else float(a), None if b is None else float(b)] for v, a, b in rep["catches"]],
                onset_lam=rep["onset_lam"], onset_mu=rep["onset_mu"], onset_reflex=rep["onset_reflex"])
    np.savez_compressed(OUT / f"session_{seed}.npz", ticks=np.array(ses.ticks), meta=json.dumps(meta))
    return meta


if __name__ == "__main__":
    seeds = [int(sys.argv[1])] if len(sys.argv) > 1 else range(5000, 5200)
    method = sys.argv[2] if len(sys.argv) > 2 else "l2a_e"
    if method != "l2a_e":
        p, n, ses, rep = session(seeds[0], method)
        print(save(seeds[0], p, n, ses, rep, method)["time"])
        sys.exit()
    for s in seeds:
        p, n, *_ = patient(s, "myo", "base")
        if len(seeds) > 1 and not (n.phenotype.startswith("moderate") and 0.3 < n.lam < 0.9 and "guard" not in n.phenotype):
            continue
        p, n, ses, rep = session(s)
        ok = rep["reflex"] == "yes" and rep.get("lam") is not None and abs(rep["lam"] - n.lam) < 0.06 and 3 <= len(rep["speeds"]) <= 5
        print(s, n.phenotype, "lam", round(n.lam, 3), "report", rep["reflex"], rep.get("lam"), rep["speeds"], "t", round(ses.t, 1), flush=True)
        if ok or len(seeds) == 1:
            meta = save(s, p, n, ses, rep)
            print("saved", json.dumps({k: meta[k] for k in ("seed", "lam", "lam_hat", "lam_sd", "speeds", "time")}))
            break
