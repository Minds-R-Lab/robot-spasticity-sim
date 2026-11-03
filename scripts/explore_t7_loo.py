"""After test 7, exploratory and not pre-specified: which realism drives the collapse under `real`?

Scenario `real-X` is `real` with X turned off.  The base patient is the `real` patient of the same seed (same tag), so arms
are paired across scenarios up to the realism draws, which shift when a part is off.  Use seeds no test has used (84000 on).
Same command line as run2.py.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run2                                                                    # noqa: E402
from elbow import real                                                         # noqa: E402
from elbow.world import k_ref_lit, myo_gravity, sample_lit, sample_myo, LitArm, MyoArm   # noqa: E402

_parts = real.parts


def parts(scenario):
    if scenario.startswith("real-"):
        return _parts("real") - {scenario[5:]}
    return _parts(scenario)


def patient(seed, world, scenario):
    tag = zlib.crc32(f"{world}{'real' if scenario.startswith('real') else scenario}".encode())
    rng = np.random.default_rng([seed, tag])
    if world == "lit":
        p, n = sample_lit(rng, scenario)
        return p, n, LitArm, p["G"], p["phi"], k_ref_lit(p)
    p, n = sample_myo(rng, scenario)
    g, phi = myo_gravity(p)
    return p, n, MyoArm, g, phi, p["k80"]


real.parts = parts
run2.patient = patient

if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--world", default="lit")
    ap.add_argument("--scenario", default="real")
    ap.add_argument("--method", required=True)
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run2.run(a.seed, a.world, a.scenario, a.method, a.session)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
