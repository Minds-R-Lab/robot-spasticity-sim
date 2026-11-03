"""Re-run some test 6 sessions with the current code and compare their reports with the frozen results, field by field.

Test 7 added code to the simulator; this shows that test 6 patients and sessions are unchanged.
Usage: python scripts/check_t6_repro.py [n per arm] [test 6 results folder] [case index]
Run it where test 6 ran: floating point differs between machines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run2 import run                                 # noqa: E402

T6 = ROOT / "results" / "test6"


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    T6 = Path(sys.argv[2]) if len(sys.argv) > 2 else T6
    cases = [("lit_base", "pub_e"), ("lit_base", "l2a_t"), ("myo_base", "l2a_e"), ("myo_base", "short_t"),
             ("lit_force", "l2a_e"), ("lit_react", "pub_t"), ("myo_early", "short_e"), ("lit_sum", "l2a_t")]
    if len(sys.argv) > 3:
        cases = [cases[int(sys.argv[3])]]
    bad = 0
    for tag, m in cases:
        rows = [json.loads(x) for f in sorted((T6 / tag).glob(f"*.{m}.jsonl")) for x in f.read_text().splitlines()[:k]]
        for r in rows:
            w, sc = tag.split("_", 1)
            new = json.loads(json.dumps(run(r["seed"], w, sc, m, r["session"])))      # as saved: tuples become lists
            ok = same(new["report"], r["report"]) and same(new["truth"], r["truth"]) and new["stretches"] == r["stretches"]
            bad += not ok
            print(f"{tag:10s} {m:8s} seed {r['seed']} s{r['session']}: {'same' if ok else 'DIFFERENT'}", flush=True)
    print("all same" if bad == 0 else f"{bad} different")
