"""Score the exploratory leave-one-out run (explore_t7_loo.py).  Not pre-specified."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report2 import load                                      # noqa: E402
from test6_report import arms, compare, table                 # noqa: E402

ARMS = ["short_e", "l2a_e", "hyb_e", "short_t", "l2a_t", "hyb_t"]

if __name__ == "__main__":
    d = arms(load(sys.argv[1]))
    for w in ("lit", "myo"):
        ts = [t for t in sorted(d) if t.startswith(w)]
        ts = [f"{w}_real"] + [t for t in ts if t != f"{w}_real"]
        print(f"\n## {w}: real, and real with one realism off (150 patients, session 1)")
        table(d, ts, ARMS, rt=False)
        for t in ts:
            compare(d, t, "hyb_e", "short_e", dict(lam=0.03, k=0.0, q=0.0))
            compare(d, t, "l2a_t", "short_t", dict(lam=0.0))
