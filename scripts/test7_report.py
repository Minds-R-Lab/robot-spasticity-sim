"""The scoring fixed in PROTOCOL7.md.  It reuses the test 6 scoring functions unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report2 import load                                                       # noqa: E402
from test6_report import arms, compare, k_e, lam_e, paired, q_e, said, table   # noqa: E402

ARMS = ["pub_e", "short_e", "l2a_e", "hyb_e", "pub_t", "short_t", "l2a_t", "hyb_t"]
REAL = ["lit_real", "myo_real"]
BASE = {"lit_real": "lit_base", "myo_real": "myo_base"}


def degrade(d, t, b):
    """The same patients in the realistic world (A) and in the base world (B), arm by arm."""
    print(f"\n#### {t} (A) against {b} (B), the same patients\n")
    print("| Arm | Endpoint | n | A | B | A - B (95% CI) |\n| --- | --- | --- | --- | --- | --- |")
    for m in ARMS:
        if m not in d[t] or m not in d.get(b, {}):
            continue
        A, B = d[t][m], d[b][m]
        for name, f, sel in (("threshold error (rad)", lam_e, lambda x, y: x["truth"]["reach"] and said(x) and said(y)),
                             ("stiffness error", k_e, lambda x, y: True), ("range error (rad)", q_e, lambda x, y: True)):
            x = paired(A, B, f, sel)
            if x:
                print(f"| {m} | {name} | {x['n']} | {x['a']:.3f} | {x['b']:.3f} | {x['diff']:+.3f} ({x['lo']:+.3f} to {x['hi']:+.3f}) |")


if __name__ == "__main__":
    d = arms(load(sys.argv[1]))
    n = sum(len(v) for t in d.values() for k, v in t.items() if not k.startswith("hyb"))
    print(f"# Test 7: {n} sessions")
    real = [t for t in REAL if t in d]
    print("\n## Realistic patients, session 1, with retest")
    table(d, real, ARMS, rt=True)
    print("\n## Pre-specified comparisons, realistic patients")
    for t in real:
        compare(d, t, "short_e", "pub_e", dict(lam=0.03, dec=0.03))
        compare(d, t, "hyb_e", "short_e", dict(lam=0.03, k=0.0, q=0.0))
        compare(d, t, "l2a_t", "short_t", dict(lam=0.0))
        compare(d, t, "l2a_e", "short_e", {})
        compare(d, t, "short_t", "pub_t", dict(lam=0.03, dec=0.03))
        compare(d, t, "hyb_t", "short_t", dict(lam=0.03, k=0.0, q=0.0))
    print("\n## Realistic against base, the same patients")
    for t in real:
        if BASE[t] in d:
            degrade(d, t, BASE[t])
    base = [t for t in ("lit_base", "myo_base") if t in d]
    if base:
        print("\n## Base patients of this test, session 1, with retest")
        table(d, base, ARMS, rt=True)
    one = [t for t in sorted(d) if t not in REAL and t not in base]
    print("\n## One realism at a time")
    table(d, one, ARMS, rt=any(k[1] == 2 for t in one for m in d[t] for k in d[t][m]))
    for t in one:
        compare(d, t, "short_e", "pub_e", dict(lam=0.03, dec=0.03))
        compare(d, t, "hyb_e", "short_e", dict(lam=0.03, k=0.0, q=0.0))
        compare(d, t, "l2a_t", "short_t", dict(lam=0.0))
