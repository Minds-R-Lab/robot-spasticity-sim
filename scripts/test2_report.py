"""The scoring fixed in PROTOCOL2.md."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import load, med, sign_p, table                    # noqa: E402
from test_report import compare                                # noqa: E402


def err(r, k):
    return abs(r["report"][k] - r["truth"][k])


def said_yes(r):
    return r["report"]["reflex"] == "yes" and r["report"].get("lam") is not None


def pair(a, b, f, name):
    seeds = sorted(set(a) & set(b))
    s = [i for i in seeds if a[i]["truth"]["reach"] and said_yes(a[i]) and said_yes(b[i])]
    x, y = np.array([f(a[i]) for i in s]), np.array([f(b[i]) for i in s])
    w, l = int((x < y).sum()), int((x > y).sum())
    print(f"| {name} | {len(s)} | {med(x):.3f} | {med(y):.3f} | {w} to {l} | {sign_p(w, l):.2g} |")


def delay_section(root, pop):
    main, d50, orc = load(f"{root}/{pop}"), load(f"{root}/d50/{pop}"), load(f"{root}/oracle/{pop}")
    for m in ("l2a_e", "l2a_t"):
        rs = [r for r in main[m].values() if r["truth"]["reach"] and said_yes(r) and r["report"].get("delay") is not None]
        e = np.array([abs(r["report"]["delay"] - r["truth"]["delay"]) for r in rs])
        f = np.array([abs(0.05 - r["truth"]["delay"]) for r in rs])
        print(f"\n### {m}: the delay\n\nIdentified delay error {med(e) * 1000:.1f} ms; a fixed 50 ms would be {med(f) * 1000:.1f} ms;"
              f" better on {int((e < f).sum())}, worse on {int((e > f).sum())}, p = {sign_p(int((e < f).sum()), int((e > f).sum())):.2g}, n = {len(e)}.\n")
        print("| Endpoint | n | Identified | Other | Patients better | p |\n| --- | --- | --- | --- | --- | --- |")
        pair(main[m], d50[m], lambda r: err(r, "lam"), "threshold, against fixed 50 ms")
        pair(main[m], d50[m], lambda r: err(r, "mu"), "speed slope, against fixed 50 ms")
        pair(main[m], orc[m], lambda r: err(r, "lam"), "threshold, against true delay")
        pair(main[m], orc[m], lambda r: err(r, "mu"), "speed slope, against true delay")


def guarding(d):
    print("\n### Range in guarding patients\n")
    for m in ("rule_t", "l2a_t", "rule_e", "l2a_e"):
        rs = [r for r in d[m].values() if r["truth"]["guard"]]
        e = np.array([r["report"]["q_end"] - r["truth"]["q_s"] for r in rs])
        print(f"{m}: n {len(e)}, median error {np.median(e):+.3f}, median absolute {np.median(np.abs(e)):.3f}, over by more than 0.2 rad {int((e > 0.2).sum())},"
              f" under by more than 0.2 rad {int((e < -0.2).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("pop")
    a = ap.parse_args()
    d = load(f"{a.root}/{a.pop}")
    rng = np.random.default_rng(0)
    print(f"# {a.pop}: {sum(len(v) for v in d.values())} sessions")
    compare(d, "l2a_e", "rule_e", rng)
    compare(d, "l2a_t", "rule_t", rng)
    print("\n### The model alone: the law on the rule's fixed schedule")
    compare(d, "l2a_e_fixed", "rule_e", rng)
    compare(d, "l2a_t_fixed", "rule_t", rng)
    delay_section(a.root, a.pop)
    guarding(d)
    print("\n### Groups")
    table(d)
    for name, f in (("no reflex", lambda r: not r["truth"]["reflex"]),
                    ("reflex in reach", lambda r: r["truth"]["reach"]),
                    ("guarding", lambda r: r["truth"]["guard"]),
                    ("voluntary effort", lambda r: r["truth"]["active"]),
                    ("contracture", lambda r: "contracture" in r["truth"]["phenotype"]),
                    ("mild", lambda r: r["truth"]["phenotype"].startswith("mild")),
                    ("moderate", lambda r: r["truth"]["phenotype"].startswith("moderate")),
                    ("severe", lambda r: r["truth"]["phenotype"].startswith("severe"))):
        table(d, f, name)
