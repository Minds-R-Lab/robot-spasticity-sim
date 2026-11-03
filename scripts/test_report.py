"""The scoring fixed in PROTOCOL.md: both comparisons, both populations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import load, med, metrics, sign_p, table      # noqa: E402

ALPHA = 0.05 / 20


def boot(x, rng, n=4000):
    x = np.array(x)
    if len(x) == 0:
        return float("nan"), float("nan")
    b = np.median(rng.choice(x, (n, len(x))), axis=1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def both_yes(r):
    return r["report"]["reflex"] == "yes" and r["report"].get("lam") is not None


def e_lam(r):
    return abs(r["report"]["lam"] - r["truth"]["lam"])


def e_mu(r):
    return abs(r["report"]["mu"] - r["truth"]["mu"])


def e_k(r):
    k = r["report"]["k80"]
    return abs(np.log(k / r["truth"]["k80"])) if k and k > 0 else 9.0


def e_q(r):
    return abs(r["report"]["q_end"] - r["truth"]["q_s"])


def decision(r):
    if not (r["truth"]["reach"] or not r["truth"]["reflex"]):
        return None
    want = "yes" if r["truth"]["reach"] else "no"
    return 0.5 if r["report"]["reflex"] == "cannot tell" else float(r["report"]["reflex"] == want)


def compare(d, a, b, rng):
    seeds = sorted(set(d[a]) & set(d[b]))
    rows = []
    for name, f, sel, sign in (("threshold", e_lam, lambda x, y: x["truth"]["reach"] and both_yes(x) and both_yes(y), -1),
                               ("speed slope", e_mu, lambda x, y: x["truth"]["reach"] and both_yes(x) and both_yes(y), -1),
                               ("stiffness", e_k, lambda x, y: True, -1),
                               ("end of range", e_q, lambda x, y: True, -1)):
        xs, ys = [], []
        for s in seeds:
            x, y = d[a][s], d[b][s]
            if sel(x, y):
                xs.append(f(x))
                ys.append(f(y))
        xs, ys = np.array(xs), np.array(ys)
        w, l = int((xs < ys).sum()), int((xs > ys).sum())
        p = sign_p(w, l)
        cx, cy = boot(xs, rng), boot(ys, rng)
        rows.append(f"| {name} | {len(xs)} | {med(xs):.3f} ({cx[0]:.3f} to {cx[1]:.3f}) | {med(ys):.3f} ({cy[0]:.3f} to {cy[1]:.3f})"
                    f" | {w} to {l} | {p:.2g} | {'yes' if p < ALPHA else 'no'} |")
    xs = [(decision(d[a][s]), decision(d[b][s])) for s in seeds if decision(d[a][s]) is not None]
    x, y = np.array(xs).T
    w, l = int((x > y).sum()), int((x < y).sum())
    p = sign_p(w, l)
    rows.append(f"| reflex decision (mean, higher is better) | {len(x)} | {x.mean():.3f} | {y.mean():.3f} | {w} to {l} | {p:.2g} |"
                f" {'yes' if p < ALPHA else 'no'} |")
    print(f"\n### {a} against {b}\n")
    print(f"| Endpoint | n | {a} | {b} | Patients better | p | Shown (p < {ALPHA}) |\n| --- | --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))
    for m in (a, b):
        x = metrics(list(d[m].values()))
        print(f"\n{m}: sens {x['sens']:.3f} spec {x['spec']:.3f} wrong {x['wrong']:.3f} cannot tell {x['unsure']:.3f};"
              f" stretches {x['stretches']:.1f}; time {x['time']:.0f} s; past end of range {x['over']:.3f}; cut-outs {x['cut']:.3f};"
              f" 95th pct resisting torque {x['resist']:.1f}; cover q {x['q_cover']:.2f} k {x['k_cover']:.2f} lam {x['lam_cover']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args()
    d = load(a.folder)
    rng = np.random.default_rng(0)
    print(f"# {a.folder}: {sum(len(v) for v in d.values())} sessions")
    compare(d, "l2a_e", "rule_e", rng)
    compare(d, "l2a_t", "rule_t", rng)
    print("\n### The model alone: the law on the rule's fixed schedule")
    compare(d, "l2a_e_fixed", "rule_e", rng)
    compare(d, "l2a_t_fixed", "rule_t", rng)
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
