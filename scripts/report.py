"""Score one run folder against the simulator's truth."""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from math import comb

import numpy as np

ORDER = ["rule_t", "l2a_t_fixed", "l2a_t", "rule_e", "l2a_e_fixed", "l2a_e"]


def load(folder):
    out = defaultdict(dict)
    for f in glob.glob(folder + "/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[r["method"]][r["seed"]] = r
    return out


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def metrics(rs):
    pos = [r for r in rs if r["truth"]["reach"]]
    neg = [r for r in rs if not r["truth"]["reflex"]]
    a = lambda s, v: sum(r["report"]["reflex"] == v for r in s)
    m = dict(n=len(rs), n_pos=len(pos), n_neg=len(neg))
    m["sens"] = a(pos, "yes") / max(len(pos), 1)
    m["spec"] = a(neg, "no") / max(len(neg), 1)
    m["wrong"] = (a(pos, "no") + a(neg, "yes")) / max(len(pos) + len(neg), 1)
    m["unsure"] = (a(pos, "cannot tell") + a(neg, "cannot tell")) / max(len(pos) + len(neg), 1)
    d = [abs(r["report"]["lam"] - r["truth"]["lam"]) for r in pos if r["report"]["reflex"] == "yes" and r["report"]["lam"] is not None]
    m["lam_err"], m["lam_n"] = med(d), len(d)
    d = [abs(r["report"]["mu"] - r["truth"]["mu"]) for r in pos if r["report"]["reflex"] == "yes" and r["report"]["mu"] is not None]
    m["mu_err"] = med(d)
    d = [abs(r["report"]["delay"] - r["truth"]["delay"]) for r in pos if r["report"]["reflex"] == "yes" and r["report"].get("delay") is not None]
    m["delay_err"] = med(d)
    d = [abs(np.log(r["report"]["k80"] / r["truth"]["k80"])) for r in rs if r["report"]["k80"] and r["report"]["k80"] > 0]
    m["k_err"] = med(d)
    m["k_bad"] = sum(1 for r in rs if not r["report"]["k80"] or r["report"]["k80"] <= 0
                     or abs(np.log(r["report"]["k80"] / r["truth"]["k80"])) > np.log(1.5)) / max(len(rs), 1)
    m["q_err"] = med([abs(r["report"]["q_end"] - r["truth"]["q_s"]) for r in rs])
    m["stretches"] = float(np.mean([r["stretches"] for r in rs]))
    m["time"] = float(np.mean([r["time_s"] for r in rs]))
    m["over"] = sum(r["safety"]["overstretch"] > 0.02 for r in rs) / max(len(rs), 1)
    m["cut"] = sum(r["cutouts"] > 0 for r in rs) / max(len(rs), 1)
    m["resist"] = float(np.quantile([r["safety"]["max_resist"] for r in rs], 0.95))
    c = [r["report"]["q_lo"] <= r["truth"]["q_s"] <= r["report"]["q_hi"] for r in rs if "q_lo" in r["report"]]
    m["q_cover"] = float(np.mean(c)) if c else float("nan")
    c = [r["report"]["k_lo"] <= r["truth"]["k80"] <= r["report"]["k_hi"] for r in rs if "k_lo" in r["report"]]
    m["k_cover"] = float(np.mean(c)) if c else float("nan")
    c = [abs(r["report"]["lam"] - r["truth"]["lam"]) <= 2 * r["report"]["lam_sd"] for r in pos
         if r["report"].get("lam_sd") and r["report"]["reflex"] == "yes"]
    m["lam_cover"] = float(np.mean(c)) if c else float("nan")
    return m


def table(data, sel=lambda r: True, title="everyone"):
    print(f"\n{title}")
    hdr = ("method", "n", "sens", "spec", "wrong", "unsure", "lam_err", "lam_n", "mu_err", "k_err", "k_bad", "q_err",
           "stretch", "time", "over", "cut", "resist95", "q_cov", "k_cov", "lam_cov")
    print(("{:<12}" + "{:>8}" * (len(hdr) - 1)).format(*hdr))
    for k in ORDER:
        if k not in data:
            continue
        m = metrics([r for r in data[k].values() if sel(r)])
        print(("{:<12}{:>8d}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.3f}{:>8d}{:>8.3f}{:>8.3f}{:>8.2f}{:>8.3f}{:>8.1f}{:>8.0f}"
               "{:>8.2f}{:>8.2f}{:>8.1f}{:>8.2f}{:>8.2f}{:>8.2f}").format(
            k, m["n"], m["sens"], m["spec"], m["wrong"], m["unsure"], m["lam_err"], m["lam_n"], m["mu_err"], m["k_err"],
            m["k_bad"], m["q_err"], m["stretches"], m["time"], m["over"], m["cut"], m["resist"], m["q_cover"],
            m["k_cover"], m["lam_cover"]))


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(max(w, l), n + 1)) / 2 ** n)


def paired(data, a, b, f, name):
    """Patient by patient, is a closer to the truth than b?  f(r) is the error."""
    seeds = sorted(set(data[a]) & set(data[b]))
    w = l = 0
    for s in seeds:
        x, y = f(data[a][s]), f(data[b][s])
        if x is None or y is None or x == y:
            continue
        w += x < y
        l += x > y
    print(f"  {name}: {a} better on {w}, {b} better on {l}, p = {sign_p(w, l):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args()
    d = load(a.folder)
    table(d)
    for name, f in (("no reflex", lambda r: not r["truth"]["reflex"]),
                    ("reflex in reach", lambda r: r["truth"]["reach"]),
                    ("guarding", lambda r: r["truth"]["guard"]),
                    ("voluntary effort", lambda r: r["truth"]["active"]),
                    ("contracture", lambda r: "contracture" in r["truth"]["phenotype"]),
                    ("mild", lambda r: r["truth"]["phenotype"].startswith("mild")),
                    ("severe", lambda r: r["truth"]["phenotype"].startswith("severe"))):
        table(d, f, name)
