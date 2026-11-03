"""Score a folder of second-generation runs: folder/<world>_<scenario>/*.jsonl."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ORDER = ["pub_t", "short_t", "l2a_t", "l2a_t6", "pub_e", "short_e", "l2a_e", "l2a_e6", "l2a_e12", "l2a_e_rand", "l2a_e_asc", "l2a_e_fixed"]


def load(folder):
    out = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(folder + "/*/*.jsonl"):
        tag = os.path.basename(os.path.dirname(f))
        for line in open(f):
            r = json.loads(line)
            out[tag][r["method"]][(r["seed"], r["session"])] = r
    return out


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def score_one(r):
    t = r["truth"]
    if not (t["reach"] or not t["reflex"]):
        return None
    want = "yes" if t["reach"] else "no"
    c = r["report"]["reflex"]
    return 0.5 if c == "cannot tell" else float(c == want)


def metrics(rs):
    pos = [r for r in rs if r["truth"]["reach"]]
    neg = [r for r in rs if not r["truth"]["reflex"]]
    gneg = [r for r in neg if r["truth"]["guard"]]
    a = lambda s, v: np.mean([r["report"]["reflex"] == v for r in s]) if s else float("nan")
    d = [r["report"]["reflex"] == "yes" and r["report"]["lam"] is not None for r in pos]
    yes = [r for r, k in zip(pos, d) if k]
    m = dict(n=len(rs), score=float(np.mean([x for x in map(score_one, rs) if x is not None])), sens=a(pos, "yes"), spec=a(neg, "no"),
             gspec=a(gneg, "no"), unsure=a(pos + neg, "cannot tell"))
    m["lam"] = med([abs(r["report"]["lam"] - r["truth"]["lam"]) for r in yes])
    m["lam_b"] = med([r["report"]["lam"] - r["truth"]["lam"] for r in yes])
    m["mu"] = med([abs(r["report"]["mu"] - r["truth"]["mu"]) for r in yes if r["report"]["mu"] is not None])
    m["delay"] = med([abs(r["report"]["delay"] - r["truth"]["delay"]) for r in yes if r["report"].get("delay") is not None])
    m["k"] = med([abs(np.log(r["report"]["k80"] / r["truth"]["k80"])) if (r["report"]["k80"] or 0) > 0 else 9.0 for r in rs])
    m["q"] = med([abs(r["report"]["q_end"] - r["truth"]["q_s"]) for r in rs])
    m["stretch"] = float(np.mean([r["stretches"] for r in rs]))
    m["time"] = float(np.mean([r["time_s"] for r in rs]))
    m["over"] = float(np.mean([r["safety"]["overstretch"] > 0.02 for r in rs]))
    m["cut"] = float(np.mean([r["cutouts"] > 0 for r in rs]))
    m["qcov"] = np.mean([r["report"]["q_lo"] <= r["truth"]["q_s"] <= r["report"]["q_hi"] for r in rs if "q_lo" in r["report"]]) if rs and "q_lo" in rs[0]["report"] else float("nan")
    m["kcov"] = np.mean([r["report"]["k_lo"] <= r["truth"]["k80"] <= r["report"]["k_hi"] for r in rs if "k_lo" in r["report"]]) if rs and "k_lo" in rs[0]["report"] else float("nan")
    x = [abs(r["report"]["lam"] - r["truth"]["lam"]) <= 2 * r["report"]["lam_sd"] for r in yes if r["report"].get("lam_sd")]
    m["lcov"] = float(np.mean(x)) if x else float("nan")
    return m


def retest(v):
    """Test-retest of the threshold across two sessions: standard error of measurement, smallest detectable change, ICC(2,1)."""
    a, b = [], []
    for (seed, s), r in v.items():
        if s == 1 and (seed, 2) in v:
            o = v[(seed, 2)]
            if r["truth"]["reach"] and r["report"]["reflex"] == "yes" and o["report"]["reflex"] == "yes" \
                    and r["report"]["lam"] is not None and o["report"]["lam"] is not None:
                a.append(r["report"]["lam"])
                b.append(o["report"]["lam"])
    if len(a) < 5:
        return None
    a, b = np.array(a), np.array(b)
    n = len(a)
    m = (a + b) / 2
    msb = 2 * m.var(ddof=1)
    msw = ((a - b) ** 2).sum() / (2 * n)
    mst = n * (a.mean() - b.mean()) ** 2
    mse = (((a - m) ** 2 + (b - m) ** 2).sum() - mst) / max(n - 1, 1)
    icc = (msb - mse) / (msb + mse + 2 * (mst - mse) / n)
    sem = (a - b).std(ddof=1) / np.sqrt(2)
    return dict(n=n, sem=float(sem), mdc=float(1.96 * np.sqrt(2) * sem), icc=float(icc))


def table(d, keys=None):
    h = f"{'tag':<16}{'method':<12}{'n':>4}{'score':>7}{'sens':>6}{'spec':>6}{'gspec':>6}{'unsr':>6}{'lam':>7}{'bias':>7}{'mu':>7}{'dly':>7}{'k':>7}{'q':>7}{'str':>6}{'time':>6}{'over':>6}{'cut':>6}{'qcov':>6}{'kcov':>6}{'lcov':>6}{'mdc':>7}{'icc':>6}"
    print(h)
    for t in sorted(d):
        for m in ORDER:
            if m not in d[t]:
                continue
            rs = [r for (s, k), r in d[t][m].items() if k == 1] or list(d[t][m].values())
            x = metrics(rs)
            rt = retest(d[t][m])
            print(f"{t:<16}{m:<12}{x['n']:>4}{x['score']:>7.3f}{x['sens']:>6.2f}{x['spec']:>6.2f}{x['gspec']:>6.2f}{x['unsure']:>6.2f}{x['lam']:>7.3f}{x['lam_b']:>7.3f}{x['mu']:>7.3f}{x['delay']:>7.3f}{x['k']:>7.3f}{x['q']:>7.3f}{x['stretch']:>6.1f}{x['time']:>6.0f}{x['over']:>6.2f}{x['cut']:>6.2f}{x['qcov']:>6.2f}{x['kcov']:>6.2f}{x['lcov']:>6.2f}"
                  + (f"{rt['mdc']:>7.3f}{rt['icc']:>6.2f}" if rt else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args()
    table(load(a.folder))
