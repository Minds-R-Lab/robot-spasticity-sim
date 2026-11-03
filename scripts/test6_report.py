"""The scoring fixed in PROTOCOL6.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import sign_p                                         # noqa: E402
from report2 import load, med, metrics, retest, score_one        # noqa: E402

R = np.random.default_rng(0)


def hybrid(r):
    """The assessor's session, with threshold and decision from the onset line."""
    p = dict(r["report"])
    p["reflex"], p["lam"], p["mu"] = p["onset_reflex"], p["onset_lam"], p["onset_mu"]
    return dict(r, method=r["method"].replace("l2a", "hyb"), report=p)


def arms(d):
    for t in d:
        for m in ("l2a_e", "l2a_t"):
            if m in d[t]:
                d[t][m.replace("l2a", "hyb")] = {k: hybrid(r) for k, r in d[t][m].items()}
    return d


def said(r):
    return r["report"]["reflex"] == "yes" and r["report"].get("lam") is not None


def lam_e(r):
    return abs(r["report"]["lam"] - r["truth"]["lam"])


def k_e(r):
    k = r["report"]["k80"]
    return abs(np.log(k / r["truth"]["k80"])) if k and k > 0 else 9.0


def q_e(r):
    return abs(r["report"]["q_end"] - r["truth"]["q_s"])


def paired(A, B, f, sel):
    ks = [k for k in sorted(set(A) & set(B)) if k[1] == 1 and sel(A[k], B[k])]
    x, y = np.array([f(A[k]) for k in ks]), np.array([f(B[k]) for k in ks])
    d = x - y
    if len(d) == 0:
        return None
    b = [np.median(d[R.integers(0, len(d), len(d))]) for _ in range(3000)]
    w, l = int((d < 0).sum()), int((d > 0).sum())
    return dict(n=len(d), a=med(x), b=med(y), diff=float(np.median(d)), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)), w=w, l=l, p=sign_p(w, l))


def dec_diff(A, B):
    ks = [k for k in sorted(set(A) & set(B)) if k[1] == 1 and score_one(A[k]) is not None]
    x, y = np.array([score_one(A[k]) for k in ks]), np.array([score_one(B[k]) for k in ks])
    d = x - y
    b = [d[R.integers(0, len(d), len(d))].mean() for _ in range(3000)]
    return dict(n=len(d), a=float(x.mean()), b=float(y.mean()), diff=float(d.mean()), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)),
                w=int((d > 0).sum()), l=int((d < 0).sum()), p=sign_p(int((d > 0).sum()), int((d < 0).sum())))


def show(name, x, margin=None, lower_better=True):
    if x is None:
        print(f"| {name} | 0 | | | | | |")
        return
    verdict = ""
    if margin is not None:
        verdict = "equivalent" if x["lo"] > -margin and x["hi"] < margin else ("better" if (x["hi"] < -margin if lower_better else x["lo"] > margin) else "not shown")
    print(f"| {name} | {x['n']} | {x['a']:.3f} | {x['b']:.3f} | {x['diff']:+.3f} ({x['lo']:+.3f} to {x['hi']:+.3f}) | {x['w']} to {x['l']}, p {x['p']:.2g} | {verdict} |")


HEAD = "| Endpoint | n | A | B | A - B (95% CI) | Patients | Verdict |\n| --- | --- | --- | --- | --- | --- | --- |"


def compare(d, t, a, b, margins):
    A, B = d[t][a], d[t][b]
    both = lambda x, y: x["truth"]["reach"] and said(x) and said(y)
    print(f"\n#### {t}: {a} (A) against {b} (B)\n\n{HEAD}")
    show("threshold error (rad)", paired(A, B, lam_e, both), margins.get("lam"))
    show("stiffness error", paired(A, B, k_e, lambda x, y: True), margins.get("k"))
    show("range error (rad)", paired(A, B, q_e, lambda x, y: True), margins.get("q"))
    x = dec_diff(A, B)
    m = margins.get("dec")
    verdict = "" if m is None else ("equivalent" if x["lo"] > -m and x["hi"] < m else ("better" if x["lo"] > m else "not shown"))
    print(f"| decision score | {x['n']} | {x['a']:.3f} | {x['b']:.3f} | {x['diff']:+.3f} ({x['lo']:+.3f} to {x['hi']:+.3f}) | {x['w']} to {x['l']}, p {x['p']:.2g} | {verdict} |")


def table(d, tags, methods, rt=False):
    print("\n| Arm | n | Score | Sens | Spec | Spec, guarding (k/n) | Threshold | Bias | Slope | Stiffness | Range | Stretches | Time | Cut-outs/session | Cut-outs/stretch | Past end | Cover lam | Cover lam, raw | MDC | ICC |")
    print("| --- " * 20 + "|")
    for t in tags:
        for m in methods:
            if m not in d[t]:
                continue
            rs = [r for k, r in d[t][m].items() if k[1] == 1]
            x = metrics(rs)
            g = [r for r in rs if r["truth"]["guard"] and not r["truth"]["reflex"]]
            gk = sum(r["report"]["reflex"] == "no" for r in g)
            cuts = sum(r["cutouts"] for r in rs)
            st = sum(r["stretches"] for r in rs)
            yes = [r for r in rs if r["truth"]["reach"] and said(r)]
            raw = [abs(r["report"]["lam"] - r["truth"]["lam"]) <= 2 * r["report"]["lam_sd_raw"] for r in yes if r["report"].get("lam_sd_raw")]
            rr = retest(d[t][m]) if rt else None
            print(f"| {t} {m} | {x['n']} | {x['score']:.3f} | {x['sens']:.3f} | {x['spec']:.3f} | {gk}/{len(g)} | {x['lam']:.3f} | {x['lam_b']:+.3f} | {x['mu']:.3f} | {x['k']:.3f} | {x['q']:.3f} | {x['stretch']:.2f} | {x['time']:.0f} | "
                  f"{x['cut']:.3f} | {cuts / max(st, 1):.4f} | {x['over']:.3f} | {x['lcov']:.2f} | {np.mean(raw) if raw else float('nan'):.2f} | "
                  + (f"{rr['mdc']:.3f} | {rr['icc']:.2f} |" if rr else " | |"))


if __name__ == "__main__":
    d = arms(load(sys.argv[1]))
    base = ["lit_base", "myo_base"]
    print(f"# Test 6: {sum(len(v) for t in d.values() for k, v in t.items() if not k.startswith('hyb'))} sessions")
    print("\n## Base, session 1, with retest")
    table(d, base, ["pub_e", "short_e", "l2a_e", "hyb_e", "pub_t", "short_t", "l2a_t", "hyb_t"], rt=True)
    print("\n## Pre-specified comparisons")
    for t in base:
        compare(d, t, "short_e", "pub_e", dict(lam=0.03, dec=0.03))
        compare(d, t, "hyb_e", "short_e", dict(lam=0.03, k=0.0, q=0.0))
        compare(d, t, "l2a_e", "short_e", {})
        compare(d, t, "short_t", "pub_t", dict(lam=0.03, dec=0.03))
        compare(d, t, "hyb_t", "short_t", dict(lam=0.03, k=0.0, q=0.0))
        compare(d, t, "l2a_t", "short_t", {})
    print("\n## Scenarios")
    sc = [t for t in sorted(d) if t not in base]
    table(d, sc, ["pub_e", "short_e", "l2a_e", "hyb_e", "pub_t", "short_t", "l2a_t", "hyb_t"])
    for t in sc:
        compare(d, t, "hyb_e", "short_e", {})
