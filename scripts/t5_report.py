"""Scoring of PROTOCOL5.md."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import sign_p                        # noqa: E402
from report2 import load, med, metrics, score_one    # noqa: E402

ALPHA = 0.05 / 20


def said(r):
    return r["report"]["reflex"] == "yes" and r["report"].get("lam") is not None


def e_k(r):
    k = r["report"]["k80"]
    return abs(np.log(k / r["truth"]["k80"])) if k and k > 0 else 9.0


d = load(sys.argv[1])
for t in ("lit_early", "myo_early"):
    print(f"\n{t}")
    for m in ("pub_e", "l2a_e", "pub_t", "l2a_t"):
        x = metrics(list(d[t][m].values()))
        print(f"  {m}: n {x['n']} score {x['score']:.3f} sens {x['sens']:.3f} spec {x['spec']:.3f} gspec {x['gspec']:.2f} lam {x['lam']:.3f} bias {x['lam_b']:+.3f} mu {x['mu']:.3f} k {x['k']:.3f} q {x['q']:.3f} str {x['stretch']:.1f} time {x['time']:.0f} over {x['over']:.3f} cut {x['cut']:.3f} cover {x['qcov']:.2f} {x['kcov']:.2f} {x['lcov']:.2f}")
    for a, b in (("l2a_e", "pub_e"), ("l2a_t", "pub_t")):
        A, B = d[t][a], d[t][b]
        keys = sorted(set(A) & set(B))
        both = [k for k in keys if A[k]["truth"]["reach"] and said(A[k]) and said(B[k])]
        for nm, f, ks in (("threshold", lambda r: abs(r["report"]["lam"] - r["truth"]["lam"]), both), ("slope", lambda r: abs(r["report"]["mu"] - r["truth"]["mu"]), both),
                          ("stiffness", e_k, keys), ("range", lambda r: abs(r["report"]["q_end"] - r["truth"]["q_s"]), keys)):
            x, y = np.array([f(A[k]) for k in ks]), np.array([f(B[k]) for k in ks])
            w, l = int((x < y).sum()), int((x > y).sum())
            p = sign_p(w, l)
            print(f"  {a} v {b} {nm}: n {len(ks)} {med(x):.3f} v {med(y):.3f}  {w} to {l}  p {p:.2g} {'SHOWN' if p < ALPHA else ''}")
        ks = [k for k in keys if score_one(A[k]) is not None]
        x, y = np.array([score_one(A[k]) for k in ks]), np.array([score_one(B[k]) for k in ks])
        w, l = int((x > y).sum()), int((x < y).sum())
        p = sign_p(w, l)
        print(f"  {a} v {b} decision: n {len(ks)} {x.mean():.3f} v {y.mean():.3f}  {w} to {l}  p {p:.2g} {'SHOWN' if p < ALPHA else ''}")
