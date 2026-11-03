"""The scoring fixed in PROTOCOL4.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elbow.assessors import RULE_LAM_OFF, RULE_MU_OFF, rule_report      # noqa: E402
from report import sign_p                                               # noqa: E402
from report2 import load, med, metrics, retest, score_one               # noqa: E402

ALPHA = 0.05 / 20


def boot(x, rng, n=4000):
    x = np.asarray(x)
    if len(x) == 0:
        return float("nan"), float("nan")
    b = np.median(rng.choice(x, (n, len(x))), axis=1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def recut(r, n=None, calibrated=True):
    """The rule's session scored again: fewer stretches, or without the corrections."""
    emg = r["method"] == "pub_e"
    c = [tuple(x) for x in r["report"]["catches"]]
    c = c if n is None else c[:n]
    rep = rule_report(c, r["report"]["q_end"], r["report"]["k80"], None, RULE_MU_OFF[emg] if calibrated else 0.0, RULE_LAM_OFF[emg] if calibrated else 0.0)
    return dict(r, report=rep)


def said(r):
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


def compare(a, b, name_a, name_b, rng, count=None):
    keys = sorted(k for k in set(a) & set(b) if k[1] == 1)
    both = lambda k: a[k]["truth"]["reach"] and said(a[k]) and said(b[k])
    rows = []
    for nm, f, sel in (("threshold", e_lam, both), ("speed slope", e_mu, both), ("stiffness", e_k, lambda k: True), ("end of range", e_q, lambda k: True)):
        ks = [k for k in keys if sel(k)]
        x, y = np.array([f(a[k]) for k in ks]), np.array([f(b[k]) for k in ks])
        w, l = int((x < y).sum()), int((x > y).sum())
        p = sign_p(w, l)
        cx, cy = boot(x, rng), boot(y, rng)
        rows.append(f"| {nm} | {len(ks)} | {med(x):.3f} ({cx[0]:.3f} to {cx[1]:.3f}) | {med(y):.3f} ({cy[0]:.3f} to {cy[1]:.3f}) | {w} to {l} | {p:.2g} | {'yes' if p < ALPHA else 'no'} |")
        if count is not None:
            count.append((nm, name_a, name_b, med(x), med(y), w, l, p))
    ks = [k for k in keys if score_one(a[k]) is not None]
    x, y = np.array([score_one(a[k]) for k in ks]), np.array([score_one(b[k]) for k in ks])
    w, l = int((x > y).sum()), int((x < y).sum())
    p = sign_p(w, l)
    rows.append(f"| reflex decision (mean, higher is better) | {len(ks)} | {x.mean():.3f} | {y.mean():.3f} | {w} to {l} | {p:.2g} | {'yes' if p < ALPHA else 'no'} |")
    if count is not None:
        count.append(("decision", name_a, name_b, float(x.mean()), float(y.mean()), w, l, p))
    print(f"\n#### {name_a} against {name_b}\n")
    print(f"| Endpoint | n | {name_a} | {name_b} | Patients better | p | Shown |\n| --- | --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))


def line(name, rs, v=None):
    x = metrics(rs)
    rt = retest(v) if v is not None else None
    print(f"| {name} | {x['n']} | {x['score']:.3f} | {x['sens']:.3f} | {x['spec']:.3f} | {x['gspec']:.2f} | {x['unsure']:.3f} | {x['lam']:.3f} | {x['lam_b']:+.3f} | {x['mu']:.3f} | {x['delay']:.3f} | {x['k']:.3f} | {x['q']:.3f} | {x['stretch']:.1f} | {x['time']:.0f} | {x['over']:.3f} | {x['cut']:.3f} | {x['qcov']:.2f} | {x['kcov']:.2f} | {x['lcov']:.2f} | "
          + (f"{rt['sem']:.3f} | {rt['mdc']:.3f} | {rt['icc']:.2f} |" if rt else "| | |"))


HEAD = ("| Assessor | n | Score | Sens | Spec | Spec, guarding | Cannot tell | Threshold | Bias | Slope | Delay | Stiffness | Range | Stretches | Time | Past end | Cut-outs | Cover q | Cover k | Cover lam | SEM | MDC | ICC |\n"
        + "| --- " * 23 + "|")


def groups(d):
    gs = (("no reflex", lambda r: not r["truth"]["reflex"]), ("reflex in reach", lambda r: r["truth"]["reach"]), ("guarding", lambda r: r["truth"]["guard"]),
          ("effort", lambda r: r["truth"]["active"]), ("contracture", lambda r: "contracture" in r["truth"]["phenotype"]),
          ("mild", lambda r: r["truth"]["phenotype"].startswith("mild")), ("moderate", lambda r: r["truth"]["phenotype"].startswith("moderate")),
          ("severe", lambda r: r["truth"]["phenotype"].startswith("severe")))
    for name, f in gs:
        print(f"\n#### {name}\n\n{HEAD}")
        for m in ("pub_e", "l2a_e", "pub_t", "l2a_t"):
            rs = [r for k, r in d[m].items() if k[1] == 1 and f(r)]
            if len(rs) >= 3:
                line(m, rs)


def main(root):
    rng = np.random.default_rng(0)
    d = load(root)
    d50, orc = load(root + "/d50"), load(root + "/oracle")
    out = []
    print(f"# Test 4: {sum(len(v) for t in d.values() for v in t.values())} sessions, {sum(len(v) for t in list(d50.values()) + list(orc.values()) for v in t.values())} more in the delay runs")
    for t in ("lit_base", "myo_base"):
        x = d[t]
        print(f"\n## {t}\n\n{HEAD}")
        for m in ("pub_e", "l2a_e", "l2a_e_rand", "l2a_e_asc", "pub_t", "l2a_t"):
            if m in x:
                line(m, [r for k, r in x[m].items() if k[1] == 1], x[m])
        for m in ("pub_e", "pub_t"):
            raw = {k: recut(r, None, False) for k, r in x[m].items()}
            line(m + " as published", [r for k, r in raw.items() if k[1] == 1], raw)
            for cal in (True, False):
                c4 = {k: recut(r, 4, cal) for k, r in x[m].items()}
                line(m + f" first 4, {'calibrated' if cal else 'as published'}", [r for k, r in c4.items() if k[1] == 1], c4)
        print("\n### Primary")
        compare(x["l2a_e"], x["pub_e"], "l2a_e", "pub_e calibrated", rng, out)
        compare(x["l2a_t"], x["pub_t"], "l2a_t", "pub_t calibrated", rng, out)
        print("\n### Secondary: as published, and cut to the law's effort")
        for m, l in (("pub_e", "l2a_e"), ("pub_t", "l2a_t")):
            compare(x[l], {k: recut(r, None, False) for k, r in x[m].items()}, l, m + " as published", rng)
            compare(x[l], {k: recut(r, 4, True) for k, r in x[m].items()}, l, m + " first 4 calibrated", rng)
        print("\n### Delay")
        for m in ("l2a_e", "l2a_t"):
            rs = [r for k, r in x[m].items() if k[1] == 1 and r["truth"]["reach"] and said(r) and r["report"].get("delay") is not None]
            e, f = np.array([abs(r["report"]["delay"] - r["truth"]["delay"]) for r in rs]), np.array([abs(0.05 - r["truth"]["delay"]) for r in rs])
            print(f"\n{m}: identified delay error {med(e) * 1000:.1f} ms; fixed 50 ms {med(f) * 1000:.1f} ms; better on {int((e < f).sum())}, worse on {int((e > f).sum())}, n = {len(e)}")
            compare(x[m], d50[t][m], m, m + " fixed 50 ms", rng)
            compare(x[m], orc[t][m], m, m + " true delay", rng)
        print("\n### Guarding patients: range\n")
        for m in ("pub_t", "l2a_t", "pub_e", "l2a_e"):
            rs = [r for k, r in x[m].items() if k[1] == 1 and r["truth"]["guard"]]
            e = np.array([r["report"]["q_end"] - r["truth"]["q_s"] for r in rs])
            print(f"{m}: n {len(e)}, median {np.median(e):+.3f}, over by more than 0.2 rad {int((e > 0.2).sum())}, under by more than 0.2 rad {int((e < -0.2).sum())}")
        print("\n### Groups")
        groups(x)
    print("\n## Scenarios")
    for t in ("lit_long", "lit_gspeed", "lit_noisy", "myo_gspeed"):
        x = d[t]
        print(f"\n### {t}\n\n{HEAD}")
        for m in ("pub_e", "l2a_e", "pub_t", "l2a_t"):
            line(m, [r for k, r in x[m].items() if k[1] == 1])
        raw = {k: recut(r, None, False) for k, r in x["pub_e"].items()}
        line("pub_e as published", [r for k, r in raw.items() if k[1] == 1])
        compare(x["l2a_e"], x["pub_e"], "l2a_e", "pub_e calibrated", rng)
        compare(x["l2a_t"], x["pub_t"], "l2a_t", "pub_t calibrated", rng)
    print("\n## The twenty primary tests\n")
    print("| Endpoint | A | B | A value | B value | A better | B better | p | Shown |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for nm, a, b, x, y, w, l, p in out:
        print(f"| {nm} | {a} | {b} | {x:.3f} | {y:.3f} | {w} | {l} | {p:.2g} | {'yes' if p < ALPHA else 'no'} |")
    json.dump(out, open(root + "/primary.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    main(ap.parse_args().root)
