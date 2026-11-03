"""What the rule achieves with fewer stretches: truncate each logged session to its first n stretches and score again.  The stretches are
in random order, so the first n are a fair subset."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elbow.assessors import RULE_LAM_OFF, RULE_MU_OFF, rule_report    # noqa: E402
from report2 import load, med, retest, score_one                      # noqa: E402


def cut(r, n, emg, calibrated=True):
    c = [tuple(x) if len(x) == 3 else (x[0], x[1], x[0]) for x in r["report"]["catches"]][:n]
    rep = rule_report(c, r["report"]["q_end"], r["report"]["k80"], None, RULE_MU_OFF[emg] if calibrated else 0.0, RULE_LAM_OFF[emg] if calibrated else 0.0)
    return dict(r, report=rep)


def table(folder, ns=(4, 5, 8, 10, 15, 20)):
    d = load(folder)
    print(f"{'tag':<14}{'method':<8}{'n':>4}{'score':>7}{'sens':>6}{'spec':>6}{'lam':>7}{'mu':>7}{'mdc':>7}{'icc':>6}")
    for t in sorted(d):
        for m in ("pub_e", "pub_t"):
            if m not in d[t]:
                continue
            for n in ns:
                v = {k: cut(r, n, m == "pub_e") for k, r in d[t][m].items()}
                rs = [r for (s, k), r in v.items() if k == 1]
                pos = [r for r in rs if r["truth"]["reach"]]
                neg = [r for r in rs if not r["truth"]["reflex"]]
                yes = [r for r in pos if r["report"]["reflex"] == "yes" and r["report"]["lam"] is not None]
                x = retest(v)
                print(f"{t:<14}{m:<8}{n:>4}{np.mean([y for y in map(score_one, rs) if y is not None]):>7.3f}"
                      f"{np.mean([r['report']['reflex'] == 'yes' for r in pos]):>6.2f}{np.mean([r['report']['reflex'] == 'no' for r in neg]):>6.2f}"
                      f"{med([abs(r['report']['lam'] - r['truth']['lam']) for r in yes]):>7.3f}{med([abs(r['report']['mu'] - r['truth']['mu']) for r in yes]):>7.3f}"
                      + (f"{x['mdc']:>7.3f}{x['icc']:>6.2f}" if x else ""))


if __name__ == "__main__":
    table(sys.argv[1])
