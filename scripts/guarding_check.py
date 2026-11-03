"""After the test: which way is the range wrong in guarding patients?  Not a protocol endpoint."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import load                       # noqa: E402

for pop in ("usual", "stress"):
    d = load(f"results/test/{pop}")
    print(pop)
    for m in ("rule_t", "l2a_t", "rule_e", "l2a_e"):
        rs = [r for r in d[m].values() if r["truth"]["guard"]]
        e = np.array([r["report"]["q_end"] - r["truth"]["q_s"] for r in rs])
        c = [r["report"]["q_lo"] <= r["truth"]["q_s"] <= r["report"]["q_hi"] for r in rs if "q_lo" in r["report"]]
        w = [r["report"]["q_hi"] - r["report"]["q_lo"] for r in rs if "q_lo" in r["report"]]
        print(f"  {m:<8} n {len(rs)}  median error {np.median(e):+.3f}  over by >0.2 rad {(e > 0.2).sum()}  under by >0.2 rad {(e < -0.2).sum()}"
              f"  past end {sum(r['safety']['overstretch'] > 0.02 for r in rs)}"
              + (f"  cover {np.mean(c):.2f}  median width {np.median(w):.2f}" if c else ""))
