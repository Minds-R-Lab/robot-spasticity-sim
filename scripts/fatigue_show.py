"""Summarise fatigue_check runs against the real curve: ratio of the medians, with a bootstrap interval at stretch 3 and at the last three."""
import glob, json, sys
import numpy as np
REAL = np.array([7.17, 5.87, 5.0, 4.93, 4.82, 4.45, 4.84, 4.33, 4.33, 4.11, 4.51, 4.62, 3.91, 4.25, 4.4, 4.38, 4.21, 4.55, 4.42, 4.21])
print("real ", np.round(REAL[[0, 1, 2, 3, 4, 9, 19]] / REAL[0], 2))
r = np.random.default_rng(0)
for d in sorted(glob.glob(sys.argv[1] + "/*")):
    x = [json.loads(l)["tau"] for f in glob.glob(d + "/*.jsonl") for l in open(f)]
    if not x:
        continue
    x = np.array(x)
    c = lambda y: np.median(y, axis=0) / np.median(y, axis=0)[0]
    b = np.array([c(x[r.integers(0, len(x), len(x))]) for _ in range(500)])
    m = c(x)
    print(f"{d.split('/')[-1]:<16} n {len(x):>3} {np.round(m[[0, 1, 2, 3, 4, 9, 19]], 2)}  stretch 3 {np.round(np.quantile(b[:, 2], [.025, .975]), 2)}  stretch 20 {np.round(np.quantile(b[:, 19], [.025, .975]), 2)}  first {np.median(x[:, 0]):.2f} N m")
