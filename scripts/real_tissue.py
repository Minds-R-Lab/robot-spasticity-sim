"""The tissue model on one real recording: the slow stretches of one participant in the public Patterson et al. 2022 data, the only raw trace
it holds.  Fit the first three stretches with the end of the range hidden; predict the end of the range on the other three.

usage: real_tissue.py s40_raw.csv"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elbow.model import WIN, Data, Post                 # noqa: E402

t, state, rep, pos, vel, tor, acc, *_ = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)).T
q0 = pos.min()
q = pos - q0                                           # extension is positive: 70 deg at q = 0, 150 deg at the top
print("rate", round(1 / np.median(np.diff(t[:500])), 1), "Hz; range", round(q.max(), 3), "rad; reps", sorted(set(rep)))


def rows(mask, down=1):
    """Rows at 200 Hz in the assessor's form: time, angle, speed, torque, EMG."""
    i = np.where(mask)[0][::down]
    return np.column_stack([t[i], q[i], vel[i], tor[i], np.zeros(len(i))])


def out_back(r_set, q_cut):
    d = Data()
    for rr in r_set:
        for st in (3.0, 1.0):                           # 3 extends (stretches the flexors), 1 returns
            m = (rep == rr) & (state == st) & (q <= q_cut)
            x = rows(m)
            if len(x) >= 3 * WIN:
                d.add(x, 0, 0, light=True)
    return d


info = dict(G_hat=0.05, phi_hat=0.0, rom_hat=float(q.max()), hold=0.0)    # the forearm weight was supported in this study
top = q.max()
test = (state == 3.0) & (rep >= 4)
bins = np.linspace(0.05, top - 0.01, 12)
truth = np.array([tor[test & (q >= a) & (q < b)].mean() for a, b in zip(bins[:-1], bins[1:])])
mid = 0.5 * (bins[:-1] + bins[1:])
print("mean torque on stretches 4 to 6, by angle:", np.round(truth, 2))
for frac in (1.0, 0.9, 0.8, 0.7):
    d = out_back((1.0, 2.0, 3.0), q0 * 0 + frac * top)
    post = Post(d, info, False)
    qs, C = post.passive_samples(np.random.default_rng(0), n=400)
    cm = np.median(C, axis=0)
    # the curve is zero at q = 0 by construction; align on the part of the range that was seen
    seen = mid <= frac * top
    off = np.median(truth[seen] - np.interp(mid[seen], qs, cm))
    pred = np.interp(mid, qs, cm) + off
    hid = ~seen
    print(f"seen to {frac:.0%} of range: rms in seen {np.sqrt(np.mean((pred[seen] - truth[seen]) ** 2)):.3f} N m"
          + (f", hidden: rms {np.sqrt(np.mean((pred[hid] - truth[hid]) ** 2)):.3f} N m, at the top {pred[-1]:.2f} against {truth[-1]:.2f}" if hid.any() else "")
          + f", sigma {post.sig:.3f}")
    s, e = (mid >= 0.7 * top) & (mid <= 0.95 * top), None
print("slope over 70 to 90% of range, measured:", round(float(np.polyfit(mid[(mid >= .7 * top) & (mid <= .9 * top)], truth[(mid >= .7 * top) & (mid <= .9 * top)], 1)[0]), 2), "N m/rad")
hold = (state == 4.0)
for r in sorted(set(rep[hold]))[:3]:
    m = hold & (rep == r)
    print(f"hold at the top, stretch {int(r)}: torque {tor[m][:200].mean():.2f} falling to {tor[m][-200:].mean():.2f} N m in {t[m][-1] - t[m][0]:.1f} s")
