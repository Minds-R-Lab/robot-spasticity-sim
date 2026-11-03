"""The second video: one TSRT-4 session in the realistic world of test 7 (record_real.py), on a patient with no reflex.

Left: the 2-joint MyoSuite arm, the robot lever at the motor's angle, and a close-up of the soft cuff (the spring between
lever and forearm).  Right: the motor encoder against the true forearm angle, the torque sensor, the EMG, and the onset line.
Usage: python scripts/make_video_real.py results/video/real_5057.npz out.mp4 [--preview DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from elbow.scene import Painter                       # noqa: E402
from make_video import BLUE, FPS, GREY, H, INK, ORANGE, RED, RW, W, card, phases, timeline   # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def onsets(T, meta):
    """The time and angle of each onset the rule reported: where the motor angle passed it during that fast stretch."""
    fast = [(a, b) for a, b, lab, _ in phases(meta, meta["time"]) if lab == "fast"]
    out = []
    for (a, b), c in zip(fast, meta["catches"]):
        if c[1] is None:
            continue
        i, j = np.searchsorted(T[:, 0], [a, b])
        k = i + int(np.argmax(T[i:j, 1] >= c[1]))
        out.append((float(T[k, 0]), float(c[1]), float(c[2])))
    return out


def caption(meta, t, ons):
    lab, v, n_fast, a0, b0, prev, last, vf = "rest", 0.0, 0, 0.0, 0.0, "", "", 0.0
    for a, b, l, s in phases(meta, meta["time"]):
        if l == "fast" and a <= t:
            n_fast, vf = n_fast + 1, s
        if a <= t < b:
            lab, v, a0, b0, prev = l, s, a, b, last
        last = l
    if t < 1.0:
        return "The robot holds the arm still.", "This patient has no stretch reflex. The robot does not know that."
    if lab in ("slow", "slowback") or (lab == "rest" and b0 - a0 < 1.5 and prev == "slow"):
        return "One very slow stretch.", "It gives the tissue curve and the end of range."
    if lab == "rest" and b0 - a0 < 1.5 and prev == "fast":
        lab, v = "fast", vf
    elif lab == "rest" and b0 - a0 < 1.5:
        return "The robot holds the arm.", ""
    if lab == "fast":
        seen = any(a0 <= o[0] <= t for o in ons)
        if not seen:
            return f"Fast stretch {n_fast} of 4, at {v:.1f} rad/s.", "The soft cuff lets the forearm run ahead of the motor (spring)."
        return f"Fast stretch {n_fast} of 4, at {v:.1f} rad/s.", "EMG jumps at the start: motion, not a reflex. The rule marks an onset."
    if n_fast >= 4 and t > meta["time"] - 1.0:
        return "Done: the robot calls a reflex.", "Truth: this patient has none."
    if lab == "back":
        return f"Fast stretch {n_fast} of 4 done.", "The robot brings the arm back."
    return "Rest for 10 s.", ""


class Panels:
    def __init__(self, T, meta):
        self.T, self.m = T, meta
        self.ons = onsets(T, meta)
        rest = T[(T[:, 0] > 0.2) & (T[:, 0] < 0.95), 3]
        self.hold = float(np.median(rest))
        self.fig = plt.figure(figsize=((W - RW) / 100, H / 100), dpi=100)
        gs = self.fig.add_gridspec(5, 1, height_ratios=[0.55, 1, 1, 1, 1.7], hspace=0.6, left=0.15, right=0.96, top=0.98, bottom=0.07)
        self.ax_t = self.fig.add_subplot(gs[0])
        self.ax = [self.fig.add_subplot(gs[i]) for i in (1, 2, 3)]
        self.ax_o = self.fig.add_subplot(gs[4])
        tau = T[:, 3] - self.hold
        self.lims = [(-0.1, 1.15), (-1.5, max(4.0, float(np.percentile(tau, 99.8)) + 0.5)),
                     (0, max(0.8, float(np.percentile(T[:, 4], 99.8)) * 1.1))]

    def draw(self, t):
        T, m = self.T, self.m
        i = int(np.searchsorted(T[:, 0], t))
        a = self.ax_t
        a.clear(); a.axis("off")
        c1, c2 = caption(m, t, self.ons)
        a.text(0.0, 0.95, c1, fontsize=14, fontweight="bold", color=INK, va="top", transform=a.transAxes)
        a.text(0.0, 0.25, c2, fontsize=11.5, color=INK, va="top", transform=a.transAxes)
        ax = self.ax
        for x in ax:
            x.clear(); x.set_xlim(0, m["time"]); x.tick_params(labelsize=8)
        ax[0].plot(T[:i, 0], T[:i, 2], color=INK, lw=1.0, label="true forearm angle")
        ax[0].plot(T[:i, 0], T[:i, 1], color=BLUE, lw=0.9, ls="--", label="robot's motor encoder")
        ax[0].set_ylim(*self.lims[0]); ax[0].set_ylabel("elbow angle\n(rad)", fontsize=9)
        ax[0].legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.32))
        ax[1].plot(T[:i, 0], T[:i, 3] - self.hold, color=BLUE, lw=0.7)
        ax[1].set_ylim(*self.lims[1]); ax[1].set_ylabel("torque sensor\n(N m)", fontsize=9)
        ax[2].plot(T[:i, 0], T[:i, 4], color=RED, lw=0.6)
        ax[2].set_ylim(*self.lims[2]); ax[2].set_ylabel("muscle\nsignal (EMG)", fontsize=9); ax[2].set_xlabel("time (s)", fontsize=9)
        for x in ax[:2]:
            x.set_xticklabels([])
        for to, c, u in self.ons:
            if to <= t:
                ax[0].plot([to], [c], "o", color=ORANGE, ms=6, zorder=4)
                ax[2].axvline(to, color=ORANGE, lw=0.8)
        a = self.ax_o
        a.clear()
        a.set_xlim(-0.1, 3.5); a.set_ylim(-0.3, 0.6); a.tick_params(labelsize=8)
        a.set_xlabel("speed at onset (rad/s)", fontsize=9); a.set_ylabel("onset angle\n(rad)", fontsize=9)
        pts = [(u, c) for to, c, u in self.ons if to <= t]
        if pts:
            x, y = np.array(pts).T
            a.plot(x, y, "o", color=ORANGE, ms=7, zorder=3, label="onsets the rule saw")
            a.legend(loc="lower right", fontsize=8, frameon=False)
        if t >= m["time"] - 1.0 and m["lam_hat"] is not None:
            a.plot([0.0], [m["lam_hat"]], "D", color=INK, ms=7, zorder=4)
            a.annotate("robot's threshold", (0.0, m["lam_hat"]), xytext=(10, -4), textcoords="offset points", fontsize=8.5, color=INK)
            a.text(0.1, 0.48, f"robot: reflex, threshold {m['lam_hat']:.2f} rad.   truth: no reflex", fontsize=10, color=RED, fontweight="bold")
        else:
            a.text(0.1, 0.48, "each fast stretch can add one point", fontsize=9.5, color=GREY)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]


def left(img, close, rate, gap):
    import matplotlib.figure as mf
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    f = mf.Figure(figsize=(RW / 100, H / 100), dpi=100)
    FigureCanvasAgg(f)
    a = f.add_axes([0, 0, 1, 1]); a.imshow(img); a.axis("off")
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    a.text(18, 34, "A more realistic world", fontsize=13, fontweight="bold", color=INK, bbox=box)
    a.text(18, 64, "MyoSuite arm with a shoulder (MuJoCo)", fontsize=9.5, color=INK, bbox=box)
    tag = "slow motion ×5" if rate < 1 else "fast forward ×10" if rate > 5 else "×2" if rate > 1 else "real time"
    a.text(RW - 18, 34, tag, fontsize=11, color="white", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="#444444" if rate != 1 else BLUE, ec="none"))
    h, w = close.shape[:2]
    s = min(h, w) // 2
    crop = close[h // 2 - s // 2: h // 2 + s // 2, w // 2 - s // 2: w // 2 + s // 2]
    ins = f.add_axes([0.03, 0.06, 0.4, 0.4 * RW / H])
    ins.imshow(crop); ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(ORANGE); sp.set_linewidth(2)
    ins.text(0.03, 0.97, f"close-up: the soft cuff\nforearm - motor = {gap:+.3f} rad", fontsize=8.5, color=INK, va="top",
             transform=ins.transAxes, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    a.text(RW - 18, H - 20, "lever at the motor's angle  ·  spring = soft cuff", fontsize=8.5, color=INK, ha="right")
    f.canvas.draw()
    return np.asarray(f.canvas.buffer_rgba())[:, :, :3]


def main(src, out, preview=None):
    z = np.load(src)
    T, meta = z["ticks"], json.loads(str(z["meta"]))
    ts, rates = timeline(meta, meta["time"])
    pan = Panels(T, meta)
    paint = Painter(RW, H, two=True, split=True)
    act = np.clip(T[:, 7:13], 0, 1)                   # raw activations: this patient's muscles stay nearly quiet
    sh = T[:, 6]

    def frame(t, r):
        k = min(int(np.searchsorted(T[:, 0], t)), len(T) - 1)
        img = paint.frame(T[k, 2], act[k], qm=T[k, 1], shoulder=sh[k])
        close = paint.closeup(0.26)
        return np.hstack([left(img, close, r, T[k, 2] - T[k, 1]), pan.draw(t)])

    m = meta
    intro = card(["The same robot in a more realistic world",
                  "A simulated patient with no stretch reflex, on a MyoSuite arm that has a shoulder.",
                  "Realistic EMG noise, a soft robot cuff slightly off the elbow axis, a spindle reflex law,",
                  "tissue with a history.  Protocol: TSRT-4, one slow stretch and four fast ones.",
                  "Simulation only.  A development patient, not from the test."], [26, 15, 15, 15, 12])
    rig = m["rigid"]
    outro = card([f"Robot's call: reflex, threshold {m['lam_hat']:.2f} rad.   Truth: no reflex.",
                  f"Same patient, cuff made rigid, everything else the same: {'no reflex. Correct.' if rig['reflex'] == 'no' else rig['reflex']}",
                  "In the seventh test, with every realism on, TSRT-4 called about 3 in 4 patients without a reflex spastic.",
                  "Turning off the soft cuff fixed most of it: how the robot holds the arm matters.",
                  "Simulation only."], [22, 15, 14, 14, 12])
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
        ons = onsets(T, meta)
        for t in [0.5, 6.0] + [o[0] + 0.05 for o in ons[:2]] + [ons[0][0] - 0.1, m["time"] - 0.2]:
            plt.imsave(Path(preview) / f"f_{t:05.1f}.png", frame(t, 0.2))
        plt.imsave(Path(preview) / "intro.png", intro)
        plt.imsave(Path(preview) / "outro.png", outro)
        return
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
    w.send(None)
    for _ in range(FPS * 7):
        w.send(np.ascontiguousarray(intro))
    for n, (t, r) in enumerate(zip(ts, rates)):
        last = frame(t, r)
        w.send(np.ascontiguousarray(last))
        if n % 200 == 0:
            print(n, len(ts), flush=True)
    last = frame(m["time"], 1.0)
    for _ in range(FPS * 3):
        w.send(np.ascontiguousarray(last))
    for _ in range(FPS * 8):
        w.send(np.ascontiguousarray(outro))
    w.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[3] if len(a) > 3 and a[2] == "--preview" else None)
