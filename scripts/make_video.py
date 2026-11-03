"""The explainer video: one recorded session (record_session.py) replayed on the MyoSuite elbow with a robot drawn around it.

Left: the arm, muscles coloured by activity.  Right: what the robot measures, and the onset line that gives the threshold.
Fast stretches play in slow motion and rests are fast-forwarded; the corner tag says which.
Usage: python scripts/make_video.py results/video/session_5004.npz out.mp4 [--preview DIR]
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
from elbow.scene import Painter                       # noqa: E402

W, H, RW, FPS = 1280, 720, 600, 30
BLUE, ORANGE, RED, GREY, INK = "#2a6f97", "#d9822b", "#c0392b", "#9a9a9a", "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def phases(meta, t_end):
    """(start, end, label, speed) for every stretch of time, from the session's marks."""
    mk = meta["marks"] + [[t_end, "end", 0.0]]
    return [(mk[i][0], mk[i + 1][0], mk[i][1], mk[i][2]) for i in range(len(mk) - 1) if mk[i + 1][0] > mk[i][0]]


def timeline(meta, t_end):
    """Session times to show, one per video frame, and the playback rate at each."""
    ts, rates = [], []
    ph = phases(meta, t_end)
    for k, (a, b, lab, v) in enumerate(ph):
        after_fast = k > 0 and ph[k - 1][2] == "fast"
        rate = 10.0 if lab == "rest" and b - a > 2 else 0.2 if lab == "fast" or (lab == "rest" and after_fast) else 2.0 if lab in ("slow", "slowback") else 1.0
        n = max(int(round((b - a) / rate * FPS)), 1)
        ts += list(np.linspace(a, b, n, endpoint=False))
        rates += [rate] * n
    return np.array(ts), np.array(rates)


def caption(meta, t):
    lab, v, n_fast = "rest", 0.0, 0
    for a, b, l, s in phases(meta, meta["time"]):
        if l == "fast" and a <= t:
            n_fast += 1
        if a <= t < b:
            lab, v = l, s
    total = len(meta["speeds"])
    if t < 1.0:
        return "Step 1. The robot holds the arm still.", "It records the quiet muscle signal (EMG)."
    if lab in ("slow", "slowback") or (lab == "rest" and t < 12.4):
        return "Step 2. One very slow stretch.", "Slow stretches show the stiffness of the tissue itself."
    if lab == "fast":
        return f"Step 3. Fast stretch {n_fast} of {total}, at {v:.1f} rad/s.", "The reflex switches on and the muscle resists (red)."
    if lab == "back":
        return f"Fast stretch {n_fast} of {total} done.", "The robot brings the arm back gently."
    if n_fast >= total and t > meta["trace"][-1][3] - 0.5:
        return "Done.", f"{total} fast stretches, {meta['time']:.0f} s in all."
    return "Rest for 10 s.", "The reflex recovers before the next stretch."


class Panels:
    """The right half: three live traces and the onset plot."""

    def __init__(self, T, meta):
        self.T, self.m = T, meta
        rest = T[(T[:, 0] > 0.2) & (T[:, 0] < 0.95), 3]
        self.hold = float(np.median(rest))
        self.fig = plt.figure(figsize=((W - RW) / 100, H / 100), dpi=100)
        gs = self.fig.add_gridspec(5, 1, height_ratios=[0.55, 1, 1, 1, 1.9], hspace=0.55, left=0.15, right=0.96, top=0.98, bottom=0.07)
        self.ax_t = self.fig.add_subplot(gs[0]); self.ax_t.axis("off")
        self.ax = [self.fig.add_subplot(gs[i]) for i in (1, 2, 3)]
        self.ax_o = self.fig.add_subplot(gs[4])
        self.t_end = meta["time"]
        self.ylab = ["elbow angle\n(rad)", "extra robot\ntorque (N m)", "muscle\nsignal (EMG)"]
        self.series = [T[:, 1], T[:, 3] - self.hold, T[:, 4]]
        self.lims = [(-0.05, 1.0), (-1.0, max(8.0, float(np.percentile(self.series[1], 99.9)) + 0.5)), (0, 1.3)]

    def draw(self, t):
        T, m = self.T, self.m
        i = int(np.searchsorted(T[:, 0], t))
        self.ax_t.clear(); self.ax_t.axis("off")
        c1, c2 = caption(m, t)
        self.ax_t.text(0.0, 0.95, c1, fontsize=14, fontweight="bold", color=INK, va="top", transform=self.ax_t.transAxes)
        self.ax_t.text(0.0, 0.25, c2, fontsize=11.5, color=INK, va="top", transform=self.ax_t.transAxes)
        for a, y, lab, lim in zip(self.ax, self.series, self.ylab, self.lims):
            a.clear()
            a.plot(T[:i, 0], y[:i], color=BLUE if a is not self.ax[2] else RED, lw=0.8)
            a.set_xlim(0, self.t_end); a.set_ylim(*lim); a.set_ylabel(lab, fontsize=9)
            a.tick_params(labelsize=8)
            if a is not self.ax[2]:
                a.set_xticklabels([])
        self.ax[0].axhline(m["lam"], color=ORANGE, lw=0.8, ls="--")
        self.ax[0].text(0.3, m["lam"] + 0.03, "reflex threshold (hidden from the robot)", fontsize=7.5, color=ORANGE)
        self.ax[2].set_xlabel("time (s)", fontsize=9)
        a = self.ax_o
        a.clear()
        a.set_xlim(-0.1, 3.5); a.set_ylim(-0.05, 0.9)
        a.set_xlabel("stretch speed (rad/s)", fontsize=9); a.set_ylabel("angle where\nthe reflex starts (rad)", fontsize=9)
        a.tick_params(labelsize=8)
        a.axhline(m["lam"], color=ORANGE, lw=0.8, ls="--")
        done = [tr for tr in m["trace"] if tr[3] <= t]
        pts = [(c[2], c[1]) for c, tr in zip(m["catches"], m["trace"]) if tr[3] <= t and c[1] is not None]
        if pts:
            x, y = np.array(pts).T
            a.plot(x, y, "o", color=BLUE, ms=7, zorder=3, label="where the reflex was seen")
        if done and done[-1][1] is not None:
            a.plot([0.0], [done[-1][1]], "D", color=INK, ms=7, zorder=4, label="robot's estimate of the threshold")
            a.text(0.12, 0.84, f"robot's estimate: {done[-1][1]:.2f} rad   true: {m['lam']:.2f} rad", fontsize=9.5, color=INK)
        else:
            a.text(0.12, 0.84, "each fast stretch adds one point", fontsize=9.5, color=GREY)
        if pts:
            a.legend(loc="lower right", fontsize=8, frameon=False)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]


def overlay(img, rate, act_max):
    """Corner tags on the arm image: the playback rate and the colour key."""
    import matplotlib.figure as mf
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    f = mf.Figure(figsize=(RW / 100, H / 100), dpi=100)
    FigureCanvasAgg(f)
    a = f.add_axes([0, 0, 1, 1]); a.imshow(img); a.axis("off")
    tag = "slow motion ×5" if rate < 1 else "fast forward ×10" if rate > 5 else "×2" if rate > 1 else "real time"
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8)
    a.text(18, 34, "Simulated patient and robot", fontsize=13, fontweight="bold", color=INK, bbox=box)
    a.text(18, 64, "MyoSuite elbow with 6 muscles (MuJoCo)", fontsize=9.5, color=INK, bbox=box)
    a.text(RW - 18, 34, tag, fontsize=11, color="white", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="#444444" if rate != 1 else BLUE, ec="none"))
    x0, y0 = 18, H - 40
    for k in range(60):
        c = (1 - k / 59) * np.array([.80, .45, .45]) + k / 59 * np.array([.85, .05, .05])
        a.add_patch(plt.Rectangle((x0 + 2 * k, y0), 2, 12, color=c))
    a.text(x0, y0 - 8, "muscle activity", fontsize=8.5, color=INK)
    a.text(x0, y0 + 28, "low", fontsize=8, color=INK); a.text(x0 + 120, y0 + 28, "high", fontsize=8, color=INK, ha="right")
    a.text(RW - 18, H - 20, "motor at the elbow  ·  blue cuffs  ·  white EMG electrode", fontsize=8.5, color=INK, ha="right")
    f.canvas.draw()
    return np.asarray(f.canvas.buffer_rgba())[:, :, :3]


def card(lines, sizes):
    f = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    f.patch.set_facecolor("white")
    y = 0.62
    for s, z in zip(lines, sizes):
        f.text(0.5, y, s, ha="center", va="center", fontsize=z, color=INK, fontweight="bold" if z > 20 else "normal")
        y -= 0.07 if z < 20 else 0.1
    f.canvas.draw()
    out = np.asarray(f.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(f)
    return out


def main(src, out, preview=None):
    z = np.load(src)
    T, meta = z["ticks"], json.loads(str(z["meta"]))
    ts, rates = timeline(meta, meta["time"])
    pan = Panels(T, meta)
    paint = Painter(RW, H)
    # activity above the resting floor, scaled to the strongest flexor moment of the session (actuators in model order)
    act = T[:, 6:12]
    flex = [i for i in range(paint.m.nu) if paint.m.actuator(i).name in ("BIClong", "BICshort", "BRA")]
    floor = float(act.min())
    act = (act - floor) / max(float(act[:, flex].max()) - floor, 1e-9)
    act_max = 1.0
    intro = card(["How a robot measures spasticity in under a minute",
                  "A simulated stroke patient with a stiff elbow and an overactive stretch reflex.",
                  "The robot stretches the elbow at a few speeds and watches where the reflex switches on.",
                  "Simulation only, from our ICORR 2027 study."], [26, 15, 15, 12])
    m = meta
    outro = card([f"Estimated threshold {m['lam_hat']:.2f} rad.  True threshold {m['lam']:.2f} rad.",
                  f"{len(m['speeds'])} fast stretches and one slow one, {m['time']:.0f} s in all.",
                  "The usual protocol uses 20 stretches and about 5 minutes.",
                  "This patient was picked as a clear case from development patients, not from the test."], [22, 15, 15, 12])
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
        for t in (0.5, 6.0, 23.4, 24.6, 30.0, 49.3, 51.2):
            k = int(np.searchsorted(T[:, 0], t))
            left = overlay(paint.frame(T[k, 1], np.clip(act[k] / act_max, 0, 1)), rates[min(int(np.searchsorted(ts, t)), len(rates) - 1)], act_max)
            plt.imsave(Path(preview) / f"f_{t:05.1f}.png", np.hstack([left, pan.draw(t)]))
        plt.imsave(Path(preview) / "intro.png", intro)
        plt.imsave(Path(preview) / "outro.png", outro)
        return
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
    w.send(None)
    for _ in range(FPS * 6):
        w.send(np.ascontiguousarray(intro))
    for n, (t, r) in enumerate(zip(ts, rates)):
        k = min(int(np.searchsorted(T[:, 0], t)), len(T) - 1)
        left = overlay(paint.frame(T[k, 1], np.clip(act[k] / act_max, 0, 1)), r, act_max)
        w.send(np.ascontiguousarray(np.hstack([left, pan.draw(t)])))
        if n % 200 == 0:
            print(n, len(ts), flush=True)
    last = np.hstack([left, pan.draw(meta["time"])])
    for _ in range(FPS * 3):
        w.send(np.ascontiguousarray(last))
    for _ in range(FPS * 6):
        w.send(np.ascontiguousarray(outro))
    w.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[3] if len(a) > 3 and a[2] == "--preview" else None)
