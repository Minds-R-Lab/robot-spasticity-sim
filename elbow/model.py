"""The law the assessor fits, and its posterior.

  u = G1 cos q + G2 sin q + b q' + F sgn q' + I q'' + k0 q + A1 p1(q) + A2 p2(q) + T r(q, q'; lam, mu, s) + T2 r i/5

Every weight is linear, so each candidate (stiffness rate; lam, mu, s) is one Bayesian regression with a closed-form
evidence.  r is the reflex shape on a grid; i counts the stretches, so T2 lets the reflex tire.  With EMG, torque gets an
EMG column in place of the reflex, and the reflex is identified from the EMG itself.

The reflex is driven by q(t-d) + mu q'(t-d) - lam.  At a constant speed v this is q(t) - (lam - (mu - d) v), so a constant
speed shows only mu - d: the delay d and the speed slope mu are aliased, and only a change of speed separates them.  So d is
on the grid as well, and the posterior holds the whole ridge until a stretch breaks it.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import lfilter
from scipy.special import gammaln

from .device import DT
from .patient import T_END

WIN = 10                                  # windows of 50 ms
DELAYS = np.array([4, 6, 10, 14, 18, 24]) # candidate reflex delays, in control ticks: 20 to 120 ms
TA = 0.08                                 # the assumed activation lag
FLOOR = 0.12                              # the assumed reflex drive at zero speed
Q0 = 1.0
RATES = np.array([3.0, 5.0, 7.0, 9.0, 12.0, 16.0])
KNEES = np.array([1.0, 3.0, 0.0])         # the speed factor is not assumed: it saturates at 1 or at 3 rad/s, or it is absent (0)
_L, _M, _S, _D, _K = np.meshgrid(np.linspace(0.05, 1.6, 32), [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], [3.0, 8.0, 20.0], DELAYS, KNEES,
                                 indexing="ij")
LAM, MU, SLOPE, DEL, KNE = _L.ravel(), _M.ravel(), _S.ravel(), _D.ravel(), _K.ravel()
SIGMA0, A0 = 0.15, 2.0
B0 = SIGMA0 ** 2 * (A0 - 1.0)
BETA = float(os.environ.get("ELBOW_BETA", "0.25"))     # tempers evidence: windows are not independent
P_REFLEX, P_ANGLE, P_NONE = 0.5, 0.25, 0.25             # reflex, angle-only activation (guarding), neither
T_MIN = float(os.environ.get("ELBOW_T_MIN", "0.5"))    # N m: a smaller reflex is not one a clinician would act on
E_MIN = float(os.environ.get("ELBOW_E_MIN", "0.08"))   # the same, on the EMG scale
REACH = 0.25                                           # a candidate must be able to show itself in the data
SD_FLOOR = float(os.environ.get("ELBOW_SD_FLOOR", "0.07"))     # rad: the assumed delay and lag are not exactly right
KNOTS = np.arange(0.1, 1.8, 0.2)                       # an unknown angle-only curve: a reflex must show in speed
KNOT_W = 0.15
BUMP_SD = float(os.environ.get("ELBOW_BUMP_SD", "0.6"))
Q_FLOOR, Q_FLOOR_T, K_FLOOR = 0.03, 0.06, 1.12          # interval floors: the curve's form is not exactly right
SD_REPORT = 0.10                                       # rad: the reported threshold spread, from development error tails
STUDENT_NU = 6.0
TOP_RATES = 3
GROW_MAX = 12.0
R_REL = float(os.environ.get("ELBOW_R_REL", "0.0"))        # the speed-dependent EMG as a share of the angle-only EMG beside it
BURST_ABS, BURST_REL = 0.1, 1.5
ANGLE_E = os.environ.get("ELBOW_ANGLE_E", "0") == "1"      # chosen on development patients: off for EMG, on for torque
DELAY_M = float(os.environ.get("ELBOW_DELAY_M", "0.04"))    # s: a prior on the delay, from physiology, wide on purpose
DELAY_S = float(os.environ.get("ELBOW_DELAY_S", "0.03"))
E_STEP = float(os.environ.get("ELBOW_E_STEP", "0.5"))       # the size of an angle-only EMG step (guarding)


def win(x):
    n = len(x) // WIN
    return x[:n * WIN].reshape((n, WIN) + x.shape[1:]).mean(1)


def reflex_shape(q, qd, lam=LAM, mu=MU, slope=SLOPE, delay=DEL, knee=None):
    """The reflex's shape at the control rate, for every (lam, mu, slope, delay, knee): one column each, peak about 1."""
    if knee is None:
        knee = KNE if len(lam) == len(KNE) else np.ones(len(lam))
    out = np.zeros((len(q), len(lam)), np.float32)
    c = DT / TA
    for d in np.unique(delay):
        qo = np.concatenate([np.full(d, q[0]), q[:-d]])
        qdo = np.concatenate([np.full(d, qd[0]), qd[:-d]])
        for k in np.unique(knee):
            m = (delay == d) & (knee == k)
            e = np.maximum(qo[:, None] + qdo[:, None] * mu[m] - lam[m], 0.0)
            vf = FLOOR + (1.0 - FLOOR) * np.clip(qdo / k, 0.0, 1.0) if k > 0 else np.full(len(qdo), 0.5)
            out[:, m] = lfilter([c], [1.0, c - 1.0], np.minimum(slope[m] * e * vf[:, None], 1.0), axis=0)
    return out


def curve_basis(q, rate, nk):
    """The static curve, zero at q = 0: a line, two exponentials, and small steps for what those do not fit.  A step stays up
    past the last data, so resistance that was felt is not forgotten when the curve is carried on."""
    p = lambda r: np.exp(r * (q - Q0)) - np.exp(-r * Q0)
    bump = lambda c: np.tanh((q - c) / KNOT_W) + np.tanh(c / KNOT_W)
    return np.column_stack([q, p(rate), p(2 * rate)] + [bump(c) for c in KNOTS[:nk]])


class Data:
    """The recorded moves, in 50 ms windows, with each reflex hypothesis's column."""

    def __init__(self):
        self.seg = []
        self.at_limit = False                  # the slow sweep stopped on torque
        self.gains = {}                        # by stretch: how much more EMG than the slow sweep at the same angle

    def add(self, rows, i_stretch, fast, light=False):
        if len(rows) < 3 * WIN:
            return
        t, q, qd, tau, emg = rows.T
        wq, wqd, wtau, wemg = win(q), win(qd), win(tau), win(emg)
        acc = np.gradient(wqd, WIN * DT)
        R = np.zeros((len(wq), 0), np.float32) if light else win(reflex_shape(q, qd))
        keep = np.abs(wqd) > 0.03
        self.seg.append(dict(q=wq[keep], qd=wqd[keep], tau=wtau[keep], emg=wemg[keep], acc=acc[keep], R=R[keep],
                             i=np.full(keep.sum(), float(i_stretch)), fast=np.full(keep.sum(), fast)))
        self.stack()

    def stack(self):
        for k in ("q", "qd", "tau", "emg", "acc", "i", "fast"):
            setattr(self, k, np.concatenate([s[k] for s in self.seg]))
        self.R = np.concatenate([s["R"] for s in self.seg])
        self.w = np.ones(len(self.q))

    @property
    def q_max(self):
        return float(self.q.max())

    def without(self, i):
        """The same data without stretch i, to ask whether one stretch alone decided something."""
        d = Data()
        d.seg = [s for s in self.seg if s["i"][0] != i]
        d.at_limit, d.gains = self.at_limit, self.gains
        d.stack()
        return d

    def add_gain(self, i, rows):
        """Spasticity is resistance that grows with speed.  Guarding depends on angle, so at the same angle a stretch shows
        no more EMG than the slow sweep; a reflex does.  A reflex fires late, so rows run on through the hold at the end."""
        if len(rows) < 2 * WIN:                            # a stretch stopped at once tells nothing about speed
            return
        o = self.seg[0]
        up = o["qd"] > 0
        k = np.argsort(o["q"][up])
        q, emg, qd = win(rows[:, 1]), win(rows[:, 4]), win(rows[:, 2])
        self.gains[i] = (emg - np.interp(q, o["q"][up][k], o["emg"][up][k]), np.abs(np.gradient(qd, WIN * DT)))

    def gain(self, i, art):
        """Stretch i's gain, with the motion artefact taken off.  The windows at the very top of the stretch are not read alone:
        the mean of the two largest neighbours resists a noise spike."""
        e, a = self.gains[i]
        x = e - art * a
        return float(np.convolve(x, np.ones(2) / 2, mode="valid").max()) if len(x) > 1 else 0.0

    def clean_sweep(self):
        """The sweep goes out and comes back.  Tissue, guarding and a tonic reflex are there both ways at the same angle; a
        burst of the patient's own effort is there one way only.  So EMG that one way shows and the other does not is
        taken out of the sweep."""
        seg = [s for s in self.seg if s["i"][0] == 0]
        if len(seg) != 2:
            return
        a, b = seg
        def mine(x, y):
            o = np.argsort(y["q"])
            e = np.interp(x["q"], y["q"][o], y["emg"][o])
            return x["emg"] - e > BURST_ABS + BURST_REL * e
        ka, kb = ~mine(a, b), ~mine(b, a)
        if ka.mean() < 0.3 or kb.mean() < 0.3:                # too much is spoiled to say which way is the clean one
            return
        for s_, k in ((a, ka), (b, kb)):
            for f in ("q", "qd", "tau", "emg", "acc", "R", "i", "fast"):
                s_[f] = s_[f][k]
        self.stack()

    def stretches(self):
        return sorted({int(s["i"][0]) for s in self.seg if s["i"][0] > 0})

    def n_knots(self):
        return int((KNOTS <= self.q_max).sum())

    def design(self, rate, use_emg):
        q, qd = self.q, self.qd
        cols = [np.cos(q), np.sin(q), qd, np.tanh(qd / 0.03), self.acc]
        cols += list(curve_basis(q, rate, self.n_knots()).T)
        if use_emg:
            cols.append(self.emg)
        cols.append(np.ones(len(q)))                       # a free offset: gravity and antagonist tissue at the start
        return np.column_stack(cols)


def prior(info, use_emg, nk):
    g, phi = info["G_hat"], info.get("phi_hat", 0.0)
    m = [g * np.cos(phi), -g * np.sin(phi), 0.35, 0.25, 0.09, 0.8, 1.0, 0.3] + [0.0] * nk
    s = [0.2 * g, 0.2 * g, 0.4, 0.3, 0.05, 1.0, 5.0, 5.0] + [BUMP_SD] * nk
    if use_emg:
        m, s = m + [0.0], s + [15.0]
    m, s = m + [0.0], s + [10.0]
    return np.array(m), np.array(s) / SIGMA0


def _evidence(Ln, c, yy, mLm, ldet0, n):
    """NIG log evidence; Ln and c may be stacks."""
    mn = np.linalg.solve(Ln, c[..., None])[..., 0]
    bn = B0 + 0.5 * np.maximum(yy + mLm - (c * mn).sum(-1), 1e-9)
    an = A0 + 0.5 * n
    ld = np.linalg.slogdet(Ln)[1]
    return (-0.5 * n * np.log(2 * np.pi) + 0.5 * (ldet0 - ld) + A0 * np.log(B0) - an * np.log(bn)
            + gammaln(an) - gammaln(A0)), mn, bn, an


def regress(X, y, w, m0, s0):
    """Evidence and posterior for the columns X."""
    n = len(y)
    L0 = 1.0 / s0 ** 2
    Xw = X * w[:, None]
    A = Xw.T @ X + np.diag(L0)
    b = Xw.T @ y + L0 * m0
    return _evidence(A, b, float((w * y) @ y), float((L0 * m0) @ m0), float(np.log(L0).sum()), n)


class Grid:
    """X with two more columns per candidate, r and r i: every evidence in closed form by a 2-by-2 Schur complement, so a
    grid of thousands costs about one regression."""

    def __init__(self, X, y, w, m0, s0, R, i, mz, sz):
        self.X, self.y, self.w, self.m0, self.s0, self.R, self.i, self.mz, self.sz = X, y, w, m0, s0, R, i, mz, sz
        n, H = len(y), R.shape[1]
        L0, Lz = 1.0 / s0 ** 2, 1.0 / sz ** 2
        Xw = X * w[:, None]
        A = Xw.T @ X + np.diag(L0)
        b = Xw.T @ y + L0 * m0
        R = R.astype(np.float64)
        wR, ii = R * w[:, None], i[:, None]
        Z1, Z2 = Xw.T @ R, (Xw * i[:, None]).T @ R              # X'Z for the two columns
        s11 = (wR * R).sum(0)
        s12 = (wR * R * ii).sum(0)
        s22 = (wR * R * ii * ii).sum(0)
        g1 = wR.T @ y + Lz[0] * mz[0]
        g2 = (wR * ii).T @ y + Lz[1] * mz[1]
        sol = np.linalg.solve(A, np.column_stack([b, Z1, Z2]))
        Ab, A1, A2 = sol[:, 0], sol[:, 1:H + 1], sol[:, H + 1:]
        s11 = s11 + Lz[0] - (Z1 * A1).sum(0)
        s12 = s12 - (Z1 * A2).sum(0)
        s22 = s22 + Lz[1] - (Z2 * A2).sum(0)
        g1 = g1 - Z1.T @ Ab
        g2 = g2 - Z2.T @ Ab
        det = np.maximum(s11 * s22 - s12 ** 2, 1e-12)
        self.t1 = (s22 * g1 - s12 * g2) / det                   # the posterior means of the two weights
        self.t2 = (s11 * g2 - s12 * g1) / det
        yy = float((w * y) @ y)
        mLm = float((L0 * m0) @ m0 + (Lz * mz) @ mz)
        quad = float(b @ Ab) + g1 * self.t1 + g2 * self.t2
        self.bn = B0 + 0.5 * np.maximum(yy + mLm - quad, 1e-9)
        self.an = A0 + 0.5 * n
        ld = float(np.linalg.slogdet(A)[1]) + np.log(det)
        ldet0 = float(np.log(L0).sum() + np.log(Lz).sum())
        self.ev = (-0.5 * n * np.log(2 * np.pi) + 0.5 * (ldet0 - ld) + A0 * np.log(B0) - self.an * np.log(self.bn)
                   + gammaln(self.an) - gammaln(A0))

    def one(self, h):
        """The full posterior of candidate h: mean, bn, an, precision."""
        Z = np.column_stack([self.R[:, h], self.R[:, h] * self.i]).astype(np.float64)
        X = np.column_stack([self.X, Z])
        L0 = np.concatenate([1.0 / self.s0 ** 2, 1.0 / self.sz ** 2])
        Xw = X * self.w[:, None]
        A = Xw.T @ X + np.diag(L0)
        b = Xw.T @ self.y + L0 * np.concatenate([self.m0, self.mz])
        return np.linalg.solve(A, b), float(self.bn[h]), self.an, A


_C, _W = np.meshgrid(np.arange(0.2, 1.65, 0.1), [5.0, 15.0], indexing="ij")
ANG_C, ANG_W = _C.ravel(), _W.ravel()


def angle_cols(q):
    """Activation that depends on angle and not on speed.  Evidence favours the simple story, so this story must be as
    simple to tell as the reflex, or guarding is read as a reflex."""
    a = np.minimum(ANG_W * np.maximum(q[:, None] - ANG_C, 0.0), 1.0)
    return a[:, a.max(0) >= REACH]


def log_mean(x):
    return float(np.logaddexp.reduce(x) - np.log(len(x)))


def delay_prior(hyp):
    """Log prior over the candidates, shared out by delay."""
    lp = -0.5 * ((DEL[hyp] * DT - DELAY_M) / DELAY_S) ** 2
    return lp - np.logaddexp.reduce(lp)


def _reach(data, info):
    r = data.R.max(0) >= REACH
    if "oracle_delay" in info and r.size:                  # a diagnostic only: the true delay, told to the fit
        r &= DEL == DELAYS[np.argmin(np.abs(DELAYS * DT - info["oracle_delay"]))]
    return r


class Post:
    """The joint posterior over (stiffness rate, reflex or none).  mode 'torque' reads the reflex from torque; 'emg' reads
    it from the EMG, and torque carries an EMG column."""

    def __init__(self, data: Data, info, emg: bool):
        self.data, self.info, self.emg = data, info, emg
        self.fit()
        if data.R.size and self.sig is not None:          # one robust pass: bursts (patient effort) lose their weight
            self.reweight()
            self.fit()

    def reweight(self):
        d = self.data
        r = (d.tau - self.pred) / max(self.sig, 0.05)
        d.w = (STUDENT_NU + 1.0) / (STUDENT_NU + r ** 2)

    def fit(self):
        d = self.data
        self.reach = _reach(d, self.info)
        self.hyp = np.where(self.reach)[0]
        nk = d.n_knots()
        self.nk = nk
        H, m0, s0 = len(self.hyp), *prior(self.info, self.emg, nk)
        reflex_in_torque = (not self.emg) and H > 0
        Rr = d.R[:, self.reach] if reflex_in_torque else None
        Ra = angle_cols(d.q)
        G = Ra.shape[1]
        ma, sa, zero = np.array([1.5, 0.0]), np.array([4.0, 0.01]) / SIGMA0, np.zeros(len(d.q))
        mz, sz = np.array([3.0, -0.5]), np.array([6.0, 2.0]) / SIGMA0
        log_h0 = np.zeros(len(RATES))
        log_h1 = np.zeros((len(RATES), max(H, 1))) - np.inf
        log_hg = np.zeros(len(RATES)) - np.inf
        keep = {}
        big = np.zeros((len(RATES), max(H, 1)), bool)
        Rmax = d.R[:, self.reach].max(0) if H else None
        for k, rate in enumerate(RATES):
            X = d.design(rate, self.emg)
            ev0, mn0, bn0, an = regress(X, d.tau, d.w, m0, s0)
            A = (X * d.w[:, None]).T @ X + np.diag(1.0 / s0 ** 2)
            log_h0[k] = ev0
            keep[k] = dict(X=X, h0=(mn0, bn0, an, A))
        for k in np.argsort(log_h0)[-TOP_RATES:]:           # the reflex grid is large: only on the likely stiffness rates
            X, m1, s1 = keep[k]["X"], m0, s0
            if G and not self.emg:
                ga = Grid(X, d.tau, d.w, m0, s0, Ra, zero, ma, sa)
                a = int(np.argmax(ga.ev))
                log_hg[k] = log_mean(ga.ev)
                keep[k]["hg"] = (ga, a)
                X, m1, s1 = np.column_stack([X, Ra[:, a]]), np.r_[m0, ma[0]], np.r_[s0, sa[0]]
            if reflex_in_torque:
                g = Grid(X, d.tau, d.w, m1, s1, Rr, d.i / 5.0, mz, sz)
                log_h1[k] = g.ev
                keep[k]["h1"], keep[k]["X1"] = g, X
                big[k] = g.t1 * Rmax >= T_MIN
        self.keep = keep
        self.log_h0, self.log_h1 = log_h0, log_h1
        # rate weights, and the reflex's structure
        if self.emg:
            lr = BETA * log_h0
            self.w_rate = np.exp(lr - lr.max())
            self.w_rate /= self.w_rate.sum()
            self._emg_channel()
        else:
            lp0, lpg = BETA * log_h0 + np.log(P_NONE), BETA * log_hg + np.log(P_ANGLE)
            lp1 = BETA * log_h1 + np.log(P_REFLEX) + delay_prior(self.hyp) if H else np.full((len(RATES), 1), -np.inf)
            top = max(lp0.max(), lpg.max(), lp1.max())
            p0, pg, p1 = np.exp(lp0 - top), np.exp(lpg - top), np.exp(lp1 - top)
            z = p0.sum() + pg.sum() + p1.sum()
            self.w0, self.wg, self.w1 = p0 / z, pg / z, (p1 / z if H else None)
            self.w_rate = self.w0 + self.wg + (self.w1.sum(1) if H else 0.0)
            self.w_rate = self.w_rate / self.w_rate.sum()
            self.p_reflex = float((self.w1 * big).sum()) if H else P_REFLEX
            self.w_hyp = (self.w1 * big).sum(0) if H else np.zeros(0)
        self._map()

    def _emg_channel(self):
        """The reflex from EMG, on slow and fast stretches together.  EMG = e0 + an angle-only curve + T r(q, q').  A tonic
        reflex is in the slow sweep too, so the slow sweep is data, not a baseline: speed separates the reflex from the
        angle-only part (guarding), because r grows with speed and the curve does not."""
        d = self.data
        self.p_reflex, self.w_hyp, self.sig_e = P_REFLEX, np.zeros(len(self.hyp)), None
        if (d.fast > 0).sum() < 5:
            return
        reach = _reach(d, self.info)
        H = int(reach.sum())
        self.hyp = np.where(reach)[0]
        if H == 0:
            return
        q, y, nk = d.q, d.emg, d.n_knots()
        X = np.column_stack([np.ones(len(q)), np.abs(d.acc)] + [np.tanh((q - c) / KNOT_W) + np.tanh(c / KNOT_W) for c in KNOTS[:nk]])
        m0, s0 = np.array([0.03, 0.0] + [0.0] * nk), np.array([0.2, 0.02] + [E_STEP] * nk) / SIGMA0
        Rr, lpd = d.R[:, reach], delay_prior(self.hyp)
        mz, sz = np.array([1.0, -0.2]), np.array([3.0, 1.0]) / SIGMA0
        Ra, zero = angle_cols(q), np.zeros(len(q))
        ma, sa = np.array([0.5, 0.0]), np.array([2.0, 0.01]) / SIGMA0
        w = np.ones(len(q))
        ang = 0.0
        for it in range(2):                                # one robust pass, as for torque: bursts lose their weight
            ev0 = regress(X, y, w, m0, s0)[0]
            X1, m1, s1, lpg = X, m0, s0, -np.inf
            if Ra.shape[1] and ANGLE_E:
                ga = Grid(X, y, w, m0, s0, Ra, zero, ma, sa)
                a = int(np.argmax(ga.ev))
                lpg = BETA * log_mean(ga.ev) + np.log(P_ANGLE)
                X1, m1, s1 = np.column_stack([X, Ra[:, a]]), np.r_[m0, ma[0]], np.r_[s0, sa[0]]
                ang = max(float(ga.t1[a]), 0.0)
            g = Grid(X1, y, w, m1, s1, Rr, d.i / 5.0, mz, sz)
            lp0, lp1 = BETA * ev0 + np.log(P_NONE), BETA * g.ev + np.log(P_REFLEX) + lpd
            h = int(np.argmax(lp1))
            sig = float(np.sqrt(g.bn[h] / (g.an - 1.0)))
            if it == 0:
                mn = g.one(h)[0]
                self.art = max(float(mn[1]), 0.0)
                r = (y - np.column_stack([X1, Rr[:, h], Rr[:, h] * d.i / 5.0]) @ mn) / max(sig, 0.02)
                w = (STUDENT_NU + 1.0) / (STUDENT_NU + r ** 2)
        top = max(lp0, lpg, lp1.max())
        p1 = np.exp(lp1 - top)
        z = np.exp(lp0 - top) + np.exp(lpg - top) + p1.sum()
        p1 = p1 * (g.t1 * Rr.max(0) >= max(E_MIN, R_REL * ang))
        self.p_reflex = float(p1.sum() / z)
        self.w_hyp = p1 / max(p1.sum(), 1e-300)
        self.sig_e = sig

    def _map(self):
        d = self.data
        if self.emg:
            k = int(np.argmax(self.w_rate))
            mn, bn, an, _ = self.keep[k]["h0"]
            X = self.keep[k]["X"]
            self.pred = X @ mn
            self.sig = float(np.sqrt(bn / (an - 1.0)))
            return
        if self.w1 is not None and self.w1.max() > max(self.w0.max(), self.wg.max()):
            k, h = np.unravel_index(int(np.argmax(self.w1)), self.w1.shape)
            mn, bn, an, _ = self._h1(k, h)
            R = d.R[:, self.hyp[h]].astype(np.float64)
            self.pred = np.column_stack([self.keep[k]["X1"], R, R * d.i / 5.0]) @ mn
        elif self.wg.max() > self.w0.max():
            k = int(np.argmax(self.wg))
            ga, a = self.keep[k]["hg"]
            mn, bn, an, _ = ga.one(a)
            self.pred = np.column_stack([self.keep[k]["X"], ga.R[:, a], np.zeros(len(d.q))]) @ mn
        else:
            k = int(np.argmax(self.w0))
            mn, bn, an, _ = self.keep[k]["h0"]
            self.pred = self.keep[k]["X"] @ mn
        self.sig = float(np.sqrt(bn / (an - 1.0)))

    def _h1(self, k, h):
        return self.keep[k]["h1"].one(h)

    def speed_gain(self):
        """The second largest stretch gain (Data.gain): one burst of effort cannot supply it."""
        g = sorted(self.data.gain(k, getattr(self, "art", 0.0)) for k in self.data.gains)
        return g[-2] if len(g) >= 2 else 0.0

    def replicated(self):
        """The smallest reflex probability left after taking away any one fast stretch.  The slow sweep stays: the contrast
        between slow and fast is what defines a reflex."""
        d, low = self.data, self.p_reflex
        for i in d.stretches():
            low = min(low, Post(d.without(i), self.info, self.emg).p_reflex)
        return low

    # what the assessor reports ---------------------------------------------------------------------------------------

    def lam_mu(self):
        """Posterior mean and spread of the threshold and speed slope, given that there is a reflex in reach."""
        if len(self.hyp) == 0 or self.w_hyp.sum() <= 0:
            return None
        w = self.w_hyp / self.w_hyp.sum()
        lam, mu = LAM[self.hyp], MU[self.hyp]
        ml, mm = float(w @ lam), float(w @ mu)
        return ml, float(np.sqrt(w @ (lam - ml) ** 2 + SD_FLOOR ** 2)), mm, float(np.sqrt(w @ (mu - mm) ** 2 + 0.03 ** 2))

    def delay(self):
        """The reflex delay, in seconds, given a reflex in reach."""
        if len(self.hyp) == 0 or self.w_hyp.sum() <= 0:
            return None
        w = self.w_hyp / self.w_hyp.sum()
        d = DEL[self.hyp] * DT
        m = float(w @ d)
        return m, float(np.sqrt(w @ (d - m) ** 2))

    def passive_samples(self, rng, n=300):
        """Draws of the tissue curve: (angles, curves), with torque 0 at q = 0."""
        qs = np.linspace(0.0, 2.3, 231)
        ps = self.w_rate / self.w_rate.sum()
        out = []
        for k, c in zip(*np.unique(rng.choice(len(RATES), n, p=ps), return_counts=True)):
            rate = RATES[k]
            mn, bn, an, A = self._best(k)
            V = np.linalg.inv(A)
            cov = (bn / (an - 1.0)) * V / BETA                 # the same tempering as the evidence
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(mn)))
            z = rng.standard_t(2 * an, size=(c, len(mn)))
            th = mn + z @ L.T
            out.append(th[:, 5:8 + self.nk] @ curve_basis(qs, rate, self.nk).T)
        C = np.concatenate(out)
        # past the last angle touched no fitted form is trusted: the slope goes on growing at the rate it was growing at the end of
        # the data, which may be not at all
        i = int(np.searchsorted(qs, self.data.q_max))
        if 20 <= i < len(qs):
            s2 = np.maximum((C[:, i] - C[:, i - 10]) / (qs[i] - qs[i - 10]), 1e-3)
            s1 = np.maximum((C[:, i - 10] - C[:, i - 20]) / (qs[i - 10] - qs[i - 20]), 1e-3)
            g = np.clip(np.log(s2 / s1) / (qs[i] - qs[i - 10]), 0.0, GROW_MAX)[:, None]
            x = qs[i:] - qs[i]
            C[:, i:] = C[:, i:i + 1] + s2[:, None] * np.where(g > 1e-6, (np.exp(g * x) - 1.0) / np.maximum(g, 1e-6), x)
        return qs, C

    def _best(self, k):
        if self.emg:
            return self.keep[k]["h0"]
        w1 = self.w1[k].max() if self.w1 is not None and "h1" in self.keep[k] else 0.0
        if w1 > max(self.w0[k], self.wg[k]):
            return self._h1(k, int(np.argmax(self.w1[k])))
        if self.wg[k] > self.w0[k] and "hg" in self.keep[k]:
            ga, a = self.keep[k]["hg"]
            return ga.one(a)
        return self.keep[k]["h0"]

    def tissue(self, rng, torque=T_END):
        """The end of range and the stiffness at 80% of it, with 95% intervals and a low quantile for safety."""
        qs, C = self.passive_samples(rng)
        up = np.maximum.accumulate(C, axis=1)
        hit = (up >= torque).argmax(1)
        hit = np.where((up >= torque).any(1), hit, len(qs) - 1)
        ends = qs[hit]
        step = qs[1] - qs[0]
        i7, i9 = (0.7 * ends / step).astype(int), np.clip((0.9 * ends / step).astype(int), 1, len(qs) - 1)   # slope over 70 to 90% of range
        ks = (C[np.arange(len(C)), i9] - C[np.arange(len(C)), i7]) / np.maximum((i9 - i7) * step, step)
        q_end = float(np.median(ends))
        fl = Q_FLOOR if self.emg else Q_FLOOR_T
        if not self.emg and self.data.at_limit:              # torque alone: the limit was met here, whatever caused it
            q_end = min(q_end, self.data.q_max + Q_FLOOR)
        return dict(q_end=q_end, q_lo=float(np.quantile(ends, 0.025)) - fl,
                    q_hi=float(np.quantile(ends, 0.975)) + fl, q_safe=float(np.quantile(ends, 0.005)) - fl,
                    k80=float(np.median(ks)), k_lo=float(np.quantile(ks, 0.025)) / K_FLOOR,
                    k_hi=float(np.quantile(ks, 0.975)) * K_FLOOR)
