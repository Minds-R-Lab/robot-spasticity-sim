"""The assessors.  Each takes a Session and the same clinic notes, and returns a report in the same form.

l2a       the law, identified by touch; it picks the next speed by what it would learn, and stops when it knows enough.
rule      a fixed protocol and fixed thresholds, in two forms: torque (Tardieu's catch) and EMG (the published tonic stretch
          reflex threshold method: the EMG onset angle against speed, a line, and its value at speed zero).
"""
from __future__ import annotations

import os

import numpy as np

from .device import DT, V_MAX, Session
from .model import DEL, KNE, LAM, MU, SD_FLOOR, SD_REPORT, SLOPE, WIN, Data, Post, reflex_shape, win

T_SOFT = 7.5               # N m: the software stop on every stretch, for every assessor
T_ABS = 8.5
T_SWEEP = 6.0              # N m: the slow sweep stops here, gravity removed by the notes' estimate
V_SWEEP = 0.15             # rad/s
V_CREEP = 0.06
A_DEC = 0.05
REST = float(os.environ.get("ELBOW_REST", "10.0"))       # s between stretches: the published method rests at least 10 s
SPEEDS = (0.8, 1.6, 2.4, 3.2)
REPS = 3
SPEEDS5 = (0.6, 1.2, 1.8, 2.4, 3.2)                       # the published protocol: at least 20 stretches over the speed range
LADDER = (0.6, 1.0, 1.5, 2.0, 2.6, 3.2)
OPENING = (1.0, 2.0, 3.2)                                 # up the speeds first: is there a reflex at all, and the safe order
OPEN_ON = os.environ.get("ELBOW_OPEN", "1") == "1"
MAX_STRETCH = 12
MIN_STRETCH = int(os.environ.get("ELBOW_MIN_STRETCH", "3"))
P_YES, P_NO = 0.9, 0.1
P_STOP_YES, P_STOP_NO = 0.95, 0.05
SD_LAM = float(os.environ.get("ELBOW_SD_LAM", "0.10"))
P_REPLICATE = 0.5
G_REL = float(os.environ.get("ELBOW_G_REL", "0.75"))      # the EMG speed gain a reflex must show, against this patient's quiet level
I_NOM, B_NOM, F_NOM = 0.09, 0.35, 0.25
# Fair corrections, set on development patients: a constant speed shows mu minus the delay, and an onset threshold reads late.
RULE_MU_OFF = {True: 0.048, False: 0.050}
RULE_LAM_OFF = {True: -0.064, False: -0.197}


def env(name, default):
    return float(os.environ.get("ELBOW_" + name, default))


def soft_stop(h0=0.0):
    """The software stop acts on the rise above the torque that holds the arm at rest, and keeps clear of the cut-out."""
    def f(t, q, qd, tau, emg):
        return abs(tau - h0) > T_SOFT or abs(tau) > T_ABS
    return f


def hold_torque(ses: Session):
    """The torque that holds the arm at the start, and the patient's own resting EMG: the device reads both from its sensors."""
    r = ses.rest(1.0)
    ses.emg_quiet = float(np.median(r[:, 4])) * 2.0 + env("EMG_QUIET_ADD", 0.05)
    return float(np.median(r[:, 3]))


def rest_quiet(ses: Session, use_emg, seconds=REST, tries=5):
    """Rest, then wait up to tries more seconds for a quiet arm: the same check for every assessor.  Quiet means a low
    EMG, or, without EMG, a holding torque near its usual value at this posture."""
    rows = ses.rest(seconds)
    for _ in range(tries):
        last = rows[-int(0.5 / DT):]
        ses.holds.append(float(last[:, 3].mean()))
        usual = float(np.median(ses.holds))
        noisy = last[:, 4].mean() > ses.emg_quiet if use_emg else abs(last[:, 3].mean() - usual) > env("TAU_QUIET", 0.3)
        if not noisy:
            return
        ses.holds.pop()
        rows = ses.rest(1.0)


def sweep_stop(info):
    """The slow sweep's stop: gravity removed by the notes' estimate."""
    g, phi, h0 = info["G_hat"], info.get("phi_hat", 0.0), info["hold"]

    def f(t, q, qd, tau, emg):
        return tau - h0 - g * (np.cos(q + phi) - np.cos(phi)) - F_NOM > T_SWEEP
    return f


def checked_stretch(ses: Session, q_end, v, use_emg, stop):
    """One stretch and the way back.  With EMG, read it at the end of the return, when a reflex must be silent and the arm moves
    too slowly for a motion artefact.  EMG there is the patient's own effort, and the stretch is not used.  rest_quiet has already
    read it before the stretch."""
    ses.n_fast += 1
    e = ses.move(q_end, v, label="fast", stop=stop)
    e["hold"] = ses.rest(0.2)
    back = ses.move(0.0, 0.5, label="back", record=False)
    if not use_emg:
        return e, True
    return e, back["rows"][-int(0.1 / DT):, 4].mean() < ses.emg_quiet


# the rule-based assessors --------------------------------------------------------------------------------------------

def rule(ses: Session, info, use_emg: bool, speeds=SPEEDS, reps=REPS, sweeps=None):
    info = dict(info, hold=hold_torque(ses))
    stop = sweep_stop(info)
    roms, outs, backs = [], [], []
    for _ in range(int(env("RULE_SWEEPS", 3)) if sweeps is None else sweeps):     # several slow sweeps: the median curve, the median range
        e_out = ses.move(2.3, V_SWEEP, label="slow", stop=stop)
        roms.append(float(ses.arm.q))
        ses.rest(0.3)
        e_back = ses.move(0.0, V_SWEEP, label="slowback")
        ses.rest(1.0)
        outs.append(e_out["rows"])
        backs.append(e_back["rows"][::-1])
    rom = float(np.median(roms))
    q_grid = np.linspace(0.05, rom, 60)
    t_out = np.median([np.interp(q_grid, o[:, 1], _sm(o[:, 3], 20)) for o in outs], axis=0)
    t_bk = np.median([np.interp(q_grid, np.maximum.accumulate(b[:, 1]), _sm(b[:, 3], 20)) for b in backs], axis=0)
    mid = 0.5 * (t_out + t_bk)
    passive = mid - info["G_hat"] * np.cos(q_grid + info.get("phi_hat", 0.0))
    s = (q_grid >= 0.7 * rom) & (q_grid <= 0.9 * rom)
    k = float(np.polyfit(q_grid[s], passive[s], 1)[0]) if s.sum() >= 3 else float("nan")
    order = list(speeds) * reps
    ses.rng.shuffle(order)
    catches, runs = [], []
    q_end = 0.9 * rom
    extra = 3                                              # a stretch spoiled by effort is repeated, a few times at most
    while order:
        v = order.pop()
        rest_quiet(ses, use_emg)
        base = ses.rest(0.5)
        e, clean = checked_stretch(ses, q_end, v, use_emg, soft_stop(info["hold"]))
        if clean:
            runs.append((v, e, base))
        elif extra:
            extra -= 1
            order.append(v)
    art = artefact(runs) if use_emg else 0.0               # the same motion artefact term the assessor fits
    for v, e, base in runs:
        c = catch_emg(e, base, ses.emg_quiet, art) if use_emg else catch_torque(e, q_grid, t_out)
        catches.append((v, c[0] if c else None, c[1] if c else None))
    return rule_report(catches, rom, k, ses, RULE_MU_OFF[use_emg], RULE_LAM_OFF[use_emg])


def _sm(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same") if len(x) > n else x


def catch_torque(e, q_grid, t_out):
    """Tardieu's catch: torque above the slow curve, once speed and inertia are taken off, held for 50 ms."""
    r = e["rows"]
    if len(r) < 3:                                         # a stretch stopped at once shows no catch
        return None
    q, qd, tau = r[:, 1], r[:, 2], r[:, 3]
    acc = np.gradient(_sm(qd, 8), DT)
    excess = _sm(tau - np.interp(q, q_grid, t_out) - I_NOM * acc - B_NOM * (qd - V_SWEEP), 8)
    return _onset(excess, q, qd, env("CATCH_T", 0.5))


def abs_acc(r):
    return np.abs(np.gradient(_sm(r[:, 2], 8), DT)) if len(r) > 2 else np.zeros(len(r))


def artefact(runs):
    """EMG per unit of acceleration, from the first 100 ms of each stretch, before any reflex: the median over stretches."""
    x = []
    for v, e, base in runs:
        r = e["rows"]
        n = int(0.1 / DT)
        if len(r) >= n:
            a = abs_acc(r[:n]).mean()
            if a > 2.0:
                x.append((r[:n, 4].mean() - base[:, 4].mean()) / a)
    return max(float(np.median(x)), 0.0) if x else 0.0


def catch_emg(e, base, quiet, art=0.0):
    r = e["rows"]
    if len(r) < 3:
        return None
    thr = base[:, 4].mean() + env("EMG_SD", 3.0) * base[:, 4].std() + env("EMG_MIN", 0.5) * quiet
    return _onset(r[:, 4] - art * abs_acc(r), r[:, 1], r[:, 2], thr)


def _onset(x, q, qd, thr):
    above = (x > thr) & (qd > 0.15)
    run = np.convolve(above, np.ones(10), mode="valid") >= 10
    i = int(np.argmax(run)) if run.any() else -1
    return (float(q[i]), float(qd[i])) if i >= 0 else None    # the angle and the speed at onset, as the published method reads them


def rule_report(catches, rom, k, ses, mu_off, lam_off):
    """A reflex is called if at least half of the stretches at the top two speeds show an onset.  The published method has no yes or no, so this
    is ours.  The line is onset angle against the speed at onset."""
    hit = [(u, c) for v, c, u in catches if c is not None]
    top = sorted({v for v, _, _ in catches})[-2:]
    n_top = sum(v in top for v, _, _ in catches)
    h_top = sum(v in top and c is not None for v, c, _ in catches)
    yes = n_top > 0 and h_top >= 0.5 * n_top
    lam, mu = None, None
    if yes and len(hit) >= 2 and np.ptp([v for v, _ in hit]) > 0.2:
        slope, inter = np.polyfit([v for v, _ in hit], [c for _, c in hit], 1)
        lam, mu = float(inter) + lam_off, float(max(-slope + mu_off, 0.0))
    elif yes:
        lam, mu = float(np.mean([c for _, c in hit])) + lam_off, 0.0
    return dict(reflex="yes" if yes else "no", p_reflex=1.0 if yes else 0.0, lam=lam, mu=mu, q_end=rom, k80=k,
                catches=list(catches))


# the Law-to-Action assessor ------------------------------------------------------------------------------------------

def trapezoid(q_end, v, acc=15.0):
    """The stretch as the device will run it, for planning."""
    q, qd, out_q, out_v = 0.0, 0.0, [], []
    while q < q_end - 1e-4:
        lim = min(v, np.sqrt(max(2 * acc * (q_end - q), 0.0)))
        qd = min(qd + acc * DT, lim)
        q += qd * DT
        out_q.append(q)
        out_v.append(qd)
        if qd < 1e-3:
            break
    return np.array(out_q), np.array(out_v)


def next_speed(post: Post, q_end, v_top, v_tried):
    """The speed whose stretch the candidates disagree about most: the information it should give, in closed form."""
    cand = [v for v in LADDER if v <= v_top]
    hyp, w = post.hyp, post.w_hyp
    if len(hyp) == 0 or w.sum() <= 0:
        return max(cand)
    idx = np.argsort(w)[::-1][:80]
    wh = w[idx] / w[idx].sum()
    h = hyp[idx]
    p1 = float(np.clip(post.p_reflex, 1e-3, 1 - 1e-3))
    sig = post.sig_e if post.emg and post.sig_e else post.sig
    amp = 0.3 if post.emg else 3.0
    best, score = max(cand), -1.0
    for v in cand:
        q, qd = trapezoid(q_end, v)
        if len(q) < 3 * WIN:
            continue
        r = win(reflex_shape(q, qd, LAM[h], MU[h], SLOPE[h], DEL[h], KNE[h])) * amp
        m = r @ wh
        var = (r - m[:, None]) ** 2 @ wh + p1 * (1 - p1) * m ** 2
        mi = 0.5 * np.log1p(var / max(sig, 0.02) ** 2).sum()
        mi *= 0.7 ** v_tried.count(v)                         # a repeat tells less than the formula says: reflex tires
        if mi > score:
            best, score = v, mi
    return best


def governor_sweep(ses: Session, info, data: Data, use_emg, rng):
    """The slow sweep.  The speed falls before the end of range the posterior cannot yet rule out."""
    q_low = {"v": max(info["rom_hat"] - 0.6, 0.3)}
    rows = []
    stop0 = sweep_stop(info)

    def stop(t, q, qd, tau, emg):
        rows.append((t, q, qd, tau, emg))
        if len(rows) >= int(1.0 / DT) and len(rows) % int(1.0 / DT) == 0:
            d = Data()
            d.add(np.array(rows), 0, 0, light=True)
            if len(d.q) >= 15:
                q_low["v"] = Post(d, info, use_emg).tissue(rng)["q_safe"]
        return stop0(t, q, qd, tau, emg)

    def gov(q):
        return max(V_CREEP, np.sqrt(2 * A_DEC * max(q_low["v"] - q, 0.0)))

    e = ses.move(2.3, V_SWEEP, label="slow", stop=stop, governor=gov)
    data.add(e["rows"], 0, 0)
    data.at_limit = e["why"] != "reached"
    ses.rest(0.3)
    e = ses.move(0.0, 0.3, label="slowback")
    data.add(e["rows"], 0, 0)
    if use_emg:
        data.clean_sweep()
    ses.rest(1.0)


def G_MIN(ses):
    return G_REL * ses.emg_quiet


DEC_GAIN = os.environ.get("ELBOW_DEC_GAIN", "1") == "1"    # with EMG, the speed gain alone decides whether there is a reflex
TISS_SWEEP = os.environ.get("ELBOW_TISS_SWEEP", "e")       # tissue from the slow sweep: e with EMG, all always, no never


def l2a(ses: Session, info, use_emg: bool, schedule="adaptive", rng=None, min_stretch=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    data = Data()
    info = dict(info, hold=hold_torque(ses))
    governor_sweep(ses, info, data, use_emg, rng)
    if not data.seg:                                       # the sweep stopped at once, often on a burst of effort: rest and try once more
        ses.rest(3.0)
        info = dict(info, hold=hold_torque(ses))
        governor_sweep(ses, info, data, use_emg, rng)
    if not data.seg:                                       # still nothing to read: say so
        return dict(reflex="cannot tell", p_reflex=0.5, lam=None, lam_sd=None, mu=None, mu_sd=None, delay=None, delay_sd=None, lam_sd_raw=None,
                    onset_reflex="cannot tell", onset_lam=None, onset_mu=None, onset_lam_raw=None, catches=[], speeds=[], trace=[], sigma=None,
                    speed_gain=None, q_end=float(ses.arm.q), q_lo=0.0, q_hi=2.3, q_safe=0.0, k80=None, k_lo=0.0, k_hi=1e9)
    post = Post(data, info, use_emg)
    tiss = post.tissue(rng)
    q_end = 0.9 * data.q_max                                # the same reach as the rule: within what the sweep touched
    tried, v_top, trace, runs = [], 1.0, [], []
    fixed = list(SPEEDS5) * 4
    ses.rng.shuffle(fixed)
    for i in range(1, (len(fixed) if schedule == "fixed" else MAX_STRETCH) + 1):
        if schedule == "fixed":
            v = fixed[i - 1]
        elif schedule == "rand":                            # the ablation: the same law and stop, the speed by chance
            v = float(rng.choice([x for x in LADDER if x <= v_top]))
        elif schedule == "asc":                             # the ablation: a fixed ascending ladder, again and again
            v = LADDER[(i - 1) % len(LADDER)]
        elif OPEN_ON and len(tried) < len(OPENING):
            v = OPENING[len(tried)]
        elif use_emg and (DEC_GAIN or post.p_reflex >= P_YES) and post.speed_gain() < G_MIN(ses):
            v = v_top                                       # speed has not shown a reflex yet: go faster
        else:
            v = next_speed(post, q_end, v_top, tried)
        rest_quiet(ses, use_emg)
        base = ses.rest(0.5)
        e, clean = checked_stretch(ses, q_end, v, use_emg, soft_stop(info["hold"]))
        if not clean:
            continue
        runs.append((v, e, base))
        data.add(e["rows"], i, 1)
        if use_emg:
            data.add_gain(i, np.vstack([e["rows"], e["hold"]]))
        tried.append(v)
        v_top = max(v_top, min(v + 1.2, 3.2))
        post = Post(data, info, use_emg)
        tr = post.lam_mu()
        trace.append((v, tr[0] if tr else None, tr[1] if tr else None, tr[2] if tr else None, post.speed_gain() if use_emg else None, ses.t))
        if schedule == "fixed" or len(tried) < (MIN_STRETCH if min_stretch is None else min_stretch):
            continue
        lm = post.lam_mu()
        slow_only = use_emg and post.speed_gain() < G_MIN(ses)     # no gain with speed: whatever this is, it is not a reflex
        if use_emg and DEC_GAIN:
            yes = not slow_only and lm is not None and lm[1] < SD_LAM
            no = slow_only and max(tried) >= 3.2
        else:
            yes = post.p_reflex >= P_STOP_YES and not slow_only and lm is not None and lm[1] < SD_LAM \
                and post.replicated() >= P_REPLICATE
            no = (post.p_reflex <= P_STOP_NO or slow_only) and max(tried) >= 3.2
        if yes or no:
            break
    if TISS_SWEEP == "all" or (TISS_SWEEP == "e" and use_emg):
        pass                                                # the tissue read from the slow sweep stands: the muscles were quiet
    else:
        tiss = post.tissue(rng)
    lm, dl = post.lam_mu(), post.delay()
    p = post.p_reflex
    if use_emg and post.speed_gain() < G_MIN(ses):
        p = min(p, P_NO)
    elif use_emg and DEC_GAIN:
        p = max(p, P_YES)
    elif p >= P_YES and post.replicated() < P_REPLICATE:    # one stretch alone must not decide it
        p = 0.5 * (P_YES + P_NO)
    # the same stretches, read the rule's way: the onset line.  The hybrid reports this threshold with the model's tissue.
    if use_emg:
        art = artefact(runs)
        cs = [catch_emg(e, b, ses.emg_quiet, art) for v, e, b in runs]
    else:
        s0 = data.seg[0]
        o = np.argsort(s0["q"][s0["qd"] > 0])
        qg, tg = s0["q"][s0["qd"] > 0][o], s0["tau"][s0["qd"] > 0][o]
        cs = [catch_torque(e, qg, tg) for v, e, b in runs]
    catches = [(v, c[0] if c else None, c[1] if c else None) for (v, e, b), c in zip(runs, cs)]
    onset = rule_report(catches, 0.0, 0.0, None, RULE_MU_OFF[use_emg], RULE_LAM_OFF[use_emg])
    onset_raw = rule_report(catches, 0.0, 0.0, None, 0.0, 0.0)
    r = dict(reflex="yes" if p >= P_YES else "no" if p <= P_NO else "cannot tell", p_reflex=p,
             onset_reflex=onset["reflex"], onset_lam=onset["lam"], onset_mu=onset["mu"], onset_lam_raw=onset_raw["lam"], catches=catches,
             lam=lm[0] if lm else None, lam_sd=float(np.sqrt(max(lm[1] ** 2 - SD_FLOOR ** 2, 0.0) + SD_REPORT ** 2)) if lm else None,
             mu=lm[2] if lm else None,
             mu_sd=lm[3] if lm else None, delay=dl[0] if dl else None, delay_sd=dl[1] if dl else None,
             lam_sd_raw=float(np.sqrt(max(lm[1] ** 2 - SD_FLOOR ** 2, 0.0))) if lm else None,
             speeds=tried, trace=trace, sigma=post.sig, speed_gain=post.speed_gain() if use_emg else None, **tiss)
    return r


METHODS = {"rule_t": lambda s, i, r: rule(s, i, False), "rule_e": lambda s, i, r: rule(s, i, True),
           "pub_t": lambda s, i, r: rule(s, i, False, SPEEDS5, 4), "pub_e": lambda s, i, r: rule(s, i, True, SPEEDS5, 4),
           "l2a_t": lambda s, i, r: l2a(s, i, False, rng=r), "l2a_e": lambda s, i, r: l2a(s, i, True, rng=r),
           "l2a_t_fixed": lambda s, i, r: l2a(s, i, False, "fixed", r),
           "l2a_e_fixed": lambda s, i, r: l2a(s, i, True, "fixed", r),
           "short_e": lambda s, i, r: rule(s, i, True, SPEEDS, 1, 1), "short_t": lambda s, i, r: rule(s, i, False, SPEEDS, 1, 1),
           "l2a_e6": lambda s, i, r: l2a(s, i, True, rng=r, min_stretch=6), "l2a_t6": lambda s, i, r: l2a(s, i, False, rng=r, min_stretch=6),
           "l2a_e12": lambda s, i, r: l2a(s, i, True, rng=r, min_stretch=12),
           "l2a_e_rand": lambda s, i, r: l2a(s, i, True, "rand", r), "l2a_t_rand": lambda s, i, r: l2a(s, i, False, "rand", r),
           "l2a_e_asc": lambda s, i, r: l2a(s, i, True, "asc", r), "l2a_t_asc": lambda s, i, r: l2a(s, i, False, "asc", r)}
