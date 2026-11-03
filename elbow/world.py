"""Patients, set from published ranges (see the README).

Test 7 adds a more realistic world (spindle, emgreal, device, thixo, position, real): see real.py.

Round 6, after a referee's report: EMG now follows the neural drive, not the force-producing activation, so it leads force as real EMG does.  New scenarios
change the reflex form or attack the assessor's assumptions:
  knee3     the speed factor saturates at 3 rad/s, not 1.
  flat      no speed factor: the reflex depends on speed only through its threshold.
  sum       the SpasticMyoElbow form: separate length and velocity terms, each with its own threshold.
  force     feedback from muscle force and its rate, as in Falisse et al. 2018.
  greach    guarding that starts early enough for fast stretches to enter it, and grows with speed.
  react     a tensing a reaction time after a fast stretch begins, with no angle gate, growing with speed.

Two bodies carry the same nervous system:

  LitArm   the analytic elbow of patient.py, with its tissue law.
  MyoArm   the MyoSuite elbow: six Hill-type muscles, moment arms that change with angle, real segment masses.  The assessors share
           nothing with it, and the reflex acts on muscle length and velocity, not on joint angle.

What is new against patient.py, and why:
  delay            20 to 50 ms, the evidence for the short-latency reflex at the elbow.  Long latency is a named worst case.
  speed slope      centred near the published median plus the delay, because published values are not delay corrected.
  reflex torque    up to 12 N m in severe cases.
  fatigue          two time constants, so stretches a minute apart can halve the reflex over 30 stretches, as measured.
  threshold        it wanders from moment to moment, and from day to day: real thresholds are not constants.
  EMG              Gaussian-modulated noise, a rest floor and a motion artefact, not a clean envelope.

Scenarios: base, long (long-latency worst case), gspeed (guarding that grows with speed, the case the speed gain test assumes away),
noisy (three times the EMG noise and artefact, twice the threshold wander), early (the reflex falls fast over the first stretches to a
plateau and rest does not bring it back, set to the curve of 18 people in Patterson et al. 2022, unlike the milder fatigue of base).
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .real import add_realism

PHYS_DT = 0.001
T_END = 6.0
EPS_V = 0.02
MYO_XML = os.path.expanduser("~/myo/myo_sim/myo_sim/models/legacy/elbow/myoelbow_1dof6muscles.xml")
MYO2_XML = os.path.expanduser("~/myo/myo_sim/myo_sim/models/legacy/elbow/myoelbow_2dof6muscles.xml")   # test 7: + shoulder
EARLY_R = (float(os.environ.get("ELBOW_ER0", "1.0")), float(os.environ.get("ELBOW_ER1", "2.0")))       # per second of full activation
EARLY_FLOOR = (float(os.environ.get("ELBOW_EF0", "0.0")), float(os.environ.get("ELBOW_EF1", "0.2")))
Q0_MYO = 1.8                                  # the stretch starts flexed; assessor angle q = Q0_MYO - elbow flexion


@dataclass
class Neural:
    """The nervous system of one patient."""
    reflex: bool
    lam: float
    mu: float
    gain: float
    tau_max: float
    delay: float
    Ta: float
    vel_floor: float
    r_fast: float
    T_fast: float
    r_slow: float
    T_slow: float
    jit_sd: float
    jit_tau: float
    day_sd: float
    guard: bool
    gq: float                       # guarding starts at this fraction of the range
    guard_tau: float
    guard_speed: float              # guarding grows by this factor per rad/s of stretch speed
    effort_rate: float
    effort_amp: float
    emg_gain: float
    emg_rest: float
    emg_art: float
    phenotype: str = ""
    h_floor: float = 0.0            # the plateau the slow part of fatigue falls to
    form: str = "base"              # the reflex law: base, knee3, flat, sum, force
    v_t: float = 1.5                # sum: the velocity threshold
    f_t: float = 2.0                # force: the force threshold, N m
    react: float = 0.0              # react: the peak of the tensing at 3.2 rad/s, N m
    rt: float = 0.2                 # react: reaction time, s
    kv: float = 0.0                 # spindle (test 7): velocity term at 1 rad/s, rad
    kc: float = 0.0                 # spindle (test 7): the muscle's own activity on the spindle, rad at full reflex
    emg_real: dict | None = None    # emgreal (test 7): recording problems, see real.py


def sample_neural(rng, q_s, scenario="base"):
    sev = rng.choice(["none", "mild", "moderate", "severe"], p=[0.3, 0.25, 0.25, 0.2])
    guard = rng.random() < 0.15
    active = rng.random() < 0.25
    reflex = sev != "none"
    delay = rng.uniform(0.07, 0.12) if scenario == "long" else rng.uniform(0.02, 0.05)
    if reflex:
        lam = rng.uniform(0.15, 0.9 * q_s)
        mu = float(np.clip(rng.lognormal(np.log(0.09), 0.5), 0.02, 0.3))
        gain = {"mild": rng.uniform(4, 9), "moderate": rng.uniform(9, 20), "severe": rng.uniform(20, 45)}[sev]
        tau_max = {"mild": rng.uniform(1.0, 3.0), "moderate": rng.uniform(3.0, 7.0), "severe": rng.uniform(7.0, 12.0)}[sev]
    else:
        lam, mu, gain, tau_max = 9.0, 0.0, 0.0, 0.0
    x_jit = 2.0 if scenario == "noisy" else 1.0
    x_emg = 3.0 if scenario == "noisy" else 1.0
    early = scenario in ("early",)
    return Neural(reflex, lam, mu, gain, tau_max, delay, rng.uniform(0.04, 0.12), rng.uniform(0.05, 0.2),
                  rng.uniform(0.3, 0.8), rng.uniform(5.0, 15.0),
                  rng.uniform(EARLY_R[0], EARLY_R[1]) if early else rng.uniform(0.2, 0.8),
                  rng.uniform(1200.0, 3600.0) if early else rng.uniform(150.0, 400.0),
                  x_jit * rng.uniform(0.02, 0.06), rng.uniform(2.0, 6.0), rng.uniform(0.02, 0.06),
                  guard, rng.uniform(0.3, 0.55) if scenario == "greach" else rng.uniform(0.65, 0.85), rng.uniform(2.0, 5.0) if guard else 0.0,
                  rng.uniform(0.5, 2.0) if scenario in ("gspeed", "greach") else 0.0,
                  (1 / rng.uniform(15.0, 40.0)) if active else 0.0, rng.uniform(0.5, 2.5),
                  rng.uniform(0.6, 1.4), x_emg * rng.uniform(0.01, 0.03), x_emg * rng.uniform(0.002, 0.008),
                  f"{sev}{'+guard' if guard else ''}{'+active' if active else ''}",
                  h_floor=rng.uniform(EARLY_FLOOR[0], EARLY_FLOOR[1]) if early else 0.0,
                  form=scenario if scenario in ("knee3", "flat", "sum", "force") else "base",
                  v_t=rng.uniform(1.0, 2.0), f_t=rng.uniform(1.0, 3.0),
                  react=rng.uniform(1.0, 3.0) if scenario == "react" else 0.0, rt=rng.uniform(0.15, 0.3))


class Nerves:
    """Reflex state shared by both bodies: delayed sensing, fatigue, threshold wander, effort, EMG."""

    def __init__(self, n: Neural, rng, day_shift=0.0, horizon_s=1200.0, lam=None):
        self.n, self.rng = n, rng
        self.t, self.a, self.hf, self.hs, self.jit = 0.0, 0.0, 1.0, 1.0, 0.0
        self.lam = (n.lam if lam is None else lam) + day_shift
        self.env, self.noise, self.count, self.drive = 0.0, 0.0, 0, 0.0
        self.t_go, self.was_slow, self.peak_v = -9.0, True, 0.0
        self.force_old = 0.0
        k = max(int(round(n.delay / PHYS_DT)), 1)
        self.buf = deque([(0.0, 0.0)] * k, maxlen=k)
        self.fbuf = deque([0.0] * k, maxlen=k)
        self.efforts = []
        self.rest = n.emg_rest
        self.er = None
        if n.emg_real:                                   # test 7: EMG as recorded
            from .real import EmgReal
            self.rest = n.emg_rest * n.emg_real["floor"]
            self.er = EmgReal(n.emg_real, self.rest, rng)
        t = 0.0
        while n.effort_rate > 0 and t < horizon_s:
            t += rng.exponential(1.0 / n.effort_rate)
            self.efforts.append((t, rng.uniform(1.0, 3.0), n.effort_amp * rng.choice([-1.0, 1.0])))

    def effort(self):
        for t0, dur, amp in self.efforts:
            if t0 <= self.t <= t0 + dur:
                return amp * np.sin(np.pi * (self.t - t0) / dur) ** 2
        return 0.0

    def step(self, q, qd, qs_guard, acc, s=None, sd=None, force=0.0):
        """One millisecond.  The spindle reads s and sd: the joint by default, or a muscle.  force is the passive torque, for a
        force-feedback law.  Returns reflex activity in N m, guarding in N m, effort in N m."""
        n, dt = self.n, PHYS_DT
        qo, qdo = self.buf[0]
        fo = self.fbuf[0] if n.form == "force" else 0.0
        self.buf.append((q if s is None else s, qd if sd is None else sd))
        if n.form == "force":
            self.fbuf.append(force + self.a)
        self.count += 1
        if self.count % 10 == 0:                         # the threshold wanders
            s = 0.01 / n.jit_tau
            self.jit += -s * self.jit + n.jit_sd * np.sqrt(2 * s) * self.rng.standard_normal()
        target = 0.0
        if n.reflex:
            h = self.hf * self.hs
            if n.form == "sum":                          # separate length and velocity terms
                target = h * (n.gain * 0.5 * max(qo - (self.lam + self.jit), 0.0) + n.gain * 0.3 * max(qdo - n.v_t, 0.0))
            elif n.form == "force":                      # muscle force and its rate
                rate = (fo - self.force_old) / dt
                self.force_old = fo
                target = h * (n.gain * 0.15 * max(fo - n.f_t, 0.0) + n.gain * 0.02 * max(rate, 0.0))
            elif n.form == "spindle":                    # test 7: power-law velocity and the muscle's own activity
                vel = np.sign(qdo) * abs(qdo) ** 0.6
                e = max(qo + n.kv * vel + n.kc * self.a / max(n.tau_max, 1e-6) - (self.lam + self.jit), 0.0)
                target = n.gain * h * e
            else:
                e = max(qo + n.mu * qdo - (self.lam + self.jit), 0.0)
                if n.form == "flat":
                    vf = 0.5
                else:
                    knee = 3.0 if n.form == "knee3" else 1.0
                    vf = n.vel_floor + (1.0 - n.vel_floor) * min(max(qdo / knee, 0.0), 1.0)
                target = n.gain * h * e * vf
            target = min(target, n.tau_max)
            self.a += dt * (target - self.a) / n.Ta
            act = self.a / max(n.tau_max, 1e-6)
            self.hf += dt * (-n.r_fast * act * self.hf + (1.0 - self.hf) / n.T_fast)
            self.hs += dt * (-n.r_slow * act * (self.hs - n.h_floor) + (1.0 - self.hs) / n.T_slow)
        g = 0.0
        if n.guard:
            g = n.guard_tau * min(max((q - n.gq * qs_guard) / 0.15, 0.0), 1.0) * (1.0 + n.guard_speed * max(qd, 0.0))
        eff = self.effort()
        if n.react > 0:                                  # a tensing, a reaction time after a fast stretch begins
            if self.was_slow and qd > 0.5:
                self.t_go, self.was_slow = self.t, False
            elif qd < 0.1 and self.t - self.t_go > 1.5:
                self.was_slow = True
            self.peak_v = max(self.peak_v, qd) if not self.was_slow else 0.0
            x = (self.t - self.t_go - n.rt) / 0.6
            if 0.0 <= x <= 1.0:
                g += n.react * min(self.peak_v / 3.2, 1.0) * np.sin(np.pi * x) ** 2
        self.drive += dt / 0.01 * (target - self.drive)  # EMG leads force: it follows the neural drive, not the activation
        amp = n.emg_gain * (self.drive + g + abs(eff)) / 3.0
        if self.count % 3 == 0:                          # a Gaussian-modulated signal of about 170 Hz band
            self.noise = self.rng.standard_normal()
        if self.er is None:
            raw = abs((amp + n.emg_rest) * self.noise)
        else:
            raw = self.er.raw((amp + self.rest) * self.noise)
        self.env += dt / 0.02 * (raw + n.emg_art * abs(acc) - self.env)
        self.t += dt
        return self.a, g, eff


class LitArm:
    """The analytic elbow, with the published ranges."""

    def __init__(self, pp, n: Neural, rng, day_shift=0.0):
        self.p, self.nv = pp, Nerves(n, rng, day_shift)
        self.q, self.qd, self.acc, self.t = 0.0, 0.0, 0.0, 0.0
        self.max_resist, self.max_over, self.max_speed = 0.0, 0.0, 0.0
        self.thx = None
        if pp.get("thixo"):                              # test 7: tissue that stiffens at rest
            from .real import Thixo
            self.thx = Thixo(pp["thixo"], self.q)

    @property
    def emg(self):
        return self.nv.env

    def passive(self, q):
        p = self.p
        return (p["k0"] * q + p["A1"] * np.exp(p["a1"] * (q - p["q_s"])) + p["A2"] * np.exp(p["a2"] * (q - p["q_s"]))
                - p["A1"] * np.exp(-p["a1"] * p["q_s"]) - p["A2"] * np.exp(-p["a2"] * p["q_s"]))

    def step(self, u):
        p = self.p
        pas = float(self.passive(self.q))
        a, g, eff = self.nv.step(self.q, self.qd, p["q_s"], self.acc, force=pas)
        resist = pas + a + g
        if self.thx is not None:
            resist += self.thx.step(self.q, self.qd, self.passive)
        f = u - p["b"] * self.qd - p["F"] * np.tanh(self.qd / EPS_V) - p["G"] * np.cos(self.q + p["phi"]) - resist + eff
        self.acc = f / p["I"]
        self.qd += PHYS_DT * self.acc
        self.q += PHYS_DT * self.qd
        self.t += PHYS_DT
        self.max_resist = max(self.max_resist, resist)
        self.max_over = max(self.max_over, self.q - p["q_s"])
        self.max_speed = max(self.max_speed, abs(self.qd))


def sample_lit(rng, scenario="base"):
    contracture = rng.random() < 0.2
    q_s = rng.uniform(0.55, 0.9) if contracture else rng.uniform(1.0, 1.7)
    k0, a1 = rng.uniform(0.2, 1.5), rng.uniform(5.0, 13.0)
    a2, share = a1 * rng.uniform(1.5, 2.5), rng.uniform(0.1, 0.3)
    rest = max(T_END - k0 * q_s, 0.5)
    A1, A2 = rest * (1 - share), rest * share
    for _ in range(4):
        off = A1 * np.exp(-a1 * q_s) + A2 * np.exp(-a2 * q_s)
        A1, A2 = (rest + off) * (1 - share), (rest + off) * share
    p = dict(I=rng.uniform(0.06, 0.12), b=rng.uniform(0.1, 0.5), F=rng.uniform(0.1, 0.4), G=rng.uniform(0.5, 2.7),
             phi=rng.uniform(-0.25, 0.25), k0=k0, a1=a1, A1=A1, a2=a2, A2=A2, q_s=q_s)
    n = sample_neural(rng, q_s, scenario)
    if contracture:
        n.phenotype += "+contracture"
    add_realism(rng, p, n, scenario, "lit")
    return p, n


class MyoArm:
    """The MyoSuite elbow.  Flexors take the reflex through their own lengths and velocities.  One process holds one patient."""
    FLEX = ("BIClong", "BICshort", "BRA")
    EXT = ("TRIlong", "TRIlat", "TRImed")

    def __init__(self, pp, n: Neural, rng, day_shift=0.0):
        import mujoco
        self.mj = mujoco
        self.p = pp
        self.m = make_model(pp)
        self.d = mujoco.MjData(self.m)
        self.pos = pp.get("position")                    # test 7: the 2-joint arm, the shoulder held at this session's angle
        self.j = 1 if self.pos else 0                    # the elbow's joint
        j = self.j
        self.d.qpos[j] = Q0_MYO
        mujoco.mj_forward(self.m, self.d)
        self.flex = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, x) for x in self.FLEX]
        self.ext = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, x) for x in self.EXT]
        self.spin = self.flex[0]                         # the spindle sits in the long head of biceps
        self.l0 = float(self.d.actuator_length[self.spin])
        self.l0s = self.d.actuator_length[self.flex].copy()
        self.d.qpos[j] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.r = (float(self.d.actuator_length[self.spin]) - self.l0) / Q0_MYO      # metres of muscle per radian, on average
        self.rs = (self.d.actuator_length[self.flex] - self.l0s) / Q0_MYO
        lam_s = 9.0
        if n.reflex:                                     # the threshold, as a length of this muscle
            self.d.qpos[j] = Q0_MYO - min(n.lam, Q0_MYO)
            mujoco.mj_forward(self.m, self.d)
            lam_s = self.spindle()
        self.d.qpos[j] = Q0_MYO
        self.q_end = pp["q_s"]                           # the end of range that safety is measured against
        if self.pos:                                     # truth is at the reference shoulder angle (0); the session's differs
            self.sh0 = self.sh = float(self.pos["shoulder"])
            self.q_end = end_of_range(*myo_curve(pp, self.sh0))
            self.sh_rng = np.random.default_rng(self.pos["seed"])
            self.d.qpos[0] = self.sh
        mujoco.mj_forward(self.m, self.d)
        self.thx = None
        if pp.get("thixo"):                              # test 7: tissue that stiffens at rest
            from .real import Thixo
            self.thx = Thixo(pp["thixo"], self.q)
        self.nv = Nerves(n, rng, day_shift, lam=lam_s)
        # torque per unit flexor activation at the start: turns the nervous system's N m into activation
        if self.j == 0:
            self.t_per_a = max(abs(sum(self.d.actuator_moment.ravel()[a] * self.m.actuator_gainprm[a, 2] for a in self.flex)), 1.0)
        else:                                            # the 2-joint arm: each actuator's moment about the elbow only
            mom = np.zeros((self.m.nu, self.m.nv))
            mujoco.mju_sparse2dense(mom, self.d.actuator_moment, self.d.moment_rownnz, self.d.moment_rowadr, self.d.moment_colind)
            self.t_per_a = max(abs(sum(mom[a, self.j] * self.m.actuator_gainprm[a, 2] for a in self.flex)), 1.0)
        self.t, self.acc, self.qd_prev = 0.0, 0.0, 0.0
        self.max_resist, self.max_over, self.max_speed = 0.0, 0.0, 0.0

    @property
    def q(self):
        return Q0_MYO - float(self.d.qpos[self.j])

    @property
    def qd(self):
        return -float(self.d.qvel[self.j])

    @property
    def emg(self):
        return self.nv.env

    def spindle(self):
        """The spindle's stretch, in radians of elbow: the long head of biceps, or with the 2-joint arm all three flexors."""
        if not self.pos:
            return (float(self.d.actuator_length[self.spin]) - self.l0) / self.r
        return float(np.mean((self.d.actuator_length[self.flex] - self.l0s) / self.rs))

    def spindle_rate(self):
        if not self.pos:
            return float(self.d.actuator_velocity[self.spin]) / self.r
        return float(np.mean(self.d.actuator_velocity[self.flex] / self.rs))

    def step(self, u):
        if self.pos:                                     # the shoulder drifts a little in its rest
            p = self.pos
            s = p["drift_sd"] * np.sqrt(2 * PHYS_DT / p["drift_tau"])
            self.sh += -PHYS_DT / p["drift_tau"] * (self.sh - self.sh0) + s * self.sh_rng.standard_normal()
            self.d.qpos[0], self.d.qvel[0] = self.sh, 0.0
        a, g, eff = self.nv.step(self.q, self.qd, self.p["q_s"], self.acc,
                                 self.spindle(), self.spindle_rate(), force=float(self.p["curve"](self.q)))
        tot = (a + g + max(eff, 0.0)) / self.t_per_a
        c = self.d.ctrl
        c[:] = 0.0
        for k in self.flex:
            c[k] = tot
        if eff < 0:
            for k in self.ext:
                c[k] = -eff / self.t_per_a
        self.d.qfrc_applied[self.j] = -u
        if self.thx is not None:                         # resistance opposes extension, so it pushes flexion up
            self.d.qfrc_applied[self.j] += self.thx.step(self.q, self.qd, self.p["curve"])
        self.mj.mj_step(self.m, self.d)
        qd = self.qd
        self.acc = (qd - self.qd_prev) / PHYS_DT
        self.qd_prev = qd
        self.t += PHYS_DT
        self.max_resist = max(self.max_resist, float(self.p["curve"](self.q)) + a + g)
        self.max_over = max(self.max_over, self.q - self.q_end)
        self.max_speed = max(self.max_speed, abs(qd))


def make_model(pp, two=None):
    """two: the 2-joint arm (test 7 position).  At the reference shoulder angle it is the same arm, so truth uses the 1-joint one."""
    import mujoco
    two = bool(pp.get("position")) if two is None else two
    m = mujoco.MjModel.from_xml_path(MYO2_XML if two else MYO_XML)
    j = 1 if two else 0
    if two:                                          # the shoulder is held: a huge inertia stops it moving within a step
        m.dof_armature[0] = 1e4
    m.opt.timestep = PHYS_DT
    m.body_mass[:] *= pp["mass"]
    m.body_inertia[:] *= pp["mass"]
    m.dof_damping[j] = pp["b"]
    m.dof_frictionloss[j] = pp["F"]
    for name in MyoArm.FLEX:
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for prm in (m.actuator_gainprm, m.actuator_biasprm):
            prm[a, 7] = prm[a, 7] * pp["fp"]            # passive force of the flexors
            prm[a, 5] = 1.0 + (prm[a, 5] - 1.0) * pp["lmax"]   # how soon passive force rises
    return m


def myo_curve(pp, shoulder=None):
    """The tissue truth on this model: the torque to hold each angle with every muscle quiet, gravity aside.  With a shoulder
    angle (test 7 position), the same on the 2-joint arm at that angle: used only for the session's own end of range."""
    import mujoco
    two = shoulder is not None
    m = make_model(pp, two=two)
    d = mujoco.MjData(m)
    j = 1 if two else 0
    qs = np.linspace(0.0, Q0_MYO, 201)
    tau = np.zeros(len(qs))
    for i, q in enumerate(qs):
        if two:
            d.qpos[0] = shoulder
        d.qpos[j] = Q0_MYO - q
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        d.act[:] = 0.0
        mujoco.mj_forward(m, d)
        tau[i] = float(d.qfrc_actuator[j]) + float(d.qfrc_passive[j])
    return qs, tau - tau[0]


def end_of_range(qs, tau):
    up = np.maximum.accumulate(tau)
    return float(qs[int(np.argmax(up >= T_END)) if (up >= T_END).any() else len(qs) - 1])


def sample_myo(rng, scenario="base"):
    contracture = rng.random() < 0.2
    pp = dict(mass=rng.uniform(0.8, 1.25), b=rng.uniform(0.1, 0.5), F=rng.uniform(0.1, 0.4),
              fp=rng.uniform(0.6, 2.0) * (2.0 if contracture else 1.0),
              lmax=rng.uniform(0.7, 1.0) * (0.6 if contracture else 1.0))
    qs, tau = myo_curve(pp)
    up = np.maximum.accumulate(tau)
    i = int(np.argmax(up >= T_END)) if (up >= T_END).any() else len(qs) - 1
    pp["q_s"] = float(qs[i])
    pp["curve"] = lambda q, qs=qs, tau=tau: np.interp(q, qs, tau)
    pp["k80"] = float((np.interp(0.9 * pp["q_s"], qs, tau) - np.interp(0.7 * pp["q_s"], qs, tau)) / (0.2 * pp["q_s"]))
    n = sample_neural(rng, pp["q_s"], scenario)
    if contracture:
        n.phenotype += "+contracture"
    add_realism(rng, pp, n, scenario, "myo")
    return pp, n


def myo_gravity(pp):
    """The gravity torque as G cos(q + phi), in assessor angle: what a table of limb masses would give, before its error."""
    import mujoco
    m = make_model(pp, two=False)
    d = mujoco.MjData(m)
    qs = np.linspace(0.0, 1.6, 9)
    g = []
    for q in qs:
        d.qpos[0] = Q0_MYO - q
        d.qvel[0] = 0.0
        mujoco.mj_forward(m, d)
        g.append(-float(d.qfrc_bias[0]))
    a, b = np.linalg.lstsq(np.column_stack([np.cos(qs), np.sin(qs)]), np.array(g), rcond=None)[0]
    return float(np.hypot(a, b)), float(np.arctan2(-b, a))


def k_ref_lit(p):
    """The stiffness every assessor reports: the slope of the tissue curve over 70 to 90% of the range."""
    pas = lambda q: (p["k0"] * q + p["A1"] * np.exp(p["a1"] * (q - p["q_s"])) + p["A2"] * np.exp(p["a2"] * (q - p["q_s"])))
    return float((pas(0.9 * p["q_s"]) - pas(0.7 * p["q_s"])) / (0.2 * p["q_s"]))
