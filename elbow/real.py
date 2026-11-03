"""Test 7: a more realistic world.  The assessors are frozen at test 6; only the simulator changes.

Scenarios (each turns on one thing; `real` turns on all of them):
  spindle   the reflex reads a spindle whose firing grows with a power of stretch velocity (exponent 0.6) and with the
            muscle's own activity, the form of Prochazka and Gorassini (1998).  It has no separate speed factor, no knee and
            no saturation of the velocity term: none of these is in the assessor's family.  Sizes are scaled to TSRT slopes.
  emgreal   EMG as recorded: a higher noise floor, 50 Hz mains, electrode pops, a slowly drifting electrode gain, and three
            times the motion artefact (De Luca et al. 2010).
  device    the robot as built: the cuff and soft tissue are a spring and damper between motor and arm, the device axis is
            off the elbow axis, and the torque sensor is noisier, drifts and lags.  The robot's sensors read the motor side.
  thixo     tissue that stiffens at rest and gives with movement (thixotropy, Proske et al. 1993), with a short-range stiffness.
  position  (Hill body only) the 2-joint MyoSuite arm: the shoulder sits at a different angle each session, which moves the
            biceps, a two-joint muscle.  The spindle averages the three flexors.  Truth is at the reference shoulder angle.
  real      all of the above (position on the Hill body only).

Every random draw for these is made after the draws of the old scenarios, so test 6 patients are unchanged.
"""
from __future__ import annotations

import numpy as np

PHYS_DT = 0.001
REAL7 = ("spindle", "emgreal", "device", "thixo", "position", "real")


def parts(scenario):
    if scenario == "real":
        return {"spindle", "emgreal", "device", "thixo", "position"}
    return {scenario} if scenario in REAL7 else set()


def add_realism(rng, p, n, scenario, world):
    """Draw this patient's realism, after every draw of the old scenarios.  Changes p and n in place."""
    on = parts(scenario)
    if not on:
        return
    if "spindle" in on:
        n.form = "spindle"
        n.kv = n.mu                                         # velocity term at 1 rad/s, rad: a TSRT-sized slope
        n.kc = rng.uniform(0.03, 0.12)                      # activity feeds back on the spindle, rad at full reflex
        n.mu = n.kv * 3.2 ** 0.6 / 3.2                      # so the truth's catch at 3.2 rad/s matches the law
    if "emgreal" in on:
        n.emg_real = dict(mains=rng.uniform(0.5, 2.0), pop_rate=rng.uniform(0.05, 0.2), pop_amp=rng.uniform(5.0, 20.0),
                          gain_sd=rng.uniform(0.1, 0.25), gain_tau=rng.uniform(30.0, 120.0), floor=rng.uniform(2.0, 3.0))
        n.emg_art *= 3.0
    if "device" in on:
        p["device"] = dict(k=rng.uniform(40.0, 120.0), zeta=rng.uniform(0.1, 0.6), Im=rng.uniform(0.02, 0.05), bm=0.05,
                           mis=rng.uniform(0.02, 0.08), mis_ph=rng.uniform(-np.pi, np.pi), scale=rng.uniform(0.95, 1.05),
                           noise=0.15, drift_sd=rng.uniform(0.05, 0.2), drift_tau=rng.uniform(60.0, 240.0), lag=1 / (2 * np.pi * 25.0))
    if "thixo" in on:
        p["thixo"] = dict(peak=rng.uniform(0.3, 1.5), w=0.02, share=rng.uniform(0.1, 0.4), T_set=rng.uniform(10.0, 40.0),
                          c_break=rng.uniform(0.1, 0.3), T_anchor=2.0)
    if "position" in on and world == "myo":
        p["position"] = dict(drift_sd=0.02, drift_tau=30.0)


class Thixo:
    """Resistance that builds at rest and breaks with movement.  x is the share of bonds formed; it starts at 1 (rested)."""

    def __init__(self, c, q0=0.0):
        self.c, self.x, self.qa = c, 1.0, q0

    def step(self, q, qd, curve):
        c = self.c
        self.x += PHYS_DT * ((1.0 - self.x) / c["T_set"] - self.x * abs(qd) / c["c_break"])
        self.qa += PHYS_DT * (q - self.qa) / c["T_anchor"]
        return self.x * (c["peak"] * np.tanh((q - self.qa) / c["w"]) + c["share"] * (float(curve(q)) - float(curve(self.qa))))


class EmgReal:
    """Recording problems added to the simulated EMG.  Works on the raw sample before the envelope."""

    def __init__(self, c, rest, rng):
        self.c, self.rest, self.rng = c, rest, rng
        self.gain, self.pop_t, self.t = 1.0, -1.0, 0.0
        self.pop_amp = 0.0

    def raw(self, sample):
        c, dt = self.c, PHYS_DT
        self.t += dt
        s = c["gain_sd"] * np.sqrt(2 * dt / c["gain_tau"])
        self.gain += -dt / c["gain_tau"] * (self.gain - 1.0) + s * self.rng.standard_normal()
        x = max(self.gain, 0.2) * sample + c["mains"] * self.rest * np.sin(2 * np.pi * 50.0 * self.t)
        if self.t - self.pop_t > 0.03 and self.rng.random() < c["pop_rate"] * dt:
            self.pop_t, self.pop_amp = self.t, c["pop_amp"] * self.rest * self.rng.choice([-1.0, 1.0])
        if self.t - self.pop_t < 0.02:
            x += self.pop_amp * np.exp(-(self.t - self.pop_t) / 0.006)
        return abs(x)


class Robot:
    """The device as built, around an arm.  The motor turns a lever; the cuff and soft tissue are a spring and damper; the
    device axis is off the elbow axis, so the arm's angle is f(motor angle).  The session sees the motor encoder and a torque
    sensor between motor and cuff.  Everything else (EMG, truth for scoring) comes from the arm inside."""

    def __init__(self, arm, c, rng):
        self.arm, self.c, self.rng = arm, dict(c), rng
        self.q0 = arm.q
        if hasattr(arm, "m"):                                 # the MyoSuite arm: its inertia about the elbow
            import mujoco
            M = np.zeros((arm.m.nv, arm.m.nv))
            mujoco.mj_fullM(arm.m, M, arm.d.qM)
            inertia = float(M[arm.j, arm.j])
        else:
            inertia = arm.p["I"]
        self.c["b"] = 2.0 * c["zeta"] * np.sqrt(c["k"] * inertia)     # cuff damping as a fraction of critical
        self.qm, self.qdm = self.q0, 0.0
        self.tau_meas, self.drift, self.tau_c = 0.0, 0.0, 0.0

    def __getattr__(self, name):                              # emg, t, max_resist, ... from the arm
        return getattr(self.__dict__["arm"], name)

    def f(self, qm):
        c = self.c
        return qm + c["mis"] * (np.sin(qm + c["mis_ph"]) - np.sin(self.q0 + c["mis_ph"]))

    def fp(self, qm):
        return 1.0 + self.c["mis"] * np.cos(qm + self.c["mis_ph"])

    @property
    def q(self):
        return self.qm

    @property
    def qd(self):
        return self.qdm

    def step(self, u):
        c, dt, a = self.c, PHYS_DT, self.arm
        fp = self.fp(self.qm)
        tau_l = c["k"] * (self.f(self.qm) - a.q) + c["b"] * (fp * self.qdm - a.qd)      # on the arm
        tau_m = tau_l * fp * c["scale"]                                                   # back on the motor
        a.step(tau_l)
        acc = (u - tau_m - c["bm"] * self.qdm) / c["Im"]
        self.qdm += dt * acc
        self.qm += dt * self.qdm
        s = c["drift_sd"] * np.sqrt(2 * dt / c["drift_tau"])
        self.drift += -dt / c["drift_tau"] * self.drift + s * self.rng.standard_normal()
        self.tau_meas += dt / c["lag"] * (tau_m + self.drift - self.tau_meas)

    def torque_reading(self):
        return self.tau_meas + self.rng.normal(0.0, self.c["noise"])
