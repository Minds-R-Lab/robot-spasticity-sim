"""The simulated patient: an elbow in a one-joint device.  This is the ground truth.  The assessors never read it, and
it is deliberately richer than the model the Law-to-Action assessor fits (see README.md, "Where the model is wrong").

Angle q (rad): 0 at the start, flexed; it grows as the elbow is extended, which stretches the elbow flexors.

  I q'' = u - b q' - F tanh(q'/eps) - G cos(q + phi) - t_passive(q) - t_reflex - t_guard(q) + t_effort(t)

t_passive   tissue: a linear term and two exponentials that rise near the end of range.  The tissue end of range q_s
            is where t_passive reaches T_END.
t_reflex    the stretch reflex, after a delay, through a first-order activation.  Its threshold falls with stretch speed:
            drive = [q(t-d) + mu q'(t-d) - lam]_+, so the catch angle at speed v is about lam - mu v.  It is stronger at
            speed, it saturates, it tires with use and recovers with rest, and lam drifts slowly.
t_guard     a patient who guards tightens the muscle near the end of range: it looks like stiff tissue to a torque sensor.
t_effort    episodes of voluntary effort, which the patient does not announce.

EMG (a rectified, smoothed envelope of the flexors) sees the reflex, guarding and effort, not the tissue.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

PHYS_DT = 0.001
T_END = 6.0            # N m: passive torque that defines the end of range
EPS_V = 0.02


@dataclass
class Patient:
    I: float
    b: float
    F: float
    G: float
    phi: float
    k0: float
    a1: float
    A1: float
    a2: float
    A2: float
    q_s: float                     # tissue end of range
    reflex: bool
    lam: float
    mu: float
    gain: float
    tau_max: float
    delay: float
    Ta: float
    vel_floor: float
    hab_rate: float
    hab_rec: float
    drift_sd: float
    guard: bool
    q_g: float
    guard_tau: float
    effort_rate: float
    effort_amp: float
    emg_gain: float
    emg_noise: float
    phenotype: str = ""

    def passive(self, q):
        return (self.k0 * q + self.A1 * np.exp(self.a1 * (q - self.q_s)) + self.A2 * np.exp(self.a2 * (q - self.q_s))
                - self.A1 * np.exp(-self.a1 * self.q_s) - self.A2 * np.exp(-self.a2 * self.q_s))

    @property
    def q_ref(self):
        return 0.8 * self.q_s

    @property
    def k_ref(self):
        """Passive stiffness at 80% of the range, the clinical stiffness number (N m/rad)."""
        h = 1e-4
        return float((self.passive(self.q_ref + h) - self.passive(self.q_ref - h)) / (2 * h))


def sample_patient(rng: np.random.Generator, stress: bool = False) -> Patient:
    """A patient from the study's population.  Reflex severity, contracture, guarding and voluntary effort are independent
    overlays.  stress: a harder population fixed in PROTOCOL.md (a slower, more tiring reflex, more effort)."""
    sev = rng.choice(["none", "mild", "moderate", "severe"], p=[0.3, 0.25, 0.25, 0.2])
    contracture = rng.random() < 0.2
    guard = rng.random() < 0.15
    active = rng.random() < (0.5 if stress else 0.25)

    I = rng.uniform(0.06, 0.12)
    b = rng.uniform(0.1, 0.6)
    F = rng.uniform(0.1, 0.4)
    G = rng.uniform(0.5, 2.5)
    phi = rng.uniform(-0.25, 0.25)
    q_s = rng.uniform(0.55, 0.9) if contracture else rng.uniform(1.0, 1.7)
    k0 = rng.uniform(0.2, 1.5)
    a1 = rng.uniform(5.0, 13.0)
    a2 = a1 * rng.uniform(1.5, 2.5)
    share = rng.uniform(0.1, 0.3)
    # the two exponentials sum to what brings the passive torque to T_END at q_s
    rest = max(T_END - k0 * q_s, 0.5)
    A1, A2 = rest * (1 - share), rest * share
    for _ in range(4):                                    # (the offsets at q = 0 are small: a few fixed-point steps)
        off = A1 * np.exp(-a1 * q_s) + A2 * np.exp(-a2 * q_s)
        A1, A2 = (rest + off) * (1 - share), (rest + off) * share

    reflex = sev != "none"
    if reflex:
        lam = rng.uniform(0.15, 0.9 * q_s)
        mu = rng.uniform(0.04, 0.25)
        gain = {"mild": rng.uniform(4, 9), "moderate": rng.uniform(9, 20), "severe": rng.uniform(20, 45)}[sev]
        tau_max = {"mild": rng.uniform(1.0, 2.5), "moderate": rng.uniform(2.5, 5), "severe": rng.uniform(5, 9)}[sev]
    else:
        lam, mu, gain, tau_max = 9.0, 0.0, 0.0, 0.0
    delay = rng.uniform(0.07, 0.12) if stress else rng.uniform(0.03, 0.06)
    Ta = rng.uniform(0.04, 0.12)
    hab_rate = rng.uniform(0.3, 0.8) * (2.0 if stress else 1.0)
    hab_rec = rng.uniform(25.0, 60.0)

    return Patient(I, b, F, G, phi, k0, a1, A1, a2, A2, q_s, reflex, lam, mu, gain, tau_max, delay, Ta,
                   vel_floor=rng.uniform(0.05, 0.2), hab_rate=hab_rate, hab_rec=hab_rec,
                   drift_sd=rng.uniform(0.0, 0.003), guard=guard, q_g=rng.uniform(0.65, 0.85) * q_s,
                   guard_tau=rng.uniform(2.0, 5.0) if guard else 0.0,
                   effort_rate=(1 / rng.uniform(15.0, 40.0)) if active else 0.0, effort_amp=rng.uniform(0.5, 2.5),
                   emg_gain=rng.uniform(0.6, 1.4), emg_noise=rng.uniform(0.01, 0.03),
                   phenotype=f"{sev}{'+contracture' if contracture else ''}{'+guard' if guard else ''}"
                             f"{'+active' if active else ''}")


class Arm:
    """The patient's elbow in motion, stepped at 1 kHz.  Safety counters are the simulator's: no assessor reads them."""

    def __init__(self, p: Patient, rng: np.random.Generator, horizon_s: float = 400.0):
        self.p, self.rng = p, rng
        self.q, self.qd, self.t = 0.0, 0.0, 0.0
        self.a, self.h = 0.0, 1.0
        self.lam = p.lam
        self.emg = 0.0
        n = max(int(round(p.delay / PHYS_DT)), 1)
        self.buf = deque([(0.0, 0.0)] * n, maxlen=n)
        self.efforts = []
        t = 0.0
        while p.effort_rate > 0 and t < horizon_s:
            t += rng.exponential(1.0 / p.effort_rate)
            self.efforts.append((t, rng.uniform(1.0, 3.0), p.effort_amp * rng.choice([-1.0, 1.0])))
        self.max_resist, self.max_over, self.max_speed = 0.0, 0.0, 0.0

    def effort(self):
        for t0, dur, amp in self.efforts:
            if t0 <= self.t <= t0 + dur:
                return amp * np.sin(np.pi * (self.t - t0) / dur) ** 2
        return 0.0

    def guard(self):
        p = self.p
        return p.guard_tau * min(max((self.q - p.q_g) / 0.15, 0.0), 1.0) if p.guard else 0.0

    def step(self, u: float):
        p, dt = self.p, PHYS_DT
        qo, qdo = self.buf[0]
        self.buf.append((self.q, self.qd))
        if p.reflex:
            e = max(qo + p.mu * qdo - self.lam, 0.0)
            vf = p.vel_floor + (1.0 - p.vel_floor) * min(max(qdo / 1.0, 0.0), 1.0)
            target = min(p.gain * self.h * e * vf, p.tau_max)
            self.a += dt * (target - self.a) / p.Ta
            act = self.a / max(p.tau_max, 1e-6)
            self.h += dt * (-p.hab_rate * act * self.h + (1.0 - self.h) / p.hab_rec)
            self.lam += p.drift_sd * np.sqrt(dt) * self.rng.standard_normal()
        guard, eff = self.guard(), self.effort()
        pas = float(p.passive(self.q))
        resist = pas + self.a + guard
        f = (u - p.b * self.qd - p.F * np.tanh(self.qd / EPS_V) - p.G * np.cos(self.q + p.phi) - resist + eff)
        self.qd += dt * f / p.I
        self.q += dt * self.qd
        self.t += dt
        # the EMG envelope: reflex, guarding and effort, with noise; a 20 ms low-pass
        raw = p.emg_gain * (self.a / 3.0 + guard / 3.0 + abs(eff) / 3.0) + 0.02
        raw += p.emg_noise * self.rng.standard_normal() * (1.0 + 2.0 * raw)
        self.emg += dt / 0.02 * (abs(raw) - self.emg)
        self.max_resist = max(self.max_resist, resist)
        self.max_over = max(self.max_over, self.q - p.q_s)
        self.max_speed = max(self.max_speed, abs(self.qd))
