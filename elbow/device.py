"""The one-joint device: position control along a speed profile, sensors, and a torque cut-out that no software above it
can override.  Assessors see only what these sensors report."""
from __future__ import annotations

import numpy as np

from .patient import PHYS_DT, Arm, Patient

DT = 0.005                 # control and logging period (200 Hz)
SUB = int(round(DT / PHYS_DT))
U_MAX = 14.0               # N m: the motor's limit
T_STOP = 9.0               # N m: the torque cut-out: the stretch is aborted and the device backs off
V_MAX = 3.5                # rad/s: the fastest stretch the device will run
ACC = 15.0                 # rad/s^2
KP, KD = 150.0, 6.0
Q_NOISE, QD_NOISE, TAU_NOISE = 0.002, 0.02, 0.05


class Session:
    """A patient in the device.  Everything is logged per stretch; arm holds the truth, for scoring only."""

    def __init__(self, patient: Patient, rng: np.random.Generator):
        self.rng = rng
        self.arm = Arm(patient, rng)
        self.log = []                       # one dict of arrays per recorded move
        self.q_ref = 0.0
        self.n_cutouts = 0
        self.n_fast = 0
        self.holds = []
        self._u = 0.0

    @property
    def t(self):
        return self.arm.t

    def _tick(self, q_r, v_r, rows=None):
        arm = self.arm
        q_m = arm.q + self.rng.normal(0, Q_NOISE)
        qd_m = arm.qd + self.rng.normal(0, QD_NOISE)
        u = float(np.clip(KP * (q_r - q_m) + KD * (v_r - qd_m) + self._u, -U_MAX, U_MAX))
        for _ in range(SUB):
            arm.step(u)
        tau_m = u + self.rng.normal(0, TAU_NOISE)
        if rows is not None:
            rows.append((arm.t, q_m, qd_m, tau_m, arm.emg))
        return tau_m, q_m, qd_m

    def rest(self, seconds, gravity_hold=True):
        """Hold where it is for a while: the reflex recovers and the muscle settles."""
        rows = []
        for _ in range(int(seconds / DT)):
            self._tick(self.q_ref, 0.0, rows)
        return np.array(rows)

    def move(self, target, speed, label="", stop=None, governor=None, record=True):
        """Move the reference to target at speed, with ramps.  stop(t, q, qd, tau, emg) -> True ends the move.
        governor(q) -> a lower speed limit, used by the slow sweep.  Returns (log entry, why it ended)."""
        speed = min(abs(speed), V_MAX)
        rows, why = [], "reached"
        d = 1.0 if target >= self.q_ref else -1.0
        v = 0.0
        while True:
            rem = d * (target - self.q_ref)
            if rem <= 1e-4 and abs(v) < 1e-3:
                break
            lim = speed
            if governor is not None:
                lim = min(lim, governor(self.q_ref))
            lim = min(lim, np.sqrt(max(2 * ACC * rem, 0.0)))
            v = min(v + ACC * DT, lim) if v < lim else max(v - ACC * DT, lim)
            self.q_ref += d * v * DT
            tau_m, q_m, qd_m = self._tick(self.q_ref, d * v, rows)
            if abs(tau_m) > T_STOP:
                self.n_cutouts += 1
                why = "cutout"
                break
            if stop is not None and stop(self.arm.t, q_m, qd_m, tau_m, self.arm.emg):
                why = "stopped"
                break
        if why != "reached":                 # bring the reference back to where the arm is, gently
            self.q_ref = self.arm.q
        entry = dict(label=label, speed=speed, why=why, rows=np.array(rows).reshape(-1, 5))
        if record:
            self.log.append(entry)
        return entry

    def stretch(self, q_end, speed, label="fast", stop=None, back_speed=0.5):
        """One stretch and the slow way back.  Returns the stretch's log entry."""
        self.n_fast += 1
        e = self.move(q_end, speed, label=label, stop=stop)
        self.rest(0.2)
        self.move(0.0, back_speed, label="back", record=False)
        return e

    def result_truth(self):
        a = self.arm
        return dict(max_resist=a.max_resist, overstretch=max(a.max_over, 0.0), max_speed=a.max_speed)
