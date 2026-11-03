"""The same device as device.py, holding any body: the control, the sensors and the cut-out are unchanged."""
from __future__ import annotations

import numpy as np

from .device import KD, KP, Q_NOISE, QD_NOISE, SUB, U_MAX, Session


class Session2(Session):
    def __init__(self, arm, rng: np.random.Generator):
        self.rng = rng
        self.arm = arm
        self.log = []
        self.q_ref = 0.0
        self.n_cutouts = 0
        self.n_fast = 0
        self.holds = []
        self.emg_quiet = 0.12
        self._u = 0.0

    def _tick(self, q_r, v_r, rows=None):
        """As in device.py.  A test 7 robot (real.Robot) reports its own torque sensor instead of the command plus noise."""
        arm = self.arm
        if not hasattr(arm, "torque_reading"):
            return super()._tick(q_r, v_r, rows)
        q_m = arm.q + self.rng.normal(0, Q_NOISE)
        qd_m = arm.qd + self.rng.normal(0, QD_NOISE)
        u = float(np.clip(KP * (q_r - q_m) + KD * (v_r - qd_m) + self._u, -U_MAX, U_MAX))
        for _ in range(SUB):
            arm.step(u)
        tau_m = arm.torque_reading()
        if rows is not None:
            rows.append((arm.t, q_m, qd_m, tau_m, arm.emg))
        return tau_m, q_m, qd_m
