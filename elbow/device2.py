"""The same device as device.py, holding any body: the control, the sensors and the cut-out are unchanged."""
from __future__ import annotations

import numpy as np

from .device import Session


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
