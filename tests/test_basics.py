"""Checks that the simulator and the fit do what they claim."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elbow import assessors as AS                 # noqa: E402
from elbow.device import Session                  # noqa: E402
from elbow.model import DEL, LAM, MU, SLOPE, Data, Post, reflex_shape     # noqa: E402
from elbow.patient import T_END, Patient, sample_patient             # noqa: E402


def quiet(**kw):
    a = dict(I=0.09, b=0.3, F=0.2, G=1.5, phi=0.1, k0=0.8, a1=8.0, A1=4.0, a2=16.0, A2=1.0, q_s=1.3, reflex=False, lam=9.0,
             mu=0.0, gain=0.0, tau_max=0.0, delay=0.05, Ta=0.08, vel_floor=0.1, hab_rate=0.5, hab_rec=40.0, drift_sd=0.0,
             guard=False, q_g=1.0, guard_tau=0.0, effort_rate=0.0, effort_amp=1.0, emg_gain=1.0, emg_noise=0.01)
    a.update(kw)
    return Patient(**a)


def test_end_of_range_is_at_t_end():
    for s in range(50):
        p = sample_patient(np.random.default_rng(s))
        assert abs(p.passive(p.q_s) - T_END) < 0.6, (s, p.passive(p.q_s))
        assert abs(p.passive(0.0)) < 1e-9
        assert p.k_ref > 0


def test_reflex_threshold_falls_with_speed():
    p = quiet(reflex=True, lam=0.8, mu=0.15, gain=30.0, tau_max=5.0)
    on = []
    for v in (1.0, 3.0):
        ses = Session(p, np.random.default_rng(0))
        e = ses.move(1.1, v)
        r = e["rows"]
        a = np.array([0.0])
        ses2 = Session(quiet(), np.random.default_rng(0))
        e2 = ses2.move(1.1, v)
        n = min(len(r), len(e2["rows"]))
        ex = np.convolve(r[:n, 3] - e2["rows"][:n, 3], np.ones(8) / 8, mode="same")
        on.append(r[np.argmax(ex > 0.8), 1])
    assert on[1] < on[0] - 0.1, on


def test_emg_sees_the_reflex_not_the_tissue():
    ses = Session(quiet(), np.random.default_rng(0))
    e = ses.move(1.2, 0.5)
    assert e["rows"][:, 4].max() < 0.1
    ses = Session(quiet(reflex=True, lam=0.5, mu=0.1, gain=30.0, tau_max=5.0), np.random.default_rng(0))
    e = ses.move(1.2, 2.0)
    assert e["rows"][:, 4].max() > 0.5


def test_fit_recovers_a_passive_arm():
    p = quiet()
    ses = Session(p, np.random.default_rng(1))
    d = Data()
    d.add(ses.move(1.25, 0.15, label="slow")["rows"], 0, 0)
    ses.rest(0.3)
    d.add(ses.move(0.0, 0.3)["rows"], 0, 0)
    t = Post(d, dict(G_hat=1.5, rom_hat=1.3), False).tissue(np.random.default_rng(0))
    assert abs(t["q_end"] - p.q_s) < 0.05, t
    k = (p.passive(0.9 * p.q_s) - p.passive(0.7 * p.q_s)) / (0.2 * p.q_s)     # the stiffness is the slope over 70 to 90% of range
    assert abs(np.log(t["k80"] / k)) < 0.25, (t, k)


def test_shape_has_one_column_per_candidate():
    q, qd = AS.trapezoid(1.0, 2.0)
    r = reflex_shape(q, qd)
    assert r.shape == (len(q), len(LAM)) and len(LAM) == len(MU) == len(SLOPE) == len(DEL)
    assert r.max() <= 1.0 + 1e-9
