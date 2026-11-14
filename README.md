# How few stretches does a robot need to assess elbow spasticity?

A simulation study. A robot stretches the elbow of simulated stroke patients and measures how strong their stretch
reflex is. We compare the usual 20-stretch method with shorter protocols and with a model-based method.

No real patients are involved. Every patient here is simulated, so the true answer is known and each method can be
scored exactly.

**Videos:**

- [`video/elbow_robot_session.mp4`](video/elbow_robot_session.mp4) (44 s) replays one simulated session in the base
  world. The arm is the MyoSuite elbow with six muscles, coloured by activity. The plots show what the robot measures,
  and the captions explain each step.
- [`video/elbow_robot_realistic.mp4`](video/elbow_robot_realistic.mp4) replays one TSRT-4 session in the realistic world
  of the seventh test, on a patient with **no** reflex. The robot's lever turns at the motor's angle, and a spring stands
  for the soft cuff between lever and forearm (close-up in the corner). The forearm runs ahead of what the motor encoder
  says, the EMG jumps at the start of every fast stretch, and the robot calls a reflex that is not there. With the cuff
  made rigid and everything else the same, the robot calls this patient correctly.

## The problem

After a stroke, many people develop spasticity. When the elbow is stretched quickly, the flexor muscles "catch" and
resist. The resistance has two sources:

- **the reflex**: the nervous system switches the muscle on when it is stretched past a certain angle;
- **the tissue**: muscles and tendons that have become stiff, like a rubber band that lost its stretch.

They need different treatments, so a good measurement must tell them apart.

The **tonic stretch reflex threshold (TSRT)** method does this. The elbow is stretched at many speeds. For each stretch,
the angle where the muscle's electrical signal (EMG) switches on is noted. A line through these angles, taken to zero
speed, gives the **threshold**: the angle where the reflex starts even when the arm is moved very slowly. The method
asks for at least 20 stretches.

## Approach

### Simulated patients

One simulated nervous system drives two different arms:

- **Analytic arm**: a one-joint model with inertia, damping, friction, gravity and a tissue curve.
- **Hill arm**: the MyoSuite elbow in MuJoCo, with six Hill-type muscles and realistic moment arms and masses. The
  methods do not contain this arm's mechanics, so it is an independent check.

Each patient has a reflex with a threshold, a speed sensitivity, a nerve delay of 20 to 50 ms, saturation and fatigue.
The threshold drifts within a session and between days. EMG is noisy and has a motion artefact. Patients may also have
contracture, tightness that depends on angle, or unannounced voluntary effort. The ranges follow published data where
they exist.

### The robot

A one-joint device under position control at 200 Hz. It measures angle, speed, torque and EMG, and it stops if the
torque goes too high.

### Four protocols

| Protocol | What it does |
| --- | --- |
| **TSRT-20** | A motor-driven version of the standard method: 3 slow sweeps, then 20 stretches at 5 speeds from 0.6 to 3.2 rad/s. |
| **TSRT-4** | The same, with 1 slow sweep and 4 stretches. |
| **Bayesian** | Fits a model of the whole joint (tissue, reflex, inertia, gravity) after each stretch, chooses the next speed to learn the most, and stops when it is sure. Usually 3 to 4 stretches. |
| **Hybrid** | Uses the Bayesian session's stretches, reads the reflex with the TSRT line, and takes tissue stiffness from the Bayesian model. No extra stretches. |

Each protocol runs **with EMG** and **with torque only** (no electrodes).

### Test design

Before each test, the code and a written protocol (patients, methods, endpoints, what counts as failure) were
committed. Each test used fresh random seeds. This repository reports the sixth and seventh tests. Earlier tests found
faults in the design; one of them compared against a rule that could not remove the EMG motion artefact, which was unfair
to the rule, and the sixth test fixed it. The seventh test keeps every method frozen and makes the simulated world harsher
(see [A more realistic world](#a-more-realistic-world-seventh-test)).

The sixth test used 500 patients on each arm, with two sessions each on different days, and 200 patients in each of nine
harder scenarios. Comparisons are paired, with bootstrap 95% intervals. "Equivalent" means the interval stays within
0.03 rad for threshold error and 0.03 for the decision score.

## Results

Base patients, first session. Errors are medians. Threshold error is in radians. Stiffness error is the absolute log
ratio of estimated to true stiffness. Lower is better. Where two numbers are given: analytic arm / Hill arm.

**With EMG**

| Protocol | Stretches | Time (s) | Threshold error | Stiffness error |
| --- | --- | --- | --- | --- |
| TSRT-20 | 20 | 331 / 310 | 0.064 / 0.052 | 0.184 / 0.117 |
| TSRT-4 | 4 | 73 / 68 | 0.066 / 0.064 | 0.194 / 0.137 |
| Bayesian | 3.3 / 3.4 | 61 / 57 | 0.064 / 0.049 | 0.092 / 0.058 |
| Hybrid | 3.3 / 3.4 | 61 / 57 | 0.063 / 0.066 | 0.092 / 0.058 |

**Torque only**

| Protocol | Stretches | Time (s) | Threshold error | Stiffness error |
| --- | --- | --- | --- | --- |
| TSRT-20 | 20 | 328 / 306 | 0.155 / 0.114 | 0.176 / 0.117 |
| TSRT-4 | 4 | 73 / 67 | 0.187 / 0.138 | 0.196 / 0.130 |
| Bayesian | 4.4 / 4.5 | 77 / 73 | 0.063 / 0.051 | 0.096 / 0.146 |
| Hybrid | 4.4 / 4.5 | 77 / 73 | 0.141 / 0.135 | 0.096 / 0.146 |

What this means:

1. **With EMG, 4 stretches were as good as 20.** TSRT-4 was equivalent to TSRT-20 on threshold error and on the yes
   or no decision, on both arms. It took about 70 s instead of more than 300 s.
2. **The hybrid halved the stiffness error** with EMG, at no cost in stretches, and its threshold stayed equivalent to
   TSRT-4. Its stiffness error was lower than TSRT-4's in all nine scenarios.
3. **Without EMG, 4 stretches were not enough.** Equivalence with TSRT-20 failed on both endpoints and both arms.
4. **Without EMG, the model read the reflex much better.** The Bayesian method cut threshold error by 55 to 65%
   against the torque onset rules, because it separates reflex from inertia, damping and tissue by their form.
5. **TSRT-20 was the most repeatable** between days. The shorter protocols had a larger smallest detectable change.

### Where methods failed

- The Bayesian yes or no decision with EMG uses a fixed gate on how much EMG grows with speed. It missed reflexes that
  do not grow with speed (sensitivity 0.42 when the reflex had no speed factor).
- The onset line failed when length and velocity acted separately (threshold error 0.44 to 0.49 rad).
- Without EMG, no method found a reflex that does not grow with speed (sensitivity 0.24 to 0.29).
- With EMG, every method was fooled by tensing a moment after a fast stretch (called reflex in 88 to 100% of patients
  without one), by tightness that fast stretches reach, and by a reflex driven by muscle force (found in about half of
  patients at best).

## A more realistic world (seventh test)

The sixth test's world is kind: our own reflex law, clean EMG, a rigid robot, tissue without memory, and a patient placed
the same way every day. The seventh test keeps all four protocols exactly as they were and changes only the simulated
world, in five ways. Each is tested alone, and `real` turns on all of them.

| Realism | What changes |
| --- | --- |
| `spindle` | The reflex reads a muscle spindle whose firing grows with stretch speed to the power 0.6 and with the muscle's own activity (the form of Prochazka and Gorassini 1998). It lies outside the Bayesian model's family. |
| `emgreal` | EMG as recorded: higher noise floor, 50 Hz mains, electrode pops, drifting electrode gain, three times the motion artefact. |
| `device` | The cuff and soft tissue are a spring and damper between motor and arm. The robot's axis is off the elbow's, so the arm's angle differs from the motor's. The torque sensor is noisier, drifts and lags. The robot only reads its own encoder and sensor. |
| `thixo` | Tissue that stiffens at rest and gives with movement. |
| `position` | Hill arm only: the two-joint MyoSuite arm, with the shoulder at a different angle each day, which changes biceps length. |

500 new patients per arm met the realistic world (two sessions) and the base world (one session), so the two can be
paired. Each single realism had 200 more patients. 30,000 sessions in all, none crashed. The code is in `elbow/real.py`
and the scoring in `scripts/test7_report.py`.

**Results in the realistic world (all realisms together), first session, analytic / Hill arm:**

| Protocol, with EMG | Threshold error, base world | Threshold error, realistic world | Specificity, realistic world |
| --- | --- | --- | --- |
| TSRT-20 | 0.055 / 0.043 | 0.256 / 0.166 | 0.28 / 0.28 |
| TSRT-4 | 0.059 / 0.054 | 0.511 / 0.252 | 0.21 / 0.26 |
| Bayesian | 0.065 / 0.054 | 0.105 / 0.125 | 0.60 / 0.63 |
| Hybrid | 0.065 / 0.069 | 0.442 / 0.196 | 0.24 / 0.20 |

1. **Every protocol did much worse.** With EMG, the onset rules' threshold error grew three- to ninefold and their
   specificity fell from above 0.9 to about 0.25: they called many patients without a reflex spastic.
2. **Most of the sixth test's claims failed.** TSRT-4 was no longer equivalent to TSRT-20 on threshold (worse by 0.08 and
   0.04 rad). The hybrid lost its stiffness advantage. Without EMG, the Bayesian method's threshold was still more
   accurate, but it called a reflex in only 13% and 6% of the patients who had one, so it was not useful.
3. **One realism at a time, the claims mostly held,** except under `device`. A soft, misaligned cuff alone cut the
   specificity of the onset rules with EMG to 0.48 to 0.76 and removed the hybrid's stiffness advantage, because the
   robot measures the cuff spring in series with the arm.
4. **The Bayesian reading with EMG degraded least** (not a pre-specified claim): the smallest threshold error and the
   highest specificity in the realistic world, at a cost in sensitivity (0.74 / 0.75).

5. **An exploratory check after the test** (not pre-specified; `scripts/explore_t7_loo.py`, 150 new patients per arm)
   turned off one realism at a time. Turning off the soft cuff restored most of what was lost: TSRT-4's specificity with
   EMG rose from about 0.25 to 0.87 to 0.95, and the hybrid's stiffness was again better than TSRT-4's. Turning off the
   spindle restored the sensitivity of the torque-only methods.

The lesson is that how the robot measures the arm may matter more than how many stretches it makes.

### Limitations

Everything is simulated, and the nervous system in both arms is ours. The realism ranges of the seventh test are our
reading of the literature, not measured on one device. Shoulder angle stood in for forearm rotation. Some
margins and constants were tuned on development patients with known truth. The simulated TSRT method is far more
repeatable than the real one, because there is no human rater. A clinical test is the next step.

## A note on the TSRT slope

At constant stretch speed, the slope that TSRT reports mixes two things: how the threshold changes with speed, and the
nerve delay. Onset angles alone cannot separate them. A separate measure of delay, such as a tendon tap, is needed.
`scripts/alias_check.py` shows this numerically.

## Layout

| Path | What it holds |
| --- | --- |
| `docs/PATIENT_PARAMETERS.md` | Every simulated patient parameter, its range and its source |
| `docs/SCENARIOS.md` | The baseline scenarios, the scenario figure, the protocol timeline and how the protocols depart from the published method |
| `elbow/world.py` | Simulated patients: nerves, the analytic arm and the MyoSuite arm, and the nine scenarios |
| `elbow/real.py` | The seventh test's realisms: spindle, recorded EMG, the soft robot cuff, thixotropy, shoulder position |
| `elbow/device2.py` | The robot: position control, sensors, torque cut-out |
| `elbow/model.py` | The joint model the Bayesian method fits, and its posterior |
| `elbow/assessors.py` | All protocols: `pub_*` (TSRT-20), `short_*` (TSRT-4), `l2a_*` (Bayesian and hybrid); `_e` with EMG, `_t` torque only |
| `elbow/scene.py` | The 3D render of the arm and robot, for the video |
| `elbow/patient.py`, `elbow/device.py` | The simpler patient and device of the first three tests |
| `scripts/run2.py` | One patient, one protocol, one session |
| `scripts/test6_report.py`, `scripts/test7_report.py` | Scoring of the sixth and seventh tests |
| `scripts/check_t6_repro.py` | Checks that the seventh test's code leaves sixth-test patients unchanged |
| `scripts/explore_t7_loo.py`, `scripts/explore_t7_loo_report.py` | The exploratory check: the realistic world with one realism off |
| `scripts/record_session.py`, `scripts/make_video.py` | Record a session and make the first video |
| `scripts/record_real.py`, `scripts/make_video_real.py` | Record a realistic-world session and make the second video |
| `tests/` | Unit tests |

## Running

Python 3.11 or later with `numpy`, `scipy`, `matplotlib`, `mujoco` and `imageio-ffmpeg`.

The Hill arm needs the MyoSuite model files (Apache-2.0, `myo_sim` commit 93b0ca8) at
`~/myo/myo_sim/myo_sim/models/legacy/elbow/` (`myoelbow_1dof6muscles.xml`, and `myoelbow_2dof6muscles.xml` for the
`position` realism). They are not included here.

```bash
python -m pytest -q tests
python scripts/run2.py --seed 5004 --world myo --method l2a_e --out results/try.jsonl
python scripts/record_session.py 5004
python scripts/make_video.py results/video/session_5004.npz video/elbow_robot_session.mp4
python scripts/record_real.py 5057
python scripts/make_video_real.py results/video/real_5057.npz video/elbow_robot_realistic.mp4
```

The tests ran on MuJoCo 3.3.1. Newer MuJoCo (checked on 3.15) gives the same sessions.

`--world` is `lit` (analytic arm) or `myo` (Hill arm). `--scenario` is `base`, one of the scenarios in
`elbow/world.py`, or one of `spindle`, `emgreal`, `device`, `thixo`, `position`, `real`. Seeds from 5000 up are development patients; the test seeds are separate.
