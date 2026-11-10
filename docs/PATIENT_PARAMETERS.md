# Simulated patient parameters

Every value below is taken from the code (`elbow/world.py`, `elbow/real.py`, `scripts/run2.py`). Each patient draws its
parameters independently, uniformly in the range given unless stated otherwise. Units: angles in rad, torques in N m,
times in s. The elbow angle q is 0 at the flexed start and grows with extension.

## Neural model (both bodies)

| Parameter | Symbol | Range or rule | Source or basis |
| --- | --- | --- | --- |
| Severity | | none 0.30, mild 0.25, moderate 0.25, severe 0.20 (probabilities) | assumption |
| Threshold | λ | 0.15 to 0.9 q_s (none: no reflex) | assumption; within the range of motion |
| Velocity sensitivity | μ_n | log-normal, median 0.09 s, log-sd 0.5, clipped to 0.02 to 0.3 s | near published TSRT slopes (Piscitelli 2025) |
| Reflex gain | g | mild 4 to 9, moderate 9 to 20, severe 20 to 45 N m/rad | assumption |
| Reflex saturation | τ_max | mild 1 to 3, moderate 3 to 7, severe 7 to 12 N m | below the largest reported values (McPherson 2019) |
| Delay | d | 0.02 to 0.05 s | van der Krogt 2015, Turpin 2016, Rao 2021 |
| Activation time constant | T_a | 0.04 to 0.12 s | assumption |
| Speed factor at zero speed | f_0 | 0.05 to 0.2 | assumption |
| Speed factor knee | κ | 1 rad/s (3 rad/s in Knee 3; none in Flat) | assumption |
| Fast fatigue rate, recovery | r_f, T_f | 0.3 to 0.8 per s of full reflex; 5 to 15 s | set so 30 stretches a minute apart can halve the reflex (Schmit 2000) |
| Slow fatigue rate, recovery | r_s, T_s | 0.2 to 0.8; 150 to 400 s (Early: 1.0 to 2.0; 1200 to 3600 s, floor 0 to 0.2) | as above; Early fitted to Patterson 2022 |
| Threshold wander within a session | | Ornstein-Uhlenbeck, sd 0.02 to 0.06, time constant 2 to 6 s | assumption |
| Threshold shift between days | | sd 0.02 to 0.06 | assumption |
| Angle-dependent activation (guarding) | | in 15% of patients; 2 to 5 N m, starting at 0.65 to 0.85 q_s | assumption |
| Unannounced voluntary effort | | in 25% of patients; bursts every 15 to 40 s on average, 1 to 3 s long, 0.5 to 2.5 N m | assumption |
| Contracture | | in 20% of patients (shortened range, see below) | assumption |

The drive is e = [σ(t-d) + μ_n σ'(t-d) - λ - j(t)]_+, where σ is the joint angle (analytic body) or the length of the long
head of biceps in radians of elbow (Hill body), and j(t) is the wander. The reflex torque a follows
T_a a' = min(g h e f, τ_max) - a, with f = f_0 + (1 - f_0) clip(σ'(t-d)/κ, 0, 1) and fatigue h = h_f h_s.

## EMG

| Parameter | Range | Note |
| --- | --- | --- |
| Gain | 0.6 to 1.4 | |
| Rest floor | 0.01 to 0.03 | |
| Motion artefact | 0.002 to 0.008 per rad/s^2 of acceleration | |
| Signal | Gaussian noise (about 170 Hz band) times (drive + floor), rectified, 20 ms smoothing | drive leads force by a 10 ms filter |

## Analytic body (the assessor's own model)

| Parameter | Range |
| --- | --- |
| Inertia I | 0.06 to 0.12 kg m^2 |
| Damping b | 0.1 to 0.5 N m s/rad |
| Coulomb friction F | 0.1 to 0.4 N m |
| Gravity G, phase φ | 0.5 to 2.7 N m; -0.25 to 0.25 rad |
| End of range q_s | 1.0 to 1.7 rad (contracture: 0.55 to 0.9) |
| Tissue | k_0 q + two exponentials; k_0 0.2 to 1.5, rate a_1 5 to 13, a_2 = 1.5 to 2.5 a_1, share of the second 0.1 to 0.3; scaled so tissue reaches 6 N m at q_s |

## Hill body (MyoSuite `myoelbow_1dof6muscles.xml`)

| Parameter | Range |
| --- | --- |
| Segment mass scale | 0.8 to 1.25 |
| Joint damping, friction | 0.1 to 0.5 N m s/rad; 0.1 to 0.4 N m |
| Flexor passive force scale | 0.6 to 2.0 (doubled with contracture) |
| Passive length scale | 0.7 to 1.0 (times 0.6 with contracture) |
| End of range q_s | where passive torque reaches 6 N m |
| Reflex to muscles | a divided by the flexors' torque per unit activation at the start, applied to biceps long, biceps short and brachialis |

## What the assessors are told

Gravity G with 15% error, phase φ with sd 0.1 rad, and the end of range with sd 0.2 rad, as a clinician's notes would give.

## Realism test (`elbow/real.py`)

| Realism | Parameters |
| --- | --- |
| Spindle | velocity term v^0.6, sized to the patient's μ_n at 1 rad/s; activity term 0.03 to 0.12 rad at full reflex |
| Real EMG | floor x2 to x3; 50 Hz mains 0.5 to 2 floors; pops 0.05 to 0.2 per s of 5 to 20 floors; gain drift sd 0.1 to 0.25 over 30 to 120 s; motion artefact x3 |
| Real robot | cuff spring 40 to 120 N m/rad, damping ratio 0.1 to 0.6; motor inertia 0.02 to 0.05 kg m^2; misalignment amplitude 0.02 to 0.08 rad; torque sensor noise 0.15 N m, drift sd 0.05 to 0.2 N m over 60 to 240 s, 25 Hz lag; gain error 0.95 to 1.05 |
| Thixotropy | peak 0.3 to 1.5 N m; 10 to 40% extra stiffness; rebuilds over 10 to 40 s; breaks over 0.1 to 0.3 rad |
| Shoulder position | two-joint MyoSuite arm; shoulder uniform within ±0.15 rad per session, drift sd 0.02 rad over 30 s |
