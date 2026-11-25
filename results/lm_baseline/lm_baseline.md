# Numerical-IK (LM) baseline — cold vs warm initial guess

10 repetitions per condition, seeds 42-51. Descriptive only (no inferential tests vs the stochastic cells).

## Trajectory: line

| metric | LM-cold mean ± sd | LM-cold median (IQR) | LM-warm mean ± sd | LM-warm median (IQR) |
|---|---|---|---|---|
| rmse_pos_mm | 100.9 ± 10.7 | 102.6 (10.8) | 93.26 ± 70 | 101.2 (39.6) |
| rmse_ori_rad | 0.3484 ± 0.0973 | 0.3572 (0.0802) | 0.1785 ± 0.135 | 0.1564 (0.0199) |
| roughness_rad | 328.5 ± 39.7 | 323.9 (41.2) | 3.466 ± 2.7 | 2.841 (1.05) |
| jerk_rad | 572.2 ± 71.7 | 565.1 (85.3) | 1.807 ± 5.22 | 0.1564 (0.15) |
| peak_joint_velocity_rad | 6.08 ± 0.379 | 6.214 (0.142) | 0.4012 ± 1.12 | 0.04692 (0.000898) |
| peak_joint_acceleration_rad | 11.23 ± 1.16 | 11.5 (2.16) | 0.374 ± 1.14 | 0.009671 (0.0252) |
| wall_time_s | 0.3944 ± 0.0511 | 0.386 (0.0235) | 0.1828 ± 0.0584 | 0.1952 (0.038) |

Joint profiles (best run per condition): `lm_joint_profiles_line.png`

## Trajectory: corner

| metric | LM-cold mean ± sd | LM-cold median (IQR) | LM-warm mean ± sd | LM-warm median (IQR) |
|---|---|---|---|---|
| rmse_pos_mm | 112.2 ± 14.9 | 110.6 (10.7) | 85.35 ± 66.6 | 85.74 (129) |
| rmse_ori_rad | 0.4203 ± 0.106 | 0.4025 (0.114) | 0.3834 ± 0.576 | 0.1534 (0.187) |
| roughness_rad | 357.1 ± 26.2 | 362.6 (30) | 2.519 ± 1.1 | 2.079 (2.02) |
| jerk_rad | 618.1 ± 45.4 | 617.2 (46.6) | 0.357 ± 0.197 | 0.3283 (0.284) |
| peak_joint_velocity_rad | 6.224 ± 0.128 | 6.283 (3.23e-10) | 0.05883 ± 0.0142 | 0.06275 (0.025) |
| peak_joint_acceleration_rad | 11.48 ± 1 | 11.97 (1.37) | 0.06041 ± 0.0358 | 0.04104 (0.0648) |
| wall_time_s | 0.3773 ± 0.0348 | 0.3659 (0.0491) | 0.252 ± 0.121 | 0.2782 (0.179) |

Joint profiles (best run per condition): `lm_joint_profiles_corner.png`
