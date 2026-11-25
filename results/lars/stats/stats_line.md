# Planned comparisons — trajectory 'line' (N=30 runs/cell)

## Metric family: accuracy

### Rank-based planned comparisons (primary — heavy-tailed family)

| question | metric | A | B | median A (IQR) | median B (IQR) | U | p raw | p adj (FDR) | Cliff's delta |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-injection-static | 61.04 (13.8) | 0.5981 (31.5) | 691.0 | 3.77e-04 | 1.03e-03 * | 0.536 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-persistence-static | 61.04 (13.8) | 7.798 (62.2) | 668.0 | 1.30e-03 | 3.00e-03 * | 0.484 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-injection-static | pso-persistence-static | 0.5981 (31.5) | 7.798 (62.2) | 318.0 | 5.19e-02 | 6.71e-02 | -0.293 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-cold-static | ga-cold-static | 61.04 (13.8) | 48.19 (17.6) | 703.0 | 1.89e-04 | 5.67e-04 * | 0.562 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-injection-static | ga-injection-static | 0.5981 (31.5) | 1.849 (19.5) | 352.0 | 1.49e-01 | 1.66e-01 | -0.218 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-injection-static | 36.73 (14.1) | 0.1375 (23) | 684.0 | 5.56e-04 | 1.39e-03 * | 0.520 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-persistence-static | 36.73 (14.1) | 3.848 (51.7) | 661.0 | 1.86e-03 | 3.98e-03 * | 0.469 |
| Q1-memory-effect-pso | mean_pos_mm | pso-injection-static | pso-persistence-static | 0.1375 (23) | 3.848 (51.7) | 313.0 | 4.36e-02 | 6.23e-02 | -0.304 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-cold-static | ga-cold-static | 36.73 (14.1) | 21.7 (13.5) | 768.0 | 2.68e-06 | 1.00e-05 * | 0.707 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-injection-static | ga-injection-static | 0.1375 (23) | 1.11 (5.83) | 344.0 | 1.19e-01 | 1.43e-01 | -0.236 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-injection-static | 165.4 (37.5) | 3.134 (69.4) | 754.0 | 7.22e-06 | 2.41e-05 * | 0.676 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-persistence-static | 165.4 (37.5) | 26.23 (112) | 788.0 | 6.05e-07 | 3.02e-06 * | 0.751 |
| Q1-memory-effect-pso | max_pos_mm | pso-injection-static | pso-persistence-static | 3.134 (69.4) | 26.23 (112) | 316.0 | 4.84e-02 | 6.60e-02 | -0.298 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-cold-static | ga-cold-static | 165.4 (37.5) | 171.5 (47.5) | 420.0 | 6.63e-01 | 6.84e-01 | -0.067 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-injection-static | ga-injection-static | 3.134 (69.4) | 8.286 (101) | 347.0 | 1.30e-01 | 1.50e-01 | -0.229 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-injection-static | 0.04012 (0.0261) | 4.286e-06 (0.0124) | 803.0 | 1.87e-07 | 1.12e-06 * | 0.784 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-persistence-static | 0.04012 (0.0261) | 0.001421 (0.00378) | 876.0 | 3.16e-10 | 4.74e-09 * | 0.947 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-injection-static | pso-persistence-static | 4.286e-06 (0.0124) | 0.001421 (0.00378) | 310.0 | 3.92e-02 | 5.88e-02 | -0.311 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-cold-static | ga-cold-static | 0.04012 (0.0261) | 0.03188 (0.0623) | 478.0 | 6.84e-01 | 6.84e-01 | 0.062 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-injection-static | ga-injection-static | 4.286e-06 (0.0124) | 0.002316 (0.00444) | 273.0 | 9.07e-03 | 1.60e-02 * | -0.393 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-injection-static | 0.01186 (0.00844) | 2.391e-06 (0.00404) | 782.0 | 9.53e-07 | 4.09e-06 * | 0.738 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-persistence-static | 0.01186 (0.00844) | 0.0005096 (0.00112) | 840.0 | 8.48e-09 | 6.36e-08 * | 0.867 |
| Q1-memory-effect-pso | mean_ori_rad | pso-injection-static | pso-persistence-static | 2.391e-06 (0.00404) | 0.0005096 (0.00112) | 298.0 | 2.51e-02 | 4.18e-02 * | -0.338 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-cold-static | ga-cold-static | 0.01186 (0.00844) | 0.0176 (0.0175) | 300.0 | 2.71e-02 | 4.28e-02 * | -0.333 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-injection-static | ga-injection-static | 2.391e-06 (0.00404) | 0.001525 (0.00225) | 255.0 | 4.03e-03 | 8.07e-03 * | -0.433 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-injection-static | 0.1754 (0.126) | 1.81e-05 (0.0514) | 840.0 | 8.48e-09 | 6.36e-08 * | 0.867 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-persistence-static | 0.1754 (0.126) | 0.006296 (0.0122) | 893.0 | 6.07e-11 | 1.82e-09 * | 0.984 |
| Q1-memory-effect-pso | max_ori_rad | pso-injection-static | pso-persistence-static | 1.81e-05 (0.0514) | 0.006296 (0.0122) | 319.0 | 5.37e-02 | 6.71e-02 | -0.291 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-cold-static | ga-cold-static | 0.1754 (0.126) | 0.1202 (0.347) | 508.0 | 3.95e-01 | 4.24e-01 | 0.129 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-injection-static | ga-injection-static | 1.81e-05 (0.0514) | 0.008604 (0.0183) | 271.0 | 8.31e-03 | 1.56e-02 * | -0.398 |

Kruskal-Wallis omnibus across the PSO memory triple (cold / injection / persistence, static):

- rmse_pos_mm: H = 17.446, p = 1.63e-04
- mean_pos_mm: H = 16.661, p = 2.41e-04
- max_pos_mm: H = 31.532, p = 1.42e-07
- rmse_ori_rad: H = 45.504, p = 1.32e-10
- mean_ori_rad: H = 39.667, p = 2.43e-09
- max_ori_rad: H = 51.901, p = 5.37e-12

### Welch t (secondary check)

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-injection-static | 61.66 (12.8) | 31.74 (49.1) | 3.227 | 2.82e-03 | 7.06e-03 * | 0.833 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-persistence-static | 61.66 (12.8) | 33.27 (44.6) | 3.350 | 2.00e-03 | 5.45e-03 * | 0.865 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-injection-static | pso-persistence-static | 31.74 (49.1) | 33.27 (44.6) | -0.126 | 9.00e-01 | 9.47e-01 | -0.033 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-cold-static | ga-cold-static | 61.66 (12.8) | 47.56 (13.3) | 4.182 | 9.90e-05 | 3.30e-04 * | 1.080 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-injection-static | ga-injection-static | 31.74 (49.1) | 25.04 (42.4) | 0.566 | 5.74e-01 | 7.49e-01 | 0.146 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-injection-static | 35.73 (10.9) | 27.62 (45.6) | 0.948 | 3.50e-01 | 5.53e-01 | 0.245 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-persistence-static | 35.73 (10.9) | 27.91 (42.2) | 0.983 | 3.33e-01 | 5.53e-01 | 0.254 |
| Q1-memory-effect-pso | mean_pos_mm | pso-injection-static | pso-persistence-static | 27.62 (45.6) | 27.91 (42.2) | -0.026 | 9.80e-01 | 9.80e-01 | -0.007 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-cold-static | ga-cold-static | 35.73 (10.9) | 21.77 (8.18) | 5.612 | 7.14e-07 | 7.14e-06 * | 1.449 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-injection-static | ga-injection-static | 27.62 (45.6) | 21.05 (39.8) | 0.595 | 5.54e-01 | 7.49e-01 | 0.154 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-injection-static | 165.2 (27.1) | 53.43 (72.9) | 7.876 | 2.05e-09 | 3.88e-08 * | 2.034 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-persistence-static | 165.2 (27.1) | 62.56 (67.6) | 7.725 | 2.59e-09 | 3.88e-08 * | 1.995 |
| Q1-memory-effect-pso | max_pos_mm | pso-injection-static | pso-persistence-static | 53.43 (72.9) | 62.56 (67.6) | -0.503 | 6.17e-01 | 7.71e-01 | -0.130 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-cold-static | ga-cold-static | 165.2 (27.1) | 164.3 (37.2) | 0.115 | 9.09e-01 | 9.47e-01 | 0.030 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-injection-static | ga-injection-static | 53.43 (72.9) | 51.45 (71.4) | 0.106 | 9.16e-01 | 9.47e-01 | 0.027 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-injection-static | 0.0546 (0.0509) | 0.0105 (0.0197) | 4.423 | 8.08e-05 | 3.30e-04 * | 1.142 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-persistence-static | 0.0546 (0.0509) | 0.004465 (0.00856) | 5.316 | 8.98e-06 | 6.74e-05 * | 1.373 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-injection-static | pso-persistence-static | 0.0105 (0.0197) | 0.004465 (0.00856) | 1.538 | 1.32e-01 | 2.83e-01 | 0.397 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-cold-static | ga-cold-static | 0.0546 (0.0509) | 0.05616 (0.0496) | -0.120 | 9.05e-01 | 9.47e-01 | -0.031 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-injection-static | ga-injection-static | 0.0105 (0.0197) | 0.006635 (0.0106) | 0.946 | 3.49e-01 | 5.53e-01 | 0.244 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-injection-static | 0.01618 (0.0161) | 0.004461 (0.00999) | 3.380 | 1.44e-03 | 4.32e-03 * | 0.873 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-persistence-static | 0.01618 (0.0161) | 0.002282 (0.00532) | 4.478 | 7.59e-05 | 3.30e-04 * | 1.156 |
| Q1-memory-effect-pso | mean_ori_rad | pso-injection-static | pso-persistence-static | 0.004461 (0.00999) | 0.002282 (0.00532) | 1.054 | 2.97e-01 | 5.53e-01 | 0.272 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-cold-static | ga-cold-static | 0.01618 (0.0161) | 0.02075 (0.0139) | -1.179 | 2.43e-01 | 4.87e-01 | -0.304 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-injection-static | ga-injection-static | 0.004461 (0.00999) | 0.003235 (0.00455) | 0.612 | 5.44e-01 | 7.49e-01 | 0.158 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-injection-static | 0.2499 (0.252) | 0.03728 (0.0762) | 4.421 | 9.42e-05 | 3.30e-04 * | 1.142 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-persistence-static | 0.2499 (0.252) | 0.01388 (0.0202) | 5.111 | 1.80e-05 | 1.08e-04 * | 1.320 |
| Q1-memory-effect-pso | max_ori_rad | pso-injection-static | pso-persistence-static | 0.03728 (0.0762) | 0.01388 (0.0202) | 1.626 | 1.14e-01 | 2.62e-01 | 0.420 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-cold-static | ga-cold-static | 0.2499 (0.252) | 0.2662 (0.25) | -0.252 | 8.02e-01 | 9.47e-01 | -0.065 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-injection-static | ga-injection-static | 0.03728 (0.0762) | 0.02685 (0.0482) | 0.634 | 5.29e-01 | 7.49e-01 | 0.164 |

## Metric family: kinematic-executability

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | roughness_rad | pso-cold-static | pso-injection-static | 242.3 (39.2) | 4.221 (2.47) | 33.188 | 9.30e-25 | 2.74e-24 * | 8.569 |
| Q1-memory-effect-pso | roughness_rad | pso-cold-static | pso-persistence-static | 242.3 (39.2) | 2.908 (1.03) | 33.426 | 9.85e-25 | 2.74e-24 * | 8.631 |
| Q1-memory-effect-pso | roughness_rad | pso-injection-static | pso-persistence-static | 4.221 (2.47) | 2.908 (1.03) | 2.682 | 1.07e-02 | 1.57e-02 * | 0.693 |
| Q2-algorithm-effect-memory-matched | roughness_rad | pso-cold-static | ga-cold-static | 242.3 (39.2) | 122.7 (30.6) | 13.162 | 1.33e-18 | 2.76e-18 * | 3.398 |
| Q2-algorithm-effect-memory-matched | roughness_rad | pso-injection-static | ga-injection-static | 4.221 (2.47) | 4.932 (3.67) | -0.880 | 3.83e-01 | 4.35e-01 | -0.227 |
| Q1-memory-effect-pso | roughness_index | pso-cold-static | pso-injection-static | 8.356 (1.35) | 0.1456 (0.0853) | 33.188 | 9.30e-25 | 2.74e-24 * | 8.569 |
| Q1-memory-effect-pso | roughness_index | pso-cold-static | pso-persistence-static | 8.356 (1.35) | 0.1003 (0.0355) | 33.426 | 9.85e-25 | 2.74e-24 * | 8.631 |
| Q1-memory-effect-pso | roughness_index | pso-injection-static | pso-persistence-static | 0.1456 (0.0853) | 0.1003 (0.0355) | 2.682 | 1.07e-02 | 1.57e-02 * | 0.693 |
| Q2-algorithm-effect-memory-matched | roughness_index | pso-cold-static | ga-cold-static | 8.356 (1.35) | 4.232 (1.06) | 13.162 | 1.33e-18 | 2.76e-18 * | 3.398 |
| Q2-algorithm-effect-memory-matched | roughness_index | pso-injection-static | ga-injection-static | 0.1456 (0.0853) | 0.1701 (0.126) | -0.880 | 3.83e-01 | 4.35e-01 | -0.227 |
| Q1-memory-effect-pso | jerk_rad | pso-cold-static | pso-injection-static | 432.5 (71) | 2.733 (5.28) | 33.080 | 9.04e-25 | 2.74e-24 * | 8.541 |
| Q1-memory-effect-pso | jerk_rad | pso-cold-static | pso-persistence-static | 432.5 (71) | 1.588 (0.722) | 33.258 | 1.19e-24 | 2.97e-24 * | 8.587 |
| Q1-memory-effect-pso | jerk_rad | pso-injection-static | pso-persistence-static | 2.733 (5.28) | 1.588 (0.722) | 1.178 | 2.48e-01 | 3.10e-01 | 0.304 |
| Q2-algorithm-effect-memory-matched | jerk_rad | pso-cold-static | ga-cold-static | 432.5 (71) | 224 (55.1) | 12.711 | 5.89e-18 | 1.13e-17 * | 3.282 |
| Q2-algorithm-effect-memory-matched | jerk_rad | pso-injection-static | ga-injection-static | 2.733 (5.28) | 3.815 (6.56) | -0.704 | 4.85e-01 | 5.27e-01 | -0.182 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-cold-static | pso-injection-static | 6.002 (0.487) | 0.3059 (0.765) | 34.413 | 4.36e-36 | 1.09e-34 * | 8.885 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-cold-static | pso-persistence-static | 6.002 (0.487) | 0.1042 (0.0683) | 65.679 | 4.24e-34 | 3.53e-33 * | 16.958 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-injection-static | pso-persistence-static | 0.3059 (0.765) | 0.1042 (0.0683) | 1.440 | 1.61e-01 | 2.23e-01 | 0.372 |
| Q2-algorithm-effect-memory-matched | peak_joint_velocity_rad | pso-cold-static | ga-cold-static | 6.002 (0.487) | 4.71 (0.989) | 6.420 | 9.60e-08 | 1.60e-07 * | 1.658 |
| Q2-algorithm-effect-memory-matched | peak_joint_velocity_rad | pso-injection-static | ga-injection-static | 0.3059 (0.765) | 0.3791 (0.729) | -0.379 | 7.06e-01 | 7.35e-01 | -0.098 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-cold-static | pso-injection-static | 9.995 (1.35) | 0.2867 (0.774) | 34.182 | 1.86e-34 | 2.33e-33 * | 8.826 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-cold-static | pso-persistence-static | 9.995 (1.35) | 0.1049 (0.0739) | 40.082 | 4.58e-27 | 2.86e-26 * | 10.349 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-injection-static | pso-persistence-static | 0.2867 (0.774) | 0.1049 (0.0739) | 1.281 | 2.10e-01 | 2.77e-01 | 0.331 |
| Q2-algorithm-effect-memory-matched | peak_joint_acceleration_rad | pso-cold-static | ga-cold-static | 9.995 (1.35) | 7.995 (1.07) | 6.351 | 4.28e-08 | 7.64e-08 * | 1.640 |
| Q2-algorithm-effect-memory-matched | peak_joint_acceleration_rad | pso-injection-static | ga-injection-static | 0.2867 (0.774) | 0.3459 (0.721) | -0.306 | 7.60e-01 | 7.60e-01 | -0.079 |

## Metric family: computation

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | wall_time_s | pso-cold-static | pso-injection-static | 10.23 (0.175) | 10.56 (0.146) | -7.831 | 1.44e-10 | 1.44e-10 * | -2.022 |
| Q1-memory-effect-pso | wall_time_s | pso-cold-static | pso-persistence-static | 10.23 (0.175) | 14.96 (0.649) | -38.523 | 3.98e-29 | 9.95e-29 * | -9.947 |
| Q1-memory-effect-pso | wall_time_s | pso-injection-static | pso-persistence-static | 10.56 (0.146) | 14.96 (0.649) | -36.252 | 1.68e-27 | 2.79e-27 * | -9.360 |
| Q2-algorithm-effect-memory-matched | wall_time_s | pso-cold-static | ga-cold-static | 10.23 (0.175) | 30.99 (2.17) | -52.100 | 1.69e-30 | 8.43e-30 * | -13.452 |
| Q2-algorithm-effect-memory-matched | wall_time_s | pso-injection-static | ga-injection-static | 10.56 (0.146) | 25.06 (2.45) | -32.375 | 1.95e-24 | 2.44e-24 * | -8.359 |

## Metric family: effort

### Rank-based planned comparisons (primary — heavy-tailed family)

| question | metric | A | B | median A (IQR) | median B (IQR) | U | p raw | p adj (FDR) | Cliff's delta |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-injection-static | 115.2 (76.4) | 62 (100) | 618.0 | 1.01e-02 | 1.44e-02 * | 0.373 |
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-persistence-static | 115.2 (76.4) | 160 (13.1) | 299.0 | 1.82e-02 | 2.27e-02 * | -0.336 |
| Q1-memory-effect-pso | 0.001 | pso-injection-static | pso-persistence-static | 62 (100) | 160 (13.1) | 264.0 | 3.80e-03 | 7.55e-03 * | -0.413 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-cold-static | ga-cold-static | 115.2 (76.4) | 160 (0) | 195.0 | 2.21e-06 | 1.11e-05 * | -0.567 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-injection-static | ga-injection-static | 62 (100) | 160 (0) | 180.0 | 8.74e-07 | 8.74e-06 * | -0.600 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-injection-static | 72.5 (110) | 29.25 (133) | 651.5 | 2.22e-03 | 5.54e-03 * | 0.448 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-persistence-static | 72.5 (110) | 34.5 (133) | 637.0 | 4.53e-03 | 7.55e-03 * | 0.416 |
| Q1-memory-effect-pso | 0.01 | pso-injection-static | pso-persistence-static | 29.25 (133) | 34.5 (133) | 423.0 | 6.90e-01 | 6.90e-01 | -0.060 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-cold-static | ga-cold-static | 72.5 (110) | 123.2 (16) | 390.0 | 3.77e-01 | 4.18e-01 | -0.133 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-injection-static | ga-injection-static | 29.25 (133) | 17.75 (8.38) | 731.0 | 2.81e-05 | 9.36e-05 * | 0.624 |

Kruskal-Wallis omnibus across the PSO memory triple (cold / injection / persistence, static):

- 0.001: H = 12.904, p = 1.58e-03
- 0.01: H = 11.754, p = 2.80e-03

### Welch t (secondary check)

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-injection-static | 120.1 (37.3) | 100.2 (49.7) | 1.754 | 8.52e-02 | 1.10e-01 | 0.453 |
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-persistence-static | 120.1 (37.3) | 147.4 (19.6) | -3.546 | 9.43e-04 | 2.36e-03 * | -0.916 |
| Q1-memory-effect-pso | 0.001 | pso-injection-static | pso-persistence-static | 100.2 (49.7) | 147.4 (19.6) | -4.840 | 2.20e-05 | 7.34e-05 * | -1.250 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-cold-static | ga-cold-static | 120.1 (37.3) | 160 (0) | -5.856 | 2.37e-06 | 1.18e-05 * | -1.512 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-injection-static | ga-injection-static | 100.2 (49.7) | 160 (0) | -6.594 | 3.17e-07 | 3.17e-06 * | -1.702 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-injection-static | 100.3 (53.9) | 72.78 (62.9) | 1.822 | 7.38e-02 | 1.10e-01 | 0.470 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-persistence-static | 100.3 (53.9) | 74.52 (62.1) | 1.720 | 9.08e-02 | 1.10e-01 | 0.444 |
| Q1-memory-effect-pso | 0.01 | pso-injection-static | pso-persistence-static | 72.78 (62.9) | 74.52 (62.1) | -0.107 | 9.15e-01 | 9.15e-01 | -0.028 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-cold-static | ga-cold-static | 100.3 (53.9) | 125.1 (13.5) | -2.438 | 2.04e-02 | 4.08e-02 * | -0.630 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-injection-static | ga-injection-static | 72.78 (62.9) | 46.58 (57.9) | 1.678 | 9.88e-02 | 1.10e-01 | 0.433 |

## Equivalence (TOST): pso-injection-static vs ga-injection-static

TOST establishes equivalence within +-bound when BOTH one-sided Welch
tests reject (p_tost = max of the two one-sided p < 0.05). Bounds are
fractions of the cold->injection memory effect, at three anchors:
**min** = min(PSO, GA effects) — the HEADLINE, most conservative;
mean = average of the two; pso = PSO-only effect. Bootstrap CIs are
BCa, 10000 resamples, seed 20260704.

### Difference estimates (bootstrap 95% BCa CIs)

| metric | mean A | mean B | dmean (A-B) [95% CI] | Cohen's d [95% CI] |
|---|---|---|---|---|
| roughness_rad | 4.221 | 4.932 | -0.7103 [-2.662, 0.5786] | -0.227 [-0.666, 0.302] |
| jerk_rad | 2.733 | 3.815 | -1.082 [-4.543, 1.521] | -0.182 [-0.651, 0.346] |
| rmse_pos_mm | 31.74 | 25.04 | 6.699 [-16.19, 29.9] | 0.146 [-0.380, 0.659] |
| rmse_ori_rad | 0.0105 | 0.006635 | 0.003861 [-0.00309, 0.01274] | 0.244 [-0.292, 0.673] |

### TOST sensitivity over bound anchors

| metric | anchor | anchor value | fraction | +-bound | p TOST | equivalent? |
|---|---|---|---|---|---|---|
| roughness_rad | min | 117.8 | 10% | 11.78 | 5.09e-19 | YES |
| roughness_rad | min | 117.8 | 20% | 23.56 | 4.16e-33 | YES |
| roughness_rad | mean | 178 | 10% | 17.8 | 3.52e-27 | YES |
| roughness_rad | mean | 178 | 20% | 35.59 | 4.44e-42 | YES |
| roughness_rad | pso | 238.1 | 10% | 23.81 | 2.47e-33 | YES |
| roughness_rad | pso | 238.1 | 20% | 47.62 | 1.69e-48 | YES |
| jerk_rad | min | 220.2 | 10% | 22.02 | 1.25e-19 | YES |
| jerk_rad | min | 220.2 | 20% | 44.04 | 1.19e-34 | YES |
| jerk_rad | mean | 325 | 10% | 32.5 | 8.79e-28 | YES |
| jerk_rad | mean | 325 | 20% | 65 | 8.78e-44 | YES |
| jerk_rad | pso | 429.8 | 10% | 42.98 | 4.31e-34 | YES |
| jerk_rad | pso | 429.8 | 20% | 85.96 | 1.88e-50 | YES |
| rmse_pos_mm | min | 22.52 | 10% | 2.252 | 6.46e-01 | no |
| rmse_pos_mm | min | 22.52 | 20% | 4.504 | 5.73e-01 | no |
| rmse_pos_mm | mean | 26.22 | 10% | 2.622 | 6.34e-01 | no |
| rmse_pos_mm | mean | 26.22 | 20% | 5.243 | 5.49e-01 | no |
| rmse_pos_mm | pso | 29.91 | 10% | 2.991 | 6.22e-01 | no |
| rmse_pos_mm | pso | 29.91 | 20% | 5.983 | 5.24e-01 | no |
| rmse_ori_rad | min | 0.0441 | 10% | 0.00441 | 4.47e-01 | no |
| rmse_ori_rad | min | 0.0441 | 20% | 0.00882 | 1.15e-01 | no |
| rmse_ori_rad | mean | 0.04681 | 10% | 0.004681 | 4.21e-01 | no |
| rmse_ori_rad | mean | 0.04681 | 20% | 0.009362 | 9.22e-02 | no |
| rmse_ori_rad | pso | 0.0441 | 10% | 0.00441 | 4.47e-01 | no |
| rmse_ori_rad | pso | 0.0441 | 20% | 0.00882 | 1.15e-01 | no |

Effect-size guides: Cohen's d — |d| < 0.2 trivial, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large. Cliff's delta — |delta| < 0.147 negligible, 0.147-0.33 small, 0.33-0.474 medium, > 0.474 large. `*` = significant at FDR-adjusted p < 0.05 (one BH pass per family per track).
