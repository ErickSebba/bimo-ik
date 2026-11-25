# Planned comparisons — trajectory 'corner' (N=30 runs/cell)

## Metric family: accuracy

### Rank-based planned comparisons (primary — heavy-tailed family)

| question | metric | A | B | median A (IQR) | median B (IQR) | U | p raw | p adj (FDR) | Cliff's delta |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-injection-static | 66.63 (15.3) | 5.407 (91.3) | 638.0 | 5.57e-03 | 9.28e-03 * | 0.418 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-persistence-static | 66.63 (15.3) | 19.8 (161) | 581.0 | 5.37e-02 | 7.67e-02 | 0.291 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-injection-static | pso-persistence-static | 5.407 (91.3) | 19.8 (161) | 244.0 | 2.38e-03 | 5.10e-03 * | -0.458 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-cold-static | ga-cold-static | 66.63 (15.3) | 33.22 (17.5) | 889.0 | 8.99e-11 | 1.64e-09 * | 0.976 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-injection-static | ga-injection-static | 5.407 (91.3) | 2.795 (10.1) | 368.0 | 2.28e-01 | 2.36e-01 | -0.182 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-injection-static | 32.17 (9.98) | 1.824 (58.6) | 633.0 | 6.97e-03 | 1.10e-02 * | 0.407 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-persistence-static | 32.17 (9.98) | 12.47 (161) | 557.0 | 1.15e-01 | 1.28e-01 | 0.238 |
| Q1-memory-effect-pso | mean_pos_mm | pso-injection-static | pso-persistence-static | 1.824 (58.6) | 12.47 (161) | 232.0 | 1.30e-03 | 3.55e-03 * | -0.484 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-cold-static | ga-cold-static | 32.17 (9.98) | 11.6 (7.48) | 882.0 | 1.78e-10 | 1.78e-09 * | 0.960 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-injection-static | ga-injection-static | 1.824 (58.6) | 1.16 (3.51) | 371.0 | 2.46e-01 | 2.46e-01 | -0.176 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-injection-static | 195.6 (23.2) | 22.27 (177) | 776.0 | 1.49e-06 | 5.59e-06 * | 0.724 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-persistence-static | 195.6 (23.2) | 56.21 (171) | 660.0 | 1.95e-03 | 4.51e-03 * | 0.467 |
| Q1-memory-effect-pso | max_pos_mm | pso-injection-static | pso-persistence-static | 22.27 (177) | 56.21 (171) | 255.0 | 4.03e-03 | 7.12e-03 * | -0.433 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-cold-static | ga-cold-static | 195.6 (23.2) | 138.2 (84.7) | 750.0 | 9.51e-06 | 3.17e-05 * | 0.667 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-injection-static | ga-injection-static | 22.27 (177) | 14.66 (42.6) | 356.0 | 1.67e-01 | 1.79e-01 | -0.209 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-injection-static | 0.07659 (0.0542) | 3.6e-05 (0.0109) | 859.0 | 1.55e-09 | 9.28e-09 * | 0.909 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-persistence-static | 0.07659 (0.0542) | 0.003542 (0.0126) | 778.0 | 1.29e-06 | 5.51e-06 * | 0.729 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-injection-static | pso-persistence-static | 3.6e-05 (0.0109) | 0.003542 (0.0126) | 334.0 | 8.77e-02 | 1.05e-01 | -0.258 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-cold-static | ga-cold-static | 0.07659 (0.0542) | 0.04642 (0.0855) | 619.0 | 1.27e-02 | 1.91e-02 * | 0.376 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-injection-static | ga-injection-static | 3.6e-05 (0.0109) | 0.003109 (0.00354) | 253.0 | 3.67e-03 | 6.88e-03 * | -0.438 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-injection-static | 0.0269 (0.0243) | 9.174e-06 (0.00459) | 828.0 | 2.39e-08 | 1.19e-07 * | 0.840 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-persistence-static | 0.0269 (0.0243) | 0.0009037 (0.00459) | 747.0 | 1.17e-05 | 3.50e-05 * | 0.660 |
| Q1-memory-effect-pso | mean_ori_rad | pso-injection-static | pso-persistence-static | 9.174e-06 (0.00459) | 0.0009037 (0.00459) | 337.0 | 9.63e-02 | 1.11e-01 | -0.251 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-cold-static | ga-cold-static | 0.0269 (0.0243) | 0.01835 (0.0221) | 574.0 | 6.79e-02 | 8.48e-02 | 0.276 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-injection-static | ga-injection-static | 9.174e-06 (0.00459) | 0.001642 (0.00138) | 238.0 | 1.77e-03 | 4.42e-03 * | -0.471 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-injection-static | 0.3107 (0.272) | 0.0001765 (0.0364) | 887.0 | 1.09e-10 | 1.64e-09 * | 0.971 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-persistence-static | 0.3107 (0.272) | 0.01849 (0.0484) | 861.0 | 1.29e-09 | 9.28e-09 * | 0.913 |
| Q1-memory-effect-pso | max_ori_rad | pso-injection-static | pso-persistence-static | 0.0001765 (0.0364) | 0.01849 (0.0484) | 324.0 | 6.35e-02 | 8.29e-02 | -0.280 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-cold-static | ga-cold-static | 0.3107 (0.272) | 0.1963 (0.415) | 579.0 | 5.75e-02 | 7.84e-02 | 0.287 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-injection-static | ga-injection-static | 0.0001765 (0.0364) | 0.01307 (0.0201) | 246.0 | 2.62e-03 | 5.25e-03 * | -0.453 |

Kruskal-Wallis omnibus across the PSO memory triple (cold / injection / persistence, static):

- rmse_pos_mm: H = 12.826, p = 1.64e-03
- mean_pos_mm: H = 12.563, p = 1.87e-03
- max_pos_mm: H = 27.300, p = 1.18e-06
- rmse_ori_rad: H = 42.185, p = 6.91e-10
- mean_ori_rad: H = 35.681, p = 1.79e-08
- max_ori_rad: H = 54.569, p = 1.41e-12

### Welch t (secondary check)

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-injection-static | 67.3 (10.5) | 46.62 (69.3) | 1.616 | 1.17e-01 | 2.33e-01 | 0.417 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-persistence-static | 67.3 (10.5) | 76.17 (90.1) | -0.535 | 5.96e-01 | 5.96e-01 | -0.138 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-injection-static | pso-persistence-static | 46.62 (69.3) | 76.17 (90.1) | -1.423 | 1.60e-01 | 2.83e-01 | -0.367 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-cold-static | ga-cold-static | 67.3 (10.5) | 33.22 (12.3) | 11.582 | 1.47e-16 | 4.41e-15 * | 2.990 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-injection-static | ga-injection-static | 46.62 (69.3) | 27.75 (53.9) | 1.177 | 2.44e-01 | 3.06e-01 | 0.304 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-injection-static | 31.61 (7.78) | 42.06 (68.1) | -0.835 | 4.10e-01 | 4.56e-01 | -0.216 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-persistence-static | 31.61 (7.78) | 70.41 (90.9) | -2.328 | 2.70e-02 | 6.74e-02 | -0.601 |
| Q1-memory-effect-pso | mean_pos_mm | pso-injection-static | pso-persistence-static | 42.06 (68.1) | 70.41 (90.9) | -1.367 | 1.77e-01 | 2.83e-01 | -0.353 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-cold-static | ga-cold-static | 31.61 (7.78) | 12.72 (5.42) | 10.911 | 4.99e-15 | 7.49e-14 * | 2.817 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-injection-static | ga-injection-static | 42.06 (68.1) | 22.77 (51.6) | 1.237 | 2.21e-01 | 2.99e-01 | 0.319 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-injection-static | 204.3 (31.2) | 68.84 (87.7) | 7.971 | 1.75e-09 | 8.74e-09 * | 2.058 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-persistence-static | 204.3 (31.2) | 108.3 (98.1) | 5.110 | 1.17e-05 | 3.89e-05 * | 1.319 |
| Q1-memory-effect-pso | max_pos_mm | pso-injection-static | pso-persistence-static | 68.84 (87.7) | 108.3 (98.1) | -1.641 | 1.06e-01 | 2.28e-01 | -0.424 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-cold-static | ga-cold-static | 204.3 (31.2) | 143 (51.7) | 5.562 | 1.19e-06 | 5.08e-06 * | 1.436 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-injection-static | ga-injection-static | 68.84 (87.7) | 52.47 (72.9) | 0.786 | 4.35e-01 | 4.66e-01 | 0.203 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-injection-static | 0.08524 (0.0394) | 0.01233 (0.0291) | 8.153 | 6.05e-11 | 3.63e-10 * | 2.105 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-persistence-static | 0.08524 (0.0394) | 0.02621 (0.0505) | 5.050 | 5.21e-06 | 1.96e-05 * | 1.304 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-injection-static | pso-persistence-static | 0.01233 (0.0291) | 0.02621 (0.0505) | -1.305 | 1.98e-01 | 2.83e-01 | -0.337 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-cold-static | ga-cold-static | 0.08524 (0.0394) | 0.05863 (0.0475) | 2.362 | 2.17e-02 | 5.91e-02 | 0.610 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-injection-static | ga-injection-static | 0.01233 (0.0291) | 0.005111 (0.0055) | 1.334 | 1.92e-01 | 2.83e-01 | 0.345 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-injection-static | 0.02885 (0.0162) | 0.006771 (0.0198) | 4.732 | 1.56e-05 | 4.69e-05 * | 1.222 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-persistence-static | 0.02885 (0.0162) | 0.02019 (0.044) | 1.012 | 3.18e-01 | 3.82e-01 | 0.261 |
| Q1-memory-effect-pso | mean_ori_rad | pso-injection-static | pso-persistence-static | 0.006771 (0.0198) | 0.02019 (0.044) | -1.523 | 1.36e-01 | 2.54e-01 | -0.393 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-cold-static | ga-cold-static | 0.02885 (0.0162) | 0.02065 (0.0126) | 2.196 | 3.24e-02 | 7.47e-02 | 0.567 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-injection-static | ga-injection-static | 0.006771 (0.0198) | 0.002315 (0.00162) | 1.228 | 2.29e-01 | 2.99e-01 | 0.317 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-injection-static | 0.3507 (0.155) | 0.03339 (0.0647) | 10.343 | 1.04e-12 | 1.04e-11 * | 2.670 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-persistence-static | 0.3507 (0.155) | 0.05022 (0.0789) | 9.459 | 4.47e-12 | 3.35e-11 * | 2.442 |
| Q1-memory-effect-pso | max_ori_rad | pso-injection-static | pso-persistence-static | 0.03339 (0.0647) | 0.05022 (0.0789) | -0.903 | 3.70e-01 | 4.27e-01 | -0.233 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-cold-static | ga-cold-static | 0.3507 (0.155) | 0.2788 (0.254) | 1.324 | 1.92e-01 | 2.83e-01 | 0.342 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-injection-static | ga-injection-static | 0.03339 (0.0647) | 0.02398 (0.0306) | 0.720 | 4.76e-01 | 4.92e-01 | 0.186 |

## Metric family: kinematic-executability

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | roughness_rad | pso-cold-static | pso-injection-static | 223.8 (39.3) | 4.613 (2.39) | 30.524 | 1.02e-23 | 2.99e-23 * | 7.881 |
| Q1-memory-effect-pso | roughness_rad | pso-cold-static | pso-persistence-static | 223.8 (39.3) | 2.757 (1.6) | 30.815 | 9.14e-24 | 2.99e-23 * | 7.956 |
| Q1-memory-effect-pso | roughness_rad | pso-injection-static | pso-persistence-static | 4.613 (2.39) | 2.757 (1.6) | 3.534 | 8.84e-04 | 1.30e-03 * | 0.913 |
| Q2-algorithm-effect-memory-matched | roughness_rad | pso-cold-static | ga-cold-static | 223.8 (39.3) | 100.8 (39.6) | 12.083 | 1.77e-17 | 3.70e-17 * | 3.120 |
| Q2-algorithm-effect-memory-matched | roughness_rad | pso-injection-static | ga-injection-static | 4.613 (2.39) | 5.202 (2.95) | -0.849 | 4.00e-01 | 4.54e-01 | -0.219 |
| Q1-memory-effect-pso | roughness_index | pso-cold-static | pso-injection-static | 7.717 (1.35) | 0.1591 (0.0825) | 30.524 | 1.02e-23 | 2.99e-23 * | 7.881 |
| Q1-memory-effect-pso | roughness_index | pso-cold-static | pso-persistence-static | 7.717 (1.35) | 0.09506 (0.0551) | 30.815 | 9.14e-24 | 2.99e-23 * | 7.956 |
| Q1-memory-effect-pso | roughness_index | pso-injection-static | pso-persistence-static | 0.1591 (0.0825) | 0.09506 (0.0551) | 3.534 | 8.84e-04 | 1.30e-03 * | 0.913 |
| Q2-algorithm-effect-memory-matched | roughness_index | pso-cold-static | ga-cold-static | 7.717 (1.35) | 3.476 (1.37) | 12.083 | 1.77e-17 | 3.70e-17 * | 3.120 |
| Q2-algorithm-effect-memory-matched | roughness_index | pso-injection-static | ga-injection-static | 0.1591 (0.0825) | 0.1794 (0.102) | -0.849 | 4.00e-01 | 4.54e-01 | -0.219 |
| Q1-memory-effect-pso | jerk_rad | pso-cold-static | pso-injection-static | 401.1 (71.6) | 3.262 (4.96) | 30.383 | 1.08e-23 | 2.99e-23 * | 7.845 |
| Q1-memory-effect-pso | jerk_rad | pso-cold-static | pso-persistence-static | 401.1 (71.6) | 1.613 (1.25) | 30.578 | 1.26e-23 | 3.14e-23 * | 7.895 |
| Q1-memory-effect-pso | jerk_rad | pso-injection-static | pso-persistence-static | 3.262 (4.96) | 1.613 (1.25) | 1.764 | 8.70e-02 | 1.15e-01 | 0.456 |
| Q2-algorithm-effect-memory-matched | jerk_rad | pso-cold-static | ga-cold-static | 401.1 (71.6) | 182.7 (74.7) | 11.562 | 1.12e-16 | 2.16e-16 * | 2.985 |
| Q2-algorithm-effect-memory-matched | jerk_rad | pso-injection-static | ga-injection-static | 3.262 (4.96) | 3.557 (5.95) | -0.209 | 8.35e-01 | 9.08e-01 | -0.054 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-cold-static | pso-injection-static | 5.724 (0.665) | 0.3574 (0.811) | 28.044 | 1.36e-34 | 3.39e-33 * | 7.241 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-cold-static | pso-persistence-static | 5.724 (0.665) | 0.105 (0.0764) | 46.015 | 3.22e-29 | 2.69e-28 * | 11.881 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-injection-static | pso-persistence-static | 0.3574 (0.811) | 0.105 (0.0764) | 1.698 | 1.00e-01 | 1.25e-01 | 0.438 |
| Q2-algorithm-effect-memory-matched | peak_joint_velocity_rad | pso-cold-static | ga-cold-static | 5.724 (0.665) | 4.165 (1.03) | 6.964 | 7.14e-09 | 1.19e-08 * | 1.798 |
| Q2-algorithm-effect-memory-matched | peak_joint_velocity_rad | pso-injection-static | ga-injection-static | 0.3574 (0.811) | 0.3769 (0.77) | -0.096 | 9.24e-01 | 9.55e-01 | -0.025 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-cold-static | pso-injection-static | 9.762 (1.59) | 0.3846 (0.809) | 28.729 | 1.05e-29 | 1.31e-28 * | 7.418 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-cold-static | pso-persistence-static | 9.762 (1.59) | 0.1041 (0.0834) | 33.140 | 1.07e-24 | 6.68e-24 * | 8.557 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-injection-static | pso-persistence-static | 0.3846 (0.809) | 0.1041 (0.0834) | 1.888 | 6.89e-02 | 9.57e-02 | 0.487 |
| Q2-algorithm-effect-memory-matched | peak_joint_acceleration_rad | pso-cold-static | ga-cold-static | 9.762 (1.59) | 7.229 (1.18) | 6.999 | 4.36e-09 | 7.78e-09 * | 1.807 |
| Q2-algorithm-effect-memory-matched | peak_joint_acceleration_rad | pso-injection-static | ga-injection-static | 0.3846 (0.809) | 0.3961 (0.779) | -0.056 | 9.55e-01 | 9.55e-01 | -0.015 |

## Metric family: computation

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | wall_time_s | pso-cold-static | pso-injection-static | 10.49 (0.0918) | 10.5 (0.117) | -0.157 | 8.76e-01 | 8.76e-01 | -0.041 |
| Q1-memory-effect-pso | wall_time_s | pso-cold-static | pso-persistence-static | 10.49 (0.0918) | 10.8 (0.173) | -8.586 | 5.69e-11 | 9.48e-11 * | -2.217 |
| Q1-memory-effect-pso | wall_time_s | pso-injection-static | pso-persistence-static | 10.5 (0.117) | 10.8 (0.173) | -7.932 | 1.84e-10 | 2.30e-10 * | -2.048 |
| Q2-algorithm-effect-memory-matched | wall_time_s | pso-cold-static | ga-cold-static | 10.49 (0.0918) | 26.41 (3.82) | -22.798 | 4.44e-20 | 1.11e-19 * | -5.886 |
| Q2-algorithm-effect-memory-matched | wall_time_s | pso-injection-static | ga-injection-static | 10.5 (0.117) | 29.09 (2.23) | -45.674 | 1.10e-28 | 5.49e-28 * | -11.793 |

## Metric family: effort

### Rank-based planned comparisons (primary — heavy-tailed family)

| question | metric | A | B | median A (IQR) | median B (IQR) | U | p raw | p adj (FDR) | Cliff's delta |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-injection-static | 80.75 (10.9) | 61.25 (101) | 643.5 | 4.16e-03 | 4.16e-03 * | 0.430 |
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-persistence-static | 80.75 (10.9) | 160 (0) | 54.5 | 7.19e-10 | 3.60e-09 * | -0.879 |
| Q1-memory-effect-pso | 0.001 | pso-injection-static | pso-persistence-static | 61.25 (101) | 160 (0) | 157.5 | 1.81e-06 | 3.01e-06 * | -0.650 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-cold-static | ga-cold-static | 80.75 (10.9) | 160 (0) | 45.0 | 5.73e-11 | 5.73e-10 * | -0.900 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-injection-static | ga-injection-static | 61.25 (101) | 160 (0) | 139.5 | 1.11e-07 | 3.70e-07 * | -0.690 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-injection-static | 48.25 (6.12) | 29 (132) | 643.5 | 4.15e-03 | 4.16e-03 * | 0.430 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-persistence-static | 48.25 (6.12) | 160 (100) | 195.5 | 1.23e-04 | 1.54e-04 * | -0.566 |
| Q1-memory-effect-pso | 0.01 | pso-injection-static | pso-persistence-static | 29 (132) | 160 (100) | 191.5 | 6.28e-05 | 8.97e-05 * | -0.574 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-cold-static | ga-cold-static | 48.25 (6.12) | 119.2 (15) | 95.0 | 1.58e-07 | 3.95e-07 * | -0.789 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-injection-static | ga-injection-static | 29 (132) | 18.5 (4.88) | 798.0 | 2.36e-07 | 4.72e-07 * | 0.773 |

Kruskal-Wallis omnibus across the PSO memory triple (cold / injection / persistence, static):

- 0.001: H = 39.390, p = 2.80e-09
- 0.01: H = 23.905, p = 6.44e-06

### Welch t (secondary check)

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-injection-static | 91.87 (25.7) | 90.8 (46.2) | 0.111 | 9.12e-01 | 9.12e-01 | 0.029 |
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-persistence-static | 91.87 (25.7) | 157.1 (9.21) | -13.090 | 2.54e-15 | 2.54e-14 * | -3.380 |
| Q1-memory-effect-pso | 0.001 | pso-injection-static | pso-persistence-static | 90.8 (46.2) | 157.1 (9.21) | -7.713 | 9.95e-09 | 1.99e-08 * | -1.991 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-cold-static | ga-cold-static | 91.87 (25.7) | 160 (0) | -14.516 | 7.81e-15 | 3.90e-14 * | -3.748 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-injection-static | ga-injection-static | 90.8 (46.2) | 160 (0.183) | -8.201 | 4.83e-09 | 1.21e-08 * | -2.117 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-injection-static | 63.58 (36) | 67.98 (61.3) | -0.339 | 7.36e-01 | 8.18e-01 | -0.088 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-persistence-static | 63.58 (36) | 120.4 (52.3) | -4.897 | 1.01e-05 | 1.68e-05 * | -1.264 |
| Q1-memory-effect-pso | 0.01 | pso-injection-static | pso-persistence-static | 67.98 (61.3) | 120.4 (52.3) | -3.560 | 7.59e-04 | 1.08e-03 * | -0.919 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-cold-static | ga-cold-static | 63.58 (36) | 120.3 (12.1) | -8.171 | 1.15e-09 | 3.85e-09 * | -2.110 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-injection-static | ga-injection-static | 67.98 (61.3) | 37.28 (49) | 2.142 | 3.66e-02 | 4.57e-02 * | 0.553 |

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
| roughness_rad | 4.613 | 5.202 | -0.5884 [-2.121, 0.5893] | -0.219 [-0.668, 0.315] |
| jerk_rad | 3.262 | 3.557 | -0.2956 [-3.378, 2.127] | -0.054 [-0.547, 0.470] |
| rmse_pos_mm | 46.62 | 27.75 | 18.87 [-10.79, 50.31] | 0.304 [-0.207, 0.810] |
| rmse_ori_rad | 0.01233 | 0.005111 | 0.007216 [0.0002161, 0.02431] | 0.345 [-0.129, 0.679] |

### TOST sensitivity over bound anchors

| metric | anchor | anchor value | fraction | +-bound | p TOST | equivalent? |
|---|---|---|---|---|---|---|
| roughness_rad | min | 95.59 | 10% | 9.559 | 9.98e-19 | YES |
| roughness_rad | min | 95.59 | 20% | 19.12 | 9.57e-34 | YES |
| roughness_rad | mean | 157.4 | 10% | 15.74 | 2.69e-29 | YES |
| roughness_rad | mean | 157.4 | 20% | 31.48 | 1.55e-45 | YES |
| roughness_rad | pso | 219.2 | 10% | 21.92 | 6.19e-37 | YES |
| roughness_rad | pso | 219.2 | 20% | 43.84 | 1.64e-53 | YES |
| jerk_rad | min | 179.1 | 10% | 17.91 | 4.34e-18 | YES |
| jerk_rad | min | 179.1 | 20% | 35.83 | 1.61e-32 | YES |
| jerk_rad | mean | 288.5 | 10% | 28.85 | 1.03e-27 | YES |
| jerk_rad | mean | 288.5 | 20% | 57.7 | 1.33e-43 | YES |
| jerk_rad | pso | 397.9 | 10% | 39.79 | 6.57e-35 | YES |
| jerk_rad | pso | 397.9 | 20% | 79.57 | 2.74e-51 | YES |
| rmse_pos_mm | min | 5.468 | 10% | 0.5468 | 8.71e-01 | no |
| rmse_pos_mm | min | 5.468 | 20% | 1.094 | 8.64e-01 | no |
| rmse_pos_mm | mean | 13.07 | 10% | 1.307 | 8.61e-01 | no |
| rmse_pos_mm | mean | 13.07 | 20% | 2.614 | 8.42e-01 | no |
| rmse_pos_mm | pso | 20.68 | 10% | 2.068 | 8.50e-01 | no |
| rmse_pos_mm | pso | 20.68 | 20% | 4.135 | 8.19e-01 | no |
| rmse_ori_rad | min | 0.05352 | 10% | 0.005352 | 6.34e-01 | no |
| rmse_ori_rad | min | 0.05352 | 20% | 0.0107 | 2.62e-01 | no |
| rmse_ori_rad | mean | 0.06322 | 10% | 0.006322 | 5.65e-01 | no |
| rmse_ori_rad | mean | 0.06322 | 20% | 0.01264 | 1.62e-01 | no |
| rmse_ori_rad | pso | 0.07292 | 10% | 0.007292 | 4.95e-01 | no |
| rmse_ori_rad | pso | 0.07292 | 20% | 0.01458 | 9.15e-02 | no |

Effect-size guides: Cohen's d — |d| < 0.2 trivial, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large. Cliff's delta — |delta| < 0.147 negligible, 0.147-0.33 small, 0.33-0.474 medium, > 0.474 large. `*` = significant at FDR-adjusted p < 0.05 (one BH pass per family per track).
