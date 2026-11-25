# Planned comparisons — trajectory 'corner' (N=2 runs/cell)

## Metric family: accuracy

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-injection-static | 90.03 (1.11) | 165.6 (0.277) | -93.684 | 3.96e-03 | 3.96e-02 * | -93.684 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-persistence-static | 90.03 (1.11) | 10.29 (3.37) | 31.767 | 1.01e-02 | 4.35e-02 * | 31.767 |
| Q1-memory-effect-pso | rmse_pos_mm | pso-injection-static | pso-persistence-static | 165.6 (0.277) | 10.29 (3.37) | 64.892 | 9.30e-03 | 4.35e-02 * | 64.892 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-cold-static | ga-cold-static | 90.03 (1.11) | 128.9 (12.3) | -4.464 | 1.37e-01 | 1.79e-01 | -4.464 |
| Q2-algorithm-effect-memory-matched | rmse_pos_mm | pso-injection-static | ga-injection-static | 165.6 (0.277) | 35.09 (5.72) | 32.237 | 1.94e-02 | 6.48e-02 | 32.237 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-injection-static | 62.88 (6.97) | 160.9 (0.806) | -19.764 | 2.99e-02 | 7.61e-02 | -19.764 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-persistence-static | 62.88 (6.97) | 5.41 (2.15) | 11.144 | 3.79e-02 | 8.13e-02 | 11.144 |
| Q1-memory-effect-pso | mean_pos_mm | pso-injection-static | pso-persistence-static | 160.9 (0.806) | 5.41 (2.15) | 95.953 | 2.03e-03 | 3.17e-02 * | 95.953 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-cold-static | ga-cold-static | 62.88 (6.97) | 113.4 (9.88) | -5.904 | 3.51e-02 | 8.09e-02 | -5.904 |
| Q2-algorithm-effect-memory-matched | mean_pos_mm | pso-injection-static | ga-injection-static | 160.9 (0.806) | 23.62 (0.12) | 238.293 | 2.12e-03 | 3.17e-02 * | 238.293 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-injection-static | 186.6 (5.38) | 204.9 (0.642) | -4.764 | 1.27e-01 | 1.79e-01 | -4.764 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-persistence-static | 186.6 (5.38) | 31.32 (9.84) | 19.584 | 7.68e-03 | 4.35e-02 * | 19.584 |
| Q1-memory-effect-pso | max_pos_mm | pso-injection-static | pso-persistence-static | 204.9 (0.642) | 31.32 (9.84) | 24.893 | 2.49e-02 | 7.47e-02 | 24.893 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-cold-static | ga-cold-static | 186.6 (5.38) | 224.3 (25.2) | -2.062 | 2.71e-01 | 2.90e-01 | -2.062 |
| Q2-algorithm-effect-memory-matched | max_pos_mm | pso-injection-static | ga-injection-static | 204.9 (0.642) | 86.6 (33.1) | 5.051 | 1.24e-01 | 1.79e-01 | 5.051 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-injection-static | 0.09556 (0.0166) | 0.01667 (0.0104) | 5.693 | 4.29e-02 | 8.25e-02 | 5.693 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-persistence-static | 0.09556 (0.0166) | 0.03413 (0.00866) | 4.641 | 7.22e-02 | 1.20e-01 | 4.641 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-injection-static | pso-persistence-static | 0.01667 (0.0104) | 0.03413 (0.00866) | -1.822 | 2.14e-01 | 2.38e-01 | -1.822 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-cold-static | ga-cold-static | 0.09556 (0.0166) | 0.4818 (0.0444) | -11.514 | 3.05e-02 | 7.61e-02 | -11.514 |
| Q2-algorithm-effect-memory-matched | rmse_ori_rad | pso-injection-static | ga-injection-static | 0.01667 (0.0104) | 0.1031 (0.0309) | -3.742 | 1.30e-01 | 1.79e-01 | -3.742 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-injection-static | 0.07883 (0.0229) | 0.01229 (0.0064) | 3.956 | 1.31e-01 | 1.79e-01 | 3.956 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-persistence-static | 0.07883 (0.0229) | 0.02126 (0.000413) | 3.553 | 1.75e-01 | 2.09e-01 | 3.553 |
| Q1-memory-effect-pso | mean_ori_rad | pso-injection-static | pso-persistence-static | 0.01229 (0.0064) | 0.02126 (0.000413) | -1.980 | 2.96e-01 | 3.06e-01 | -1.980 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-cold-static | ga-cold-static | 0.07883 (0.0229) | 0.3677 (0.0205) | -13.295 | 5.89e-03 | 4.35e-02 * | -13.295 |
| Q2-algorithm-effect-memory-matched | mean_ori_rad | pso-injection-static | ga-injection-static | 0.01229 (0.0064) | 0.0715 (0.00811) | -8.109 | 1.73e-02 | 6.48e-02 | -8.109 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-injection-static | 0.1869 (0.0271) | 0.0419 (0.0331) | 4.794 | 4.40e-02 | 8.25e-02 | 4.794 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-persistence-static | 0.1869 (0.0271) | 0.08978 (0.0467) | 2.545 | 1.56e-01 | 1.95e-01 | 2.545 |
| Q1-memory-effect-pso | max_ori_rad | pso-injection-static | pso-persistence-static | 0.0419 (0.0331) | 0.08978 (0.0467) | -1.184 | 3.69e-01 | 3.69e-01 | -1.184 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-cold-static | ga-cold-static | 0.1869 (0.0271) | 0.8551 (0.0968) | -9.404 | 4.93e-02 | 8.70e-02 | -9.404 |
| Q2-algorithm-effect-memory-matched | max_ori_rad | pso-injection-static | ga-injection-static | 0.0419 (0.0331) | 0.2699 (0.122) | -2.555 | 2.11e-01 | 2.38e-01 | -2.555 |

## Metric family: kinematic-executability

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | smoothness_rad | pso-cold-static | pso-injection-static | 74.88 (48.1) | 1.876 (1.15) | 2.144 | 2.78e-01 | 4.22e-01 | 2.144 |
| Q1-memory-effect-pso | smoothness_rad | pso-cold-static | pso-persistence-static | 74.88 (48.1) | 4.454 (0.469) | 2.069 | 2.87e-01 | 4.22e-01 | 2.069 |
| Q1-memory-effect-pso | smoothness_rad | pso-injection-static | pso-persistence-static | 1.876 (1.15) | 4.454 (0.469) | -2.935 | 1.57e-01 | 4.22e-01 | -2.935 |
| Q2-algorithm-effect-memory-matched | smoothness_rad | pso-cold-static | ga-cold-static | 74.88 (48.1) | 46.98 (11.4) | 0.797 | 5.60e-01 | 6.45e-01 | 0.797 |
| Q2-algorithm-effect-memory-matched | smoothness_rad | pso-injection-static | ga-injection-static | 1.876 (1.15) | 4.326 (0.0505) | -3.009 | 2.04e-01 | 4.22e-01 | -3.009 |
| Q1-memory-effect-pso | smoothness_index | pso-cold-static | pso-injection-static | 8.32 (5.35) | 0.2084 (0.128) | 2.144 | 2.78e-01 | 4.22e-01 | 2.144 |
| Q1-memory-effect-pso | smoothness_index | pso-cold-static | pso-persistence-static | 8.32 (5.35) | 0.4949 (0.0521) | 2.069 | 2.87e-01 | 4.22e-01 | 2.069 |
| Q1-memory-effect-pso | smoothness_index | pso-injection-static | pso-persistence-static | 0.2084 (0.128) | 0.4949 (0.0521) | -2.935 | 1.57e-01 | 4.22e-01 | -2.935 |
| Q2-algorithm-effect-memory-matched | smoothness_index | pso-cold-static | ga-cold-static | 8.32 (5.35) | 5.22 (1.27) | 0.797 | 5.60e-01 | 6.45e-01 | 0.797 |
| Q2-algorithm-effect-memory-matched | smoothness_index | pso-injection-static | ga-injection-static | 0.2084 (0.128) | 0.4806 (0.00562) | -3.009 | 2.04e-01 | 4.22e-01 | -3.009 |
| Q1-memory-effect-pso | jerk_rad | pso-cold-static | pso-injection-static | 129.6 (76.7) | 2.529 (1.96) | 2.343 | 2.57e-01 | 4.22e-01 | 2.343 |
| Q1-memory-effect-pso | jerk_rad | pso-cold-static | pso-persistence-static | 129.6 (76.7) | 2.247 (0.967) | 2.348 | 2.56e-01 | 4.22e-01 | 2.348 |
| Q1-memory-effect-pso | jerk_rad | pso-injection-static | pso-persistence-static | 2.529 (1.96) | 2.247 (0.967) | 0.183 | 8.77e-01 | 8.77e-01 | 0.183 |
| Q2-algorithm-effect-memory-matched | jerk_rad | pso-cold-static | ga-cold-static | 129.6 (76.7) | 78.15 (17.8) | 0.924 | 5.13e-01 | 6.45e-01 | 0.924 |
| Q2-algorithm-effect-memory-matched | jerk_rad | pso-injection-static | ga-injection-static | 2.529 (1.96) | 3.237 (0.114) | -0.511 | 6.99e-01 | 7.59e-01 | -0.511 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-cold-static | pso-injection-static | 4.337 (0.631) | 0.1618 (0.0508) | 9.329 | 6.62e-02 | 3.59e-01 | 9.329 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-cold-static | pso-persistence-static | 4.337 (0.631) | 0.3562 (0.182) | 8.574 | 5.39e-02 | 3.59e-01 | 8.574 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-injection-static | pso-persistence-static | 0.1618 (0.0508) | 0.3562 (0.182) | -1.455 | 3.59e-01 | 4.99e-01 | -1.455 |
| Q2-algorithm-effect-memory-matched | peak_joint_velocity_rad | pso-cold-static | ga-cold-static | 4.337 (0.631) | 3.386 (0.287) | 1.942 | 2.44e-01 | 4.22e-01 | 1.942 |
| Q2-algorithm-effect-memory-matched | peak_joint_velocity_rad | pso-injection-static | ga-injection-static | 0.1618 (0.0508) | 0.3068 (0.0207) | -3.737 | 1.17e-01 | 4.22e-01 | -3.737 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-cold-static | pso-injection-static | 7.821 (0.399) | 0.2755 (0.156) | 24.891 | 1.06e-02 | 1.58e-01 | 24.891 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-cold-static | pso-persistence-static | 7.821 (0.399) | 0.3762 (0.139) | 24.903 | 1.26e-02 | 1.58e-01 | 24.903 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-injection-static | pso-persistence-static | 0.2755 (0.156) | 0.3762 (0.139) | -0.680 | 5.67e-01 | 6.45e-01 | -0.680 |
| Q2-algorithm-effect-memory-matched | peak_joint_acceleration_rad | pso-cold-static | ga-cold-static | 7.821 (0.399) | 6.329 (0.206) | 4.698 | 7.18e-02 | 3.59e-01 | 4.698 |
| Q2-algorithm-effect-memory-matched | peak_joint_acceleration_rad | pso-injection-static | ga-injection-static | 0.2755 (0.156) | 0.3029 (0.0131) | -0.246 | 8.46e-01 | 8.77e-01 | -0.246 |

## Metric family: computation

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | wall_time_s | pso-cold-static | pso-injection-static | 0.4306 (0.0136) | 0.4201 (0.00165) | 1.086 | 4.70e-01 | 4.70e-01 | 1.086 |
| Q1-memory-effect-pso | wall_time_s | pso-cold-static | pso-persistence-static | 0.4306 (0.0136) | 0.4138 (0.00433) | 1.663 | 3.13e-01 | 3.91e-01 | 1.663 |
| Q1-memory-effect-pso | wall_time_s | pso-injection-static | pso-persistence-static | 0.4201 (0.00165) | 0.4138 (0.00433) | 1.915 | 2.61e-01 | 3.91e-01 | 1.915 |
| Q2-algorithm-effect-memory-matched | wall_time_s | pso-cold-static | ga-cold-static | 0.4306 (0.0136) | 0.8806 (0.000312) | -46.638 | 1.36e-02 | 3.40e-02 * | -46.638 |
| Q2-algorithm-effect-memory-matched | wall_time_s | pso-injection-static | ga-injection-static | 0.4201 (0.00165) | 0.8855 (0.0129) | -50.491 | 1.12e-02 | 3.40e-02 * | -50.491 |

## Metric family: effort

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-injection-static | 20 (0) | 20 (0) | nan | nan | nan | 0.000 |
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-persistence-static | 20 (0) | 20 (0) | nan | nan | nan | 0.000 |
| Q1-memory-effect-pso | 0.001 | pso-injection-static | pso-persistence-static | 20 (0) | 20 (0) | nan | nan | nan | 0.000 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-cold-static | ga-cold-static | 20 (0) | 20 (0) | nan | nan | nan | 0.000 |
| Q2-algorithm-effect-memory-matched | 0.001 | pso-injection-static | ga-injection-static | 20 (0) | 20 (0) | nan | nan | nan | 0.000 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-injection-static | 20 (0) | 20 (0) | nan | nan | nan | 0.000 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-persistence-static | 20 (0) | 17.5 (1.41) | 2.500 | 2.42e-01 | nan | 2.500 |
| Q1-memory-effect-pso | 0.01 | pso-injection-static | pso-persistence-static | 20 (0) | 17.5 (1.41) | 2.500 | 2.42e-01 | nan | 2.500 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-cold-static | ga-cold-static | 20 (0) | 20 (0) | nan | nan | nan | 0.000 |
| Q2-algorithm-effect-memory-matched | 0.01 | pso-injection-static | ga-injection-static | 20 (0) | 20 (0) | nan | nan | nan | 0.000 |

Cohen's d guide: |d| < 0.2 trivial, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large. `*` = significant at FDR-adjusted p < 0.05.
