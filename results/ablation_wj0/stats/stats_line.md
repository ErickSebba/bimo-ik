# Planned comparisons — trajectory 'line' (N=30 runs/cell)

## Metric family: accuracy

### Rank-based planned comparisons (primary — heavy-tailed family)

| question | metric | A | B | median A (IQR) | median B (IQR) | U | p raw | p adj (FDR) | Cliff's delta |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-injection-static | 65.18 (16.5) | 13.55 (40.8) | 687.0 | 4.71e-04 | 5.66e-04 * | 0.527 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-injection-static | 39.07 (16.6) | 5.822 (26.6) | 674.0 | 9.52e-04 | 9.52e-04 * | 0.498 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-injection-static | 177.4 (28.6) | 44.68 (127) | 759.0 | 5.09e-06 | 7.64e-06 * | 0.687 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-injection-static | 0.04998 (0.0509) | 7.45e-05 (0.0114) | 802.0 | 2.03e-07 | 6.08e-07 * | 0.782 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-injection-static | 0.01566 (0.0182) | 3.595e-05 (0.00342) | 769.0 | 2.49e-06 | 4.98e-06 * | 0.709 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-injection-static | 0.2199 (0.22) | 0.0003052 (0.0508) | 838.0 | 1.01e-08 | 6.06e-08 * | 0.862 |

Kruskal-Wallis omnibus across the PSO memory triple (cold / injection / persistence, static):

- rmse_pos_mm: H = 12.277, p = 4.58e-04
- mean_pos_mm: H = 10.967, p = 9.27e-04
- max_pos_mm: H = 20.870, p = 4.91e-06
- rmse_ori_rad: H = 27.083, p = 1.95e-07
- mean_ori_rad: H = 22.243, p = 2.40e-06
- max_ori_rad: H = 32.906, p = 9.67e-09

### Welch t (secondary check)

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | rmse_pos_mm | pso-cold-static | pso-injection-static | 64.57 (14.5) | 34.31 (47.5) | 3.335 | 2.06e-03 | 3.09e-03 * | 0.861 |
| Q1-memory-effect-pso | mean_pos_mm | pso-cold-static | pso-injection-static | 37.98 (12.2) | 28.68 (43.9) | 1.119 | 2.71e-01 | 2.71e-01 | 0.289 |
| Q1-memory-effect-pso | max_pos_mm | pso-cold-static | pso-injection-static | 171.3 (32.6) | 62.57 (74) | 7.370 | 5.86e-09 | 3.52e-08 * | 1.903 |
| Q1-memory-effect-pso | rmse_ori_rad | pso-cold-static | pso-injection-static | 0.06497 (0.0459) | 0.01656 (0.0366) | 4.518 | 3.33e-05 | 6.66e-05 * | 1.167 |
| Q1-memory-effect-pso | mean_ori_rad | pso-cold-static | pso-injection-static | 0.01922 (0.0127) | 0.008905 (0.0246) | 2.042 | 4.73e-02 | 5.67e-02 | 0.527 |
| Q1-memory-effect-pso | max_ori_rad | pso-cold-static | pso-injection-static | 0.3001 (0.248) | 0.04673 (0.0929) | 5.234 | 6.85e-06 | 2.05e-05 * | 1.351 |

## Metric family: kinematic-executability

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | roughness_rad | pso-cold-static | pso-injection-static | 258.9 (42.9) | 5.219 (3.35) | 32.305 | 1.70e-24 | 2.13e-24 * | 8.341 |
| Q1-memory-effect-pso | roughness_index | pso-cold-static | pso-injection-static | 8.926 (1.48) | 0.18 (0.116) | 32.305 | 1.70e-24 | 2.13e-24 * | 8.341 |
| Q1-memory-effect-pso | jerk_rad | pso-cold-static | pso-injection-static | 460.1 (78.5) | 4.547 (7.14) | 31.638 | 2.62e-24 | 2.62e-24 * | 8.169 |
| Q1-memory-effect-pso | peak_joint_velocity_rad | pso-cold-static | pso-injection-static | 6.14 (0.32) | 0.5979 (1.06) | 27.507 | 5.89e-25 | 1.47e-24 * | 7.102 |
| Q1-memory-effect-pso | peak_joint_acceleration_rad | pso-cold-static | pso-injection-static | 10.42 (1.35) | 0.5793 (1.06) | 31.354 | 9.10e-37 | 4.55e-36 * | 8.096 |

## Metric family: computation

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | wall_time_s | pso-cold-static | pso-injection-static | 20.03 (0.881) | 18.97 (0.213) | 6.373 | 3.51e-07 | 3.51e-07 * | 1.646 |

## Metric family: effort

### Rank-based planned comparisons (primary — heavy-tailed family)

| question | metric | A | B | median A (IQR) | median B (IQR) | U | p raw | p adj (FDR) | Cliff's delta |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-injection-static | 146 (71.8) | 65 (99.5) | 607.5 | 1.43e-02 | 1.43e-02 * | 0.350 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-injection-static | 106 (108) | 30 (132) | 621.0 | 9.96e-03 | 1.43e-02 * | 0.380 |

Kruskal-Wallis omnibus across the PSO memory triple (cold / injection / persistence, static):

- 0.001: H = 6.037, p = 1.40e-02
- 0.01: H = 6.681, p = 9.75e-03

### Welch t (secondary check)

| question | metric | A | B | mean A (sd) | mean B (sd) | t | p raw | p adj (FDR) | Cohen's d |
|---|---|---|---|---|---|---|---|---|---|
| Q1-memory-effect-pso | 0.001 | pso-cold-static | pso-injection-static | 129 (34.7) | 104 (49.9) | 2.256 | 2.83e-02 | 5.66e-02 | 0.583 |
| Q1-memory-effect-pso | 0.01 | pso-cold-static | pso-injection-static | 103.3 (49.1) | 79.03 (63.7) | 1.653 | 1.04e-01 | 1.04e-01 | 0.427 |

Effect-size guides: Cohen's d — |d| < 0.2 trivial, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large. Cliff's delta — |delta| < 0.147 negligible, 0.147-0.33 small, 0.33-0.474 medium, > 0.474 large. `*` = significant at FDR-adjusted p < 0.05 (one BH pass per family per track).
