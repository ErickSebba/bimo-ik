# Ablation w_j = 0 vs baseline w_j = 0.1 — trajectory 'line' (N=30 runs/cell)

Median [Q1, Q3]. Mann-Whitney U two-sided; Cliff's delta = P(w_j=0 > w_j=0.1) - P(<).

## pso-cold-static

| metric | w_j = 0.1 (lars) | w_j = 0 (ablation) | p (MWU) | Cliff's δ |
|---|---|---|---|---|
| accuracy:rmse_pos_mm | 61.04 [54.37, 68.21] | 65.18 [58.19, 74.65] | 2.71e-01 | +0.167 |
| accuracy:rmse_ori_rad | 0.04012 [0.02754, 0.05364] | 0.04998 [0.03409, 0.08499] | 1.62e-01 | +0.211 |
| accuracy:max_pos_mm | 165.4 [145.6, 183] | 177.4 [156.3, 184.9] | 4.38e-01 | +0.118 |
| kinematic:roughness_rad | 249.8 [211, 269.6] | 256.3 [224.1, 290.1] | 1.86e-01 | +0.200 |
| kinematic:roughness_index | 8.613 [7.278, 9.296] | 8.837 [7.727, 10] | 1.86e-01 | +0.200 |
| kinematic:jerk_rad | 445 [372.7, 472.7] | 464.3 [389.5, 511.5] | 1.41e-01 | +0.222 |
| kinematic:peak_joint_velocity_rad | 6.193 [6.046, 6.283] | 6.283 [6.114, 6.283] | 1.23e-01 | +0.221 |
| effort_iterations:0.001 | 115.2 [83.62, 160] | 146 [88.25, 160] | 3.95e-01 | +0.122 |
| effort_iterations:0.01 | 72.5 [50.12, 160] | 106 [52, 160] | 7.25e-01 | +0.052 |
| wall_time_s:wall_time_s | 10.27 [10.05, 10.37] | 19.9 [19.32, 20.3] | 3.02e-11 | +1.000 |

## pso-injection-static

| metric | w_j = 0.1 (lars) | w_j = 0 (ablation) | p (MWU) | Cliff's δ |
|---|---|---|---|---|
| accuracy:rmse_pos_mm | 0.5981 [0.000728, 31.52] | 13.55 [0.0005639, 40.83] | 8.65e-01 | -0.027 |
| accuracy:rmse_ori_rad | 4.286e-06 [2.258e-06, 0.01243] | 7.45e-05 [2.318e-06, 0.01144] | 9.00e-01 | +0.020 |
| accuracy:max_pos_mm | 3.134 [0.002571, 69.4] | 44.68 [0.001466, 127] | 7.51e-01 | -0.049 |
| kinematic:roughness_rad | 3.421 [3.421, 3.621] | 3.421 [3.421, 4.576] | 2.01e-01 | +0.193 |
| kinematic:roughness_index | 0.118 [0.118, 0.1248] | 0.118 [0.118, 0.1578] | 2.01e-01 | +0.193 |
| kinematic:jerk_rad | 0.2878 [0.2182, 2.312] | 0.624 [0.2182, 5.582] | 2.46e-01 | +0.176 |
| kinematic:peak_joint_velocity_rad | 0.04805 [0.04804, 0.1359] | 0.0713 [0.04804, 0.4679] | 2.34e-01 | +0.180 |
| effort_iterations:0.001 | 62 [60, 160] | 65 [60.5, 160] | 4.99e-01 | +0.099 |
| effort_iterations:0.01 | 29.25 [27.12, 160] | 30 [28, 160] | 5.02e-01 | +0.100 |
| wall_time_s:wall_time_s | 10.55 [10.47, 10.67] | 18.95 [18.87, 19.09] | 3.02e-11 | +1.000 |

## cold vs injection within each weight setting (roughness_rad)

| setting | cold | injection | p (MWU) | Cliff's δ (cold − inj) |
|---|---|---|---|---|
| w_j = 0.1 | 249.8 [211, 269.6] | 3.421 [3.421, 3.621] | 3.02e-11 | +1.000 |
| w_j = 0 | 256.3 [224.1, 290.1] | 3.421 [3.421, 4.576] | 3.02e-11 | +1.000 |
