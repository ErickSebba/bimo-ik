# reference/ — provenance snapshots (never imported)

Frozen copies of the prior study's code, kept so this repository is self-contained evidence of
what the rebuild is based on. **Nothing in `src/` imports from here.**

- `main.py` — the canonical reference implementation (v4.0 lineage) whose numbers appeared in the
  rejected CBA 2026 submission. Ground truth for: DH parameters, joint limits, forward
  kinematics, PSO hyperparameters (pop 40, w=0.7, c1=c2=1.5, v_max = 20% of range, 80
  iterations/point in the old protocol), GA operators (elitism 2, roulette, arithmetic crossover
  0.7, Gaussian mutation 0.2 / sigma 0.1, 20% renewal), and the genuine persistent-swarm
  PSO-DFL semantics (`PSODinamico.atualizar_landscape`: keep X, V, pbest, gbest; re-evaluate
  memories on landscape change).
- `plot.py` — the plotting utilities whose figure logic `src/bimo_ik/plotting/` adapts.

Encoding note: these copies were materialized from conversation attachments whose Portuguese
comments arrived with mojibake (UTF-8 read as Latin-1). Comments and console strings were
re-encoded to readable text when copying; **all executable logic is transcribed unchanged**.

Known deviations of the new codebase from `main.py` (deliberate, documented in
`docs/methodology.md`):

1. Full-pose targets (position + orientation); `main.py` discarded `T[:3,:3]`.
2. Dimensionless fitness with characteristic-scale normalization; `main.py` summed millimetres
   with a scaled radian term.
3. The PSO-DFL adaptive weight schedule is now an explicit ablation factor (landscape mode), not
   a hidden difference between "the same" fitness functions.
4. Cold/injection/persistence initialization is an explicit factor for both algorithms
   (persistence is PSO-only by design).
5. No stochastic noise term exists in any fitness (an earlier, pre-`main.py` lineage had one;
   it must never reappear).
