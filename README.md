# bimo-ik

Comparative study of bio-inspired metaheuristics (PSO and GA) applied to **full-pose inverse
kinematics** of the BIMO 6-DoF robotic arm, redesigned after the CBA 2026 reviews so that
**population memory is an explicit, isolated experimental factor** rather than a hidden advantage
of one method.

## Design in one paragraph

A single config-driven engine runs every experimental *cell* of an intentionally unbalanced
factorial: algorithm {PSO, GA} x memory {cold, injection, persistence (PSO only)} x landscape
{static weights, adaptive weight schedule} — 10 cells — over three trajectories (line, arc,
sharp corner) whose targets are full poses (position + SLERP-interpolated orientation), each pose
verified reachable at generation time. Fitness is dimensionless (position error normalized by the
arm's reach computed from the DH table, orientation error by pi, joint continuity by the joint
range). Every point runs to a fixed iteration cap; accuracy-vs-effort is reported as an
effort-vs-epsilon curve with right-censoring, derived post hoc from per-iteration histories.
Statistics use planned pairwise comparisons (Welch t + Benjamini-Hochberg FDR + Cohen's d), never
a full-crossed ANOVA on cells that do not exist.

## Layout

- `src/bimo_ik/kinematics/` — DH model, forward kinematics, SO(3) utilities, characteristic scales
- `src/bimo_ik/trajectories/` — trajectory generators + reachability guard
- `src/bimo_ik/fitness/` — dimensionless fitness factory (static & scheduled weights)
- `src/bimo_ik/algorithms/` — PSO (cold/injection/persistence) and GA (cold/injection)
- `src/bimo_ik/experiment/` — cell matrix, runner, parallel engine, run manifest
- `src/bimo_ik/metrics/` — accuracy, kinematic executability, effort-vs-epsilon with censoring
- `src/bimo_ik/stats/` — planned comparisons, FDR, effect sizes
- `src/bimo_ik/plotting/` — figures
- `src/bimo_ik/config/` — schema + `config_lars` (conference) and `config_full` (journal) presets
- `reference/` — frozen copy of the prior study's canonical code (provenance only, never imported)
- `docs/` — methodology draft and reviewer-response table

## Usage

```bash
# environment
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]

# tests
.venv/Scripts/python -m pytest

# quick end-to-end smoke (small N)
.venv/Scripts/python scripts/run_experiment.py --config lars --smoke

# real runs
.venv/Scripts/python scripts/run_experiment.py --config lars
.venv/Scripts/python scripts/run_experiment.py --config full

# reproduction anchor vs. the old paper
.venv/Scripts/python scripts/sanity_check.py

# figures and statistics from a results directory
.venv/Scripts/python scripts/make_figures.py --results results/lars
.venv/Scripts/python scripts/run_stats.py --results results/lars
```

Every results directory contains a `manifest.json` with the config hash, seed scheme, library
versions, and git revision — a run is reproducible from its manifest alone.
