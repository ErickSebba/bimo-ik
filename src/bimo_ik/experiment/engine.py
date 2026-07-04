"""Experiment orchestration: reachability guard, manifest, parallel runs.

Results are one compressed .npz per (trajectory, cell, seed) under
    <output_dir>/<trajectory>/<cell_id>/run_<seed>.npz
plus a manifest.json at the root. Because each run owns a decorrelated,
fully determined RNG stream (see runner.run_rng), serial and parallel
execution produce bit-identical files, and interrupted experiments resume
by skipping existing files.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np

from ..config.schema import ExperimentConfig
from ..trajectories.generators import build_trajectory
from ..trajectories.reachability import validate_trajectory
from ..factors import Cell
from .manifest import write_manifest
from .runner import RunResult, run_single


def result_path(output_dir: Path, trajectory_name: str, cell_id: str, seed: int) -> Path:
    return Path(output_dir) / trajectory_name / cell_id / f"run_{seed:04d}.npz"


def save_result(path: Path, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cell_id=np.str_(result.cell_id),
        trajectory_name=np.str_(result.trajectory_name),
        seed=np.int64(result.seed),
        q_path=result.q_path,
        fitness_hist=result.fitness_hist,
        pose_error_hist=result.pose_error_hist,
        diversity_hist=result.diversity_hist,
        evals_hist=result.evals_hist,
        best_fitness=result.best_fitness,
        best_pose_error=result.best_pose_error,
        wall_time_s=np.float64(result.wall_time_s),
    )


def load_result(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _worker(task: tuple) -> RunResult:
    config, cell, spec, seed = task
    return run_single(config, cell, spec, seed)


def run_experiment(
    config: ExperimentConfig,
    output_dir: str | Path,
    n_jobs: int = 1,
    skip_existing: bool = True,
    verbose: bool = True,
) -> list[Path]:
    output_dir = Path(output_dir)

    # Reachability guard: no experiment starts against uncertified poses.
    for spec in config.trajectories:
        validate_trajectory(build_trajectory(spec), tol=config.reach_tolerance)
        if verbose:
            print(f"[guard] trajectory {spec.name!r}: all {spec.n_points} poses certified")

    write_manifest(config, output_dir)

    tasks: list[tuple[ExperimentConfig, Cell, object, int]] = []
    written: list[Path] = []
    for spec in config.trajectories:
        for cell in config.cells:
            for seed in config.run_seeds:
                path = result_path(output_dir, spec.name, cell.cell_id, seed)
                if skip_existing and path.exists():
                    if verbose:
                        print(f"[skip] {path}")
                    continue
                tasks.append((config, cell, spec, seed))

    total = len(tasks)
    if verbose:
        print(f"[engine] {total} run(s) to execute, n_jobs={n_jobs}")

    def _store(result: RunResult, index: int) -> None:
        path = result_path(output_dir, result.trajectory_name, result.cell_id, result.seed)
        save_result(path, result)
        written.append(path)
        if verbose:
            print(
                f"[{index}/{total}] {result.trajectory_name}/{result.cell_id}"
                f"/seed {result.seed}: {result.wall_time_s:.1f}s"
            )

    if n_jobs <= 1:
        for i, task in enumerate(tasks, start=1):
            _store(_worker(task), i)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, tasks), start=1):
                _store(result, i)

    return written
