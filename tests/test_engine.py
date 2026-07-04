"""Engine + config: determinism (serial == parallel), resume, config hashing."""

import numpy as np
import pytest

from bimo_ik.algorithms import GAParams, PSOParams
from bimo_ik.config import ExperimentConfig, config_hash
from bimo_ik.config.config_full import get_config as get_full
from bimo_ik.config.config_lars import get_config as get_lars
from bimo_ik.experiment import Cell, full_matrix, run_single
from bimo_ik.experiment.engine import load_result, result_path, run_experiment
from bimo_ik.trajectories import TrajectorySpec


def tiny_config(name="tiny"):
    spec = TrajectorySpec(
        "line", "line", 5, (150.0, 100.0, 200.0), (300.0, -100.0, 200.0)
    )
    return ExperimentConfig(
        name=name,
        cells=(
            Cell("pso", "cold", "static"),
            Cell("pso", "persistence", "adaptive"),
            Cell("ga", "injection", "static"),
        ),
        trajectories=(spec,),
        n_runs=2,
        seed_base=42,
        max_iter=5,
        pso=PSOParams(pop=8),
        ga=GAParams(pop=8),
    )


def test_full_matrix_has_ten_unique_cells():
    cells = full_matrix()
    assert len(cells) == 10
    assert len({c.cell_id for c in cells}) == 10
    assert sum(c.memory.value == "persistence" for c in cells) == 2  # PSO only


def test_presets_are_valid_and_distinct():
    lars, full = get_lars(), get_full()
    assert len(lars.cells) == 5 and len(lars.trajectories) == 2
    assert len(full.cells) == 10 and len(full.trajectories) == 3
    assert config_hash(lars) != config_hash(full)


def test_config_hash_is_stable_and_sensitive():
    assert config_hash(tiny_config()) == config_hash(tiny_config())
    changed = tiny_config()
    changed = ExperimentConfig(**{**changed.__dict__, "max_iter": 6})
    assert config_hash(changed) != config_hash(tiny_config())


def test_config_rejects_duplicate_cells():
    spec = TrajectorySpec("line", "line", 5, (150, 100, 200), (300, -100, 200))
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="dup",
            cells=(Cell("pso", "cold", "static"), Cell("pso", "cold", "static")),
            trajectories=(spec,),
            n_runs=1,
            seed_base=42,
            max_iter=5,
        )


def test_run_single_is_deterministic():
    cfg = tiny_config()
    cell, spec = cfg.cells[0], cfg.trajectories[0]
    a = run_single(cfg, cell, spec, seed=42)
    b = run_single(cfg, cell, spec, seed=42)
    np.testing.assert_array_equal(a.q_path, b.q_path)
    np.testing.assert_array_equal(a.pose_error_hist, b.pose_error_hist)
    c = run_single(cfg, cell, spec, seed=43)
    assert not np.array_equal(a.q_path, c.q_path)


def test_serial_and_parallel_runs_are_identical(tmp_path):
    cfg = tiny_config()
    dir_serial = tmp_path / "serial"
    dir_parallel = tmp_path / "parallel"
    paths_serial = run_experiment(cfg, dir_serial, n_jobs=1, verbose=False)
    paths_parallel = run_experiment(cfg, dir_parallel, n_jobs=2, verbose=False)
    assert len(paths_serial) == len(paths_parallel) == 3 * 2  # cells x seeds

    for spec in cfg.trajectories:
        for cell in cfg.cells:
            for seed in cfg.run_seeds:
                a = load_result(result_path(dir_serial, spec.name, cell.cell_id, seed))
                b = load_result(result_path(dir_parallel, spec.name, cell.cell_id, seed))
                for key in ("q_path", "fitness_hist", "pose_error_hist", "evals_hist"):
                    np.testing.assert_array_equal(a[key], b[key])


def test_resume_skips_existing_results(tmp_path):
    cfg = tiny_config()
    out = tmp_path / "results"
    first = run_experiment(cfg, out, n_jobs=1, verbose=False)
    assert len(first) == 6
    second = run_experiment(cfg, out, n_jobs=1, verbose=False)
    assert second == []  # everything skipped


def test_manifest_written_with_hash(tmp_path):
    import json

    cfg = tiny_config()
    run_experiment(cfg, tmp_path / "r", n_jobs=1, verbose=False)
    manifest = json.loads((tmp_path / "r" / "manifest.json").read_text())
    assert manifest["config_hash"] == config_hash(cfg)
    assert manifest["run_seeds"] == [42, 43]
    assert "numpy" in manifest["library_versions"]


def test_seed_streams_are_decorrelated_across_cells():
    cfg = tiny_config()
    spec = cfg.trajectories[0]
    a = run_single(cfg, cfg.cells[0], spec, seed=42)  # pso-cold-static
    b = run_single(cfg, cfg.cells[1], spec, seed=42)  # pso-persistence-adaptive
    # same seed, different cells: different randomness (not just different policy)
    assert not np.array_equal(a.q_path, b.q_path)
