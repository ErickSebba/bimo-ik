"""Planned comparisons: pair selection, Welch/effect sizes, FDR, report."""

import numpy as np
import pytest

from bimo_ik.stats import (
    adjust_family,
    cohens_d,
    format_report,
    planned_pairs,
    run_planned_comparisons,
)


def test_planned_pairs_full_matrix():
    from bimo_ik.factors import full_matrix

    cells = {c.cell_id for c in full_matrix()}
    pairs = planned_pairs(cells)
    questions = [q for q, _, _ in pairs]
    assert questions.count("Q1-memory-effect-pso") == 3
    assert questions.count("Q2-algorithm-effect-memory-matched") == 2
    assert questions.count("Q3-landscape-effect") == 5
    assert len(pairs) == 10


def test_planned_pairs_filters_to_available_cells():
    # LARS preset: static-only, 5 cells -> Q3 disappears entirely
    cells = {
        "pso-cold-static", "pso-injection-static", "pso-persistence-static",
        "ga-cold-static", "ga-injection-static",
    }
    pairs = planned_pairs(cells)
    assert len(pairs) == 5
    assert all(q != "Q3-landscape-effect" for q, _, _ in pairs)


def test_cohens_d_known_value():
    x = np.array([2.0, 4.0, 6.0])   # mean 4, var 4
    y = np.array([1.0, 3.0, 5.0])   # mean 3, var 4
    assert cohens_d(x, y) == pytest.approx(0.5)
    assert cohens_d(y, x) == pytest.approx(-0.5)


def test_run_planned_comparisons_detects_a_real_difference():
    rng = np.random.default_rng(0)
    values = {
        "pso-cold-static": rng.normal(10.0, 1.0, 30),
        "pso-persistence-static": rng.normal(2.0, 1.0, 30),
        "pso-injection-static": rng.normal(10.0, 1.0, 30),
    }
    comparisons = adjust_family(run_planned_comparisons(values, "rmse_pos_mm"))
    by_pair = {(c.cell_a, c.cell_b): c for c in comparisons}
    strong = by_pair[("pso-cold-static", "pso-persistence-static")]
    null = by_pair[("pso-cold-static", "pso-injection-static")]
    assert strong.p_adj < 0.001 and abs(strong.cohens_d) > 2
    assert null.p_adj > 0.05


def test_fdr_adjustment_never_decreases_p():
    rng = np.random.default_rng(1)
    values = {
        "pso-cold-static": rng.normal(0, 1, 20),
        "pso-injection-static": rng.normal(0, 1, 20),
        "pso-persistence-static": rng.normal(0, 1, 20),
        "ga-cold-static": rng.normal(0, 1, 20),
        "ga-injection-static": rng.normal(0, 1, 20),
    }
    comparisons = adjust_family(run_planned_comparisons(values, "m"))
    assert all(c.p_adj >= c.p_raw - 1e-12 for c in comparisons)
    assert all(c.p_adj <= 1.0 for c in comparisons)


def test_format_report_smoke():
    rng = np.random.default_rng(2)
    values = {
        "pso-cold-static": rng.normal(5, 1, 10),
        "ga-cold-static": rng.normal(5, 1, 10),
    }
    fam = {"accuracy": adjust_family(run_planned_comparisons(values, "rmse_pos_mm"))}
    report = format_report(fam, "unit test")
    assert "Q2-algorithm-effect-memory-matched" in report
    assert "Cohen's d" in report
