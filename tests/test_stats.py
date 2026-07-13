"""Planned comparisons: pair selection, Welch/rank tracks, FDR, report."""

import numpy as np
import pytest

from bimo_ik.stats import (
    adjust_family,
    cliffs_delta,
    cohens_d,
    format_report,
    kruskal_check,
    planned_pairs,
    rank_table_lines,
    run_planned_comparisons,
    run_rank_comparisons,
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


# --- rank-based track (heavy-tailed families) ---------------------------------

def test_cliffs_delta_known_values():
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    # half the cross-pairs favor x: delta = 0.5
    assert cliffs_delta([2, 4], [1, 3]) == pytest.approx(0.5)


def test_rank_comparisons_detect_location_shift():
    rng = np.random.default_rng(3)
    values = {
        "pso-injection-static": rng.lognormal(0.0, 1.0, 30),        # heavy-tailed
        "ga-injection-static": rng.lognormal(2.0, 1.0, 30),         # shifted
    }
    comparisons = adjust_family(run_rank_comparisons(values, "rmse_pos_mm"))
    (c,) = comparisons
    assert c.p_adj < 1e-4
    assert c.cliffs_delta < -0.6  # A well below B in rank terms
    assert c.median_a < c.median_b


def test_rank_comparisons_degenerate_all_ties():
    # both cells fully censored at the cap: constant, identical samples
    values = {
        "pso-injection-static": np.full(30, 160.0),
        "ga-injection-static": np.full(30, 160.0),
    }
    (c,) = run_rank_comparisons(values, "0.001")
    assert c.p_raw == 1.0 and c.cliffs_delta == 0.0
    assert not np.isnan(c.p_raw)


def test_rank_comparison_median_iqr_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # heavy tail
    y = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    values = {"pso-cold-static": x, "ga-cold-static": y}
    (c,) = run_rank_comparisons(values, "m")
    assert c.median_a == pytest.approx(np.median(x))
    assert c.iqr_a == pytest.approx(np.percentile(x, 75) - np.percentile(x, 25))
    assert c.median_b == pytest.approx(np.median(y))


def test_kruskal_check_degenerate_and_signal():
    const = {"pso-cold-static": np.ones(10), "pso-injection-static": np.ones(10)}
    assert kruskal_check(const, list(const)) == (0.0, 1.0)
    rng = np.random.default_rng(4)
    shifted = {
        "pso-cold-static": rng.normal(0, 1, 30),
        "pso-injection-static": rng.normal(5, 1, 30),
    }
    h, p = kruskal_check(shifted, list(shifted))
    assert p < 1e-6


def test_rank_table_renders():
    rng = np.random.default_rng(5)
    values = {
        "pso-injection-static": rng.normal(5, 1, 10),
        "ga-injection-static": rng.normal(5, 1, 10),
    }
    lines = rank_table_lines(adjust_family(run_rank_comparisons(values, "m")))
    assert "Cliff's delta" in lines[0]
    assert len(lines) == 3  # header + separator + one pair


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
