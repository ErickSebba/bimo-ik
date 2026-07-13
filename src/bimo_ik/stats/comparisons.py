"""Planned comparisons over the intentionally unbalanced cell matrix.

The design has 10 cells but is NOT a full 2x3x2 factorial (GA has no
persistence level), so no full-crossed ANOVA is fitted. Instead, three
families of planned pairwise questions are tested explicitly:

  Q1 — memory effect within PSO (static landscape):
       cold vs injection vs persistence (3 pairs).
  Q2 — algorithm effect, memory-matched (static landscape):
       PSO-cold vs GA-cold; PSO-injection vs GA-injection.
  Q3 — landscape effect within each algorithm x memory combination:
       static vs adaptive (5 pairs).

Two inference tracks share the same planned pairs and the same one-BH-FDR-
pass-per-family policy:

- Welch t-tests + pooled-SD Cohen's d — primary for well-behaved families
  (kinematic executability, computation).
- Mann-Whitney U + Cliff's delta with median/IQR summaries — primary for the
  heavy-tailed families (accuracy, effort), where sd frequently exceeds the
  mean and mean+-sd/pooled-d are misleading. Kruskal-Wallis over the PSO
  memory triple accompanies those families as an omnibus check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

_Q1 = [
    ("pso-cold-static", "pso-injection-static"),
    ("pso-cold-static", "pso-persistence-static"),
    ("pso-injection-static", "pso-persistence-static"),
]
_Q2 = [
    ("pso-cold-static", "ga-cold-static"),
    ("pso-injection-static", "ga-injection-static"),
]
_Q3_COMBOS = [
    "pso-cold", "pso-injection", "pso-persistence", "ga-cold", "ga-injection",
]


@dataclass(frozen=True)
class Comparison:
    question: str
    metric: str
    cell_a: str
    cell_b: str
    n_a: int
    n_b: int
    mean_a: float
    sd_a: float
    mean_b: float
    sd_b: float
    t_stat: float
    p_raw: float
    cohens_d: float
    p_adj: float | None = None  # filled by adjust_family


@dataclass(frozen=True)
class RankComparison:
    """Rank-based planned pair: Mann-Whitney U + Cliff's delta, median/IQR."""

    question: str
    metric: str
    cell_a: str
    cell_b: str
    n_a: int
    n_b: int
    median_a: float
    iqr_a: float
    median_b: float
    iqr_b: float
    u_stat: float
    p_raw: float
    cliffs_delta: float
    p_adj: float | None = None  # filled by adjust_family


def planned_pairs(available_cells: set[str]) -> list[tuple[str, str, str]]:
    """(question, cell_a, cell_b) for every planned pair present in a config."""
    pairs: list[tuple[str, str, str]] = []
    for a, b in _Q1:
        if a in available_cells and b in available_cells:
            pairs.append(("Q1-memory-effect-pso", a, b))
    for a, b in _Q2:
        if a in available_cells and b in available_cells:
            pairs.append(("Q2-algorithm-effect-memory-matched", a, b))
    for combo in _Q3_COMBOS:
        a, b = f"{combo}-static", f"{combo}-adaptive"
        if a in available_cells and b in available_cells:
            pairs.append(("Q3-landscape-effect", a, b))
    return pairs


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation (ddof=1)."""
    nx, ny = len(x), len(y)
    pooled = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    return float((np.mean(x) - np.mean(y)) / (pooled + 1e-12))


def run_planned_comparisons(
    values_by_cell: dict[str, np.ndarray], metric: str
) -> list[Comparison]:
    comparisons = []
    for question, a, b in planned_pairs(set(values_by_cell)):
        x, y = np.asarray(values_by_cell[a]), np.asarray(values_by_cell[b])
        t_stat, p_raw = sps.ttest_ind(x, y, equal_var=False)  # Welch
        comparisons.append(
            Comparison(
                question=question,
                metric=metric,
                cell_a=a,
                cell_b=b,
                n_a=len(x),
                n_b=len(y),
                mean_a=float(np.mean(x)),
                sd_a=float(np.std(x, ddof=1)),
                mean_b=float(np.mean(y)),
                sd_b=float(np.std(y, ddof=1)),
                t_stat=float(t_stat),
                p_raw=float(p_raw),
                cohens_d=cohens_d(x, y),
            )
        )
    return comparisons


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x > y) - P(x < y) over all cross-pairs, in [-1, 1].

    Rank-based effect size (equivalent to 2U/(n1*n2) - 1); robust to the
    heavy tails that make pooled-SD Cohen's d unpredictable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y)))


def run_rank_comparisons(
    values_by_cell: dict[str, np.ndarray], metric: str
) -> list[RankComparison]:
    """Mann-Whitney U over the planned pairs, with Cliff's delta.

    Degenerate pairs where every value in both groups is identical (e.g.
    both cells fully right-censored at the iteration cap) short-circuit to
    p = 1.0, delta = 0 — "no detectable difference", not NaN.
    """
    comparisons = []
    for question, a, b in planned_pairs(set(values_by_cell)):
        x = np.asarray(values_by_cell[a], dtype=float)
        y = np.asarray(values_by_cell[b], dtype=float)
        if np.ptp(np.concatenate([x, y])) == 0.0:
            u_stat, p_raw, delta = len(x) * len(y) / 2.0, 1.0, 0.0
        else:
            u_stat, p_raw = sps.mannwhitneyu(
                x, y, alternative="two-sided", method="asymptotic"
            )
            delta = cliffs_delta(x, y)
        comparisons.append(
            RankComparison(
                question=question,
                metric=metric,
                cell_a=a,
                cell_b=b,
                n_a=len(x),
                n_b=len(y),
                median_a=float(np.median(x)),
                iqr_a=float(np.percentile(x, 75) - np.percentile(x, 25)),
                median_b=float(np.median(y)),
                iqr_b=float(np.percentile(y, 75) - np.percentile(y, 25)),
                u_stat=float(u_stat),
                p_raw=float(p_raw),
                cliffs_delta=float(delta),
            )
        )
    return comparisons


def adjust_family(comparisons: list) -> list:
    """One Benjamini-Hochberg FDR pass over a whole metric family.

    Works for both Comparison and RankComparison (anything with p_raw/p_adj).
    """
    if not comparisons:
        return []
    p_raw = [c.p_raw for c in comparisons]
    _, p_adj, _, _ = multipletests(p_raw, alpha=0.05, method="fdr_bh")
    return [replace(c, p_adj=float(p)) for c, p in zip(comparisons, p_adj)]


def kruskal_check(values_by_cell: dict[str, np.ndarray], cells: list[str]):
    """Distribution-free omnibus companion (H statistic, p-value)."""
    groups = [np.asarray(values_by_cell[c]) for c in cells if c in values_by_cell]
    if len(groups) < 2:
        return None
    if np.ptp(np.concatenate(groups)) == 0.0:  # all values identical
        return 0.0, 1.0
    h, p = sps.kruskal(*groups)
    return float(h), float(p)


def _sig(c) -> str:
    return " *" if (c.p_adj is not None and c.p_adj < 0.05) else ""


def welch_table_lines(comparisons: list[Comparison]) -> list[str]:
    lines = [
        "| question | metric | A | B | mean A (sd) | mean B (sd) "
        "| t | p raw | p adj (FDR) | Cohen's d |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c.question} | {c.metric} | {c.cell_a} | {c.cell_b} "
            f"| {c.mean_a:.4g} ({c.sd_a:.3g}) | {c.mean_b:.4g} ({c.sd_b:.3g}) "
            f"| {c.t_stat:.3f} | {c.p_raw:.2e} | {c.p_adj:.2e}{_sig(c)} "
            f"| {c.cohens_d:.3f} |"
        )
    return lines


def rank_table_lines(comparisons: list[RankComparison]) -> list[str]:
    lines = [
        "| question | metric | A | B | median A (IQR) | median B (IQR) "
        "| U | p raw | p adj (FDR) | Cliff's delta |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c.question} | {c.metric} | {c.cell_a} | {c.cell_b} "
            f"| {c.median_a:.4g} ({c.iqr_a:.3g}) | {c.median_b:.4g} ({c.iqr_b:.3g}) "
            f"| {c.u_stat:.1f} | {c.p_raw:.2e} | {c.p_adj:.2e}{_sig(c)} "
            f"| {c.cliffs_delta:.3f} |"
        )
    return lines


def format_report(
    families: dict[str, list[Comparison]],
    title: str,
) -> str:
    """Markdown report of Welch planned comparisons (legacy single-track form)."""
    lines = [f"# {title}", ""]
    for family, comparisons in families.items():
        lines.append(f"## Metric family: {family}")
        lines.append("")
        lines += welch_table_lines(comparisons)
        lines.append("")
    lines.append(
        "Cohen's d guide: |d| < 0.2 trivial, 0.2-0.5 small, 0.5-0.8 medium, "
        "> 0.8 large. `*` = significant at FDR-adjusted p < 0.05."
    )
    lines.append("")
    return "\n".join(lines)
