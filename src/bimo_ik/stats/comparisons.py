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
from statsmodels.stats.weightstats import ttost_ind

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


# --- equivalence testing (TOST) + bootstrap CIs -------------------------------
#
# A non-significant difference is NOT evidence of equivalence (it is equally
# consistent with low power). To positively claim "PSO and GA are equivalent
# under matched injection memory," TOST is used: equivalence within +-bound is
# established when BOTH one-sided Welch tests reject at alpha = 0.05, i.e.
# p_tost = max(p_lower, p_upper) < 0.05.
#
# Bounds are never absolute magic numbers: they are fractions of the
# cold->injection memory effect (the effect the study claims is real),
# reported at three anchors as a sensitivity analysis — min(PSO, GA effects)
# is the headline (most conservative), plus the mean of the two and the
# PSO-only effect.

TOST_ANCHORS = ("min", "mean", "pso")


@dataclass(frozen=True)
class EquivalenceRow:
    anchor: str        # one of TOST_ANCHORS
    fraction: float    # bound as a fraction of the anchor value
    bound: float       # absolute half-width of the equivalence interval
    p_tost: float      # max of the two one-sided p-values
    p_lower: float
    p_upper: float
    equivalent: bool   # p_tost < 0.05


@dataclass(frozen=True)
class EquivalenceResult:
    metric: str
    cell_a: str
    cell_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    diff_mean: float               # mean_a - mean_b
    ci_diff: tuple[float, float]   # 95% BCa bootstrap CI on the difference
    cohens_d: float
    ci_d: tuple[float, float]      # 95% BCa bootstrap CI on Cohen's d
    anchors: dict[str, float]
    rows: tuple[EquivalenceRow, ...]
    bootstrap_seed: int
    n_resamples: int


def tost_equivalence(
    x: np.ndarray, y: np.ndarray, low: float, high: float
) -> tuple[float, float, float]:
    """TOST for mean equivalence within [low, high], Welch (unequal variances).

    Returns (p_tost, p_lower, p_upper): p_tost = max of the two one-sided
    Welch p-values (verified in tests against manually composed one-sided
    Welch tests).
    """
    p, res_low, res_high = ttost_ind(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float),
        low, high, usevar="unequal",
    )
    return float(p), float(res_low[1]), float(res_high[1])


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    statistic,
    n_resamples: int = 10_000,
    seed: int = 20260704,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """BCa bootstrap CI for a two-sample statistic(x, y) (fixed, recorded seed)."""
    result = sps.bootstrap(
        (np.asarray(x, dtype=float), np.asarray(y, dtype=float)),
        statistic=statistic,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="BCa",
        vectorized=False,
        paired=False,
        rng=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def memory_effect_anchors(
    values_by_cell: dict[str, np.ndarray]
) -> dict[str, float] | None:
    """The three cold->injection memory-effect anchors for TOST bounds.

    min  — min(|PSO effect|, |GA effect|): headline, most conservative;
    mean — average of the two effects;
    pso  — the PSO-only effect (the paper's headline memory effect).
    Returns None if any of the four static cold/injection cells is missing.
    """
    needed = ("pso-cold-static", "pso-injection-static",
              "ga-cold-static", "ga-injection-static")
    if any(cell not in values_by_cell for cell in needed):
        return None
    pso = abs(float(np.mean(values_by_cell["pso-cold-static"]))
              - float(np.mean(values_by_cell["pso-injection-static"])))
    ga = abs(float(np.mean(values_by_cell["ga-cold-static"]))
             - float(np.mean(values_by_cell["ga-injection-static"])))
    return {"min": min(pso, ga), "mean": (pso + ga) / 2.0, "pso": pso}


def equivalence_analysis(
    values_by_cell: dict[str, np.ndarray],
    metric: str,
    cell_a: str = "pso-injection-static",
    cell_b: str = "ga-injection-static",
    fractions: tuple[float, ...] = (0.10, 0.20),
    n_resamples: int = 10_000,
    seed: int = 20260704,
) -> EquivalenceResult | None:
    """TOST sensitivity table (anchors x fractions) + bootstrap CIs for one pair."""
    anchors = memory_effect_anchors(values_by_cell)
    if anchors is None or cell_a not in values_by_cell or cell_b not in values_by_cell:
        return None
    x = np.asarray(values_by_cell[cell_a], dtype=float)
    y = np.asarray(values_by_cell[cell_b], dtype=float)

    rows = []
    for anchor in TOST_ANCHORS:
        for fraction in fractions:
            bound = fraction * anchors[anchor]
            p, p_lo, p_hi = tost_equivalence(x, y, -bound, bound)
            rows.append(EquivalenceRow(
                anchor=anchor, fraction=fraction, bound=bound,
                p_tost=p, p_lower=p_lo, p_upper=p_hi, equivalent=p < 0.05,
            ))

    ci_diff = bootstrap_ci(
        x, y, lambda a, b: float(np.mean(a) - np.mean(b)),
        n_resamples=n_resamples, seed=seed,
    )
    ci_d = bootstrap_ci(x, y, cohens_d, n_resamples=n_resamples, seed=seed)

    return EquivalenceResult(
        metric=metric, cell_a=cell_a, cell_b=cell_b, n_a=len(x), n_b=len(y),
        mean_a=float(np.mean(x)), mean_b=float(np.mean(y)),
        diff_mean=float(np.mean(x) - np.mean(y)),
        ci_diff=ci_diff, cohens_d=cohens_d(x, y), ci_d=ci_d,
        anchors=anchors, rows=tuple(rows),
        bootstrap_seed=seed, n_resamples=n_resamples,
    )


def equivalence_section_lines(results: list[EquivalenceResult]) -> list[str]:
    """Markdown for the TOST section: CI table + anchor-sensitivity table."""
    if not results:
        return []
    first = results[0]
    lines = [
        f"## Equivalence (TOST): {first.cell_a} vs {first.cell_b}",
        "",
        "TOST establishes equivalence within +-bound when BOTH one-sided Welch",
        "tests reject (p_tost = max of the two one-sided p < 0.05). Bounds are",
        "fractions of the cold->injection memory effect, at three anchors:",
        "**min** = min(PSO, GA effects) — the HEADLINE, most conservative;",
        "mean = average of the two; pso = PSO-only effect. Bootstrap CIs are",
        f"BCa, {first.n_resamples} resamples, seed {first.bootstrap_seed}.",
        "",
        "### Difference estimates (bootstrap 95% BCa CIs)",
        "",
        "| metric | mean A | mean B | dmean (A-B) [95% CI] | Cohen's d [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.metric} | {r.mean_a:.4g} | {r.mean_b:.4g} "
            f"| {r.diff_mean:.4g} [{r.ci_diff[0]:.4g}, {r.ci_diff[1]:.4g}] "
            f"| {r.cohens_d:.3f} [{r.ci_d[0]:.3f}, {r.ci_d[1]:.3f}] |"
        )
    lines += [
        "",
        "### TOST sensitivity over bound anchors",
        "",
        "| metric | anchor | anchor value | fraction | +-bound | p TOST | equivalent? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for row in r.rows:
            verdict = "YES" if row.equivalent else "no"
            lines.append(
                f"| {r.metric} | {row.anchor} | {r.anchors[row.anchor]:.4g} "
                f"| {row.fraction:.0%} | {row.bound:.4g} "
                f"| {row.p_tost:.2e} | {verdict} |"
            )
    lines.append("")
    return lines


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
