"""Publication figures (adapting the prior study's plot.py layouts).

Encoding contract (kept fixed across every figure so a reader can transfer
what they learned from one figure to the next):

- HUE encodes the memory mode — the factor this study isolates:
  cold #2a78d6 (blue), injection #1baf7a (aqua), persistence #e34948 (red).
  The triple was checked with the dataviz palette validator (worst adjacent
  CVD deltaE 21.2 on the light surface; the aqua slot is sub-3:1 contrast, so
  every figure carries a legend and the stats reports provide the table view).
- LINESTYLE/HATCH encodes the algorithm: PSO solid, GA dashed/hatched
  (identity is never carried by color alone).
- Landscape mode is a facet or an explicit label suffix, never a repaint.

Single y-axis everywhere; grids are recessive hairlines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..fitness.fitness import FitnessWeights, scheduled_weights
from ..metrics.effort import summarize_effort
from ..trajectories.generators import Trajectory

MEMORY_COLORS = {"cold": "#2a78d6", "injection": "#1baf7a", "persistence": "#e34948"}
ALGO_LINESTYLES = {"pso": "-", "ga": "--"}
ALGO_HATCHES = {"pso": "", "ga": "//"}
GRID = dict(color="#e1e0d9", linewidth=0.8)
INK_MUTED = "#898781"


def _split(cell_id: str) -> tuple[str, str, str]:
    algorithm, memory, landscape = cell_id.split("-")
    return algorithm, memory, landscape


def cell_style(cell_id: str) -> dict:
    algorithm, memory, landscape = _split(cell_id)
    label = f"{algorithm.upper()} {memory}"
    if landscape == "adaptive":
        label += " (adaptive)"
    return {
        "color": MEMORY_COLORS[memory],
        "linestyle": ALGO_LINESTYLES[algorithm],
        "hatch": ALGO_HATCHES[algorithm],
        "label": label,
    }


def _style_axes(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _best_run(cell_data: dict) -> dict:
    rmse = [a["rmse_pos_mm"] for a in cell_data["accuracy"]]
    return cell_data["runs"][int(np.argmin(rmse))]


# ---------------------------------------------------------------------------

def fig_trajectory_3d(
    per_cell: dict, trajectory: Trajectory, out_dir: Path, cells: list[str] | None = None
) -> Path:
    """Desired path vs the best run of each selected cell (qualitative)."""
    from ..kinematics.dh import fk_position

    cells = list(per_cell) if cells is None else cells
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(*trajectory.positions.T, "--", color="#0b0b0b", lw=2.0,
            label="desired", zorder=5)
    for cell_id in cells:
        style = cell_style(cell_id)
        run = _best_run(per_cell[cell_id])
        achieved = np.array([fk_position(q) for q in run["q_path"]])
        ax.plot(*achieved.T, style["linestyle"], color=style["color"], lw=1.8,
                marker="o", ms=3, alpha=0.85, label=style["label"])

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Tracking, best run per cell — {trajectory.name}")
    ax.view_init(elev=28, azim=-30)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_dir, f"trajectory3d_{trajectory.name}.png")


def fig_error_per_point(
    per_cell: dict, trajectory: Trajectory, out_dir: Path, which: str = "pos"
) -> Path:
    """Mean +- sd tracking error per trajectory point, all runs."""
    from ..metrics.accuracy import pose_errors

    fig, ax = plt.subplots(figsize=(9, 4.5))
    points = np.arange(1, len(trajectory) + 1)
    for cell_id, data in per_cell.items():
        style = cell_style(cell_id)
        errors = []
        for run in data["runs"]:
            pos_err, ori_err = pose_errors(run["q_path"], trajectory)
            errors.append(pos_err if which == "pos" else ori_err)
        errors = np.array(errors)
        mu, sd = errors.mean(axis=0), errors.std(axis=0)
        ax.fill_between(points, mu - sd, mu + sd, color=style["color"], alpha=0.12)
        ax.plot(points, mu, style["linestyle"], color=style["color"], lw=2,
                label=style["label"])

    unit = "mm" if which == "pos" else "rad"
    ax.set_xlabel("trajectory point")
    ax.set_ylabel(f"{'position' if which == 'pos' else 'orientation'} error ({unit})")
    ax.set_title(f"Tracking error per point (mean ± sd) — {trajectory.name}")
    ax.legend(fontsize=8)
    _style_axes(ax)
    return _save(fig, out_dir, f"error_per_point_{which}_{trajectory.name}.png")


def fig_joint_profiles(per_cell: dict, trajectory: Trajectory, out_dir: Path) -> Path:
    """Joint-angle evolution of the best run, one panel per cell."""
    cells = list(per_cell)
    n = len(cells)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    for k, cell_id in enumerate(cells):
        ax = axes[k // ncols][k % ncols]
        style = cell_style(cell_id)
        run = _best_run(per_cell[cell_id])
        q_deg = np.degrees(run["q_path"])
        for j in range(q_deg.shape[1]):
            ax.plot(q_deg[:, j], lw=1.6, alpha=0.9)
        ax.set_title(style["label"], fontsize=10)
        _style_axes(ax)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("trajectory point")
    for row in axes:
        row[0].set_ylabel("joint angle (deg)")
    fig.suptitle(f"Joint profiles, best run — {trajectory.name}", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, f"joint_profiles_{trajectory.name}.png")


def fig_metric_boxplots(
    per_cell: dict, trajectory_name: str, out_dir: Path,
    metrics: tuple[tuple[str, str, str], ...] = (
        ("accuracy", "rmse_pos_mm", "position RMSE (mm)"),
        ("accuracy", "rmse_ori_rad", "orientation RMSE (rad)"),
        ("kinematic", "smoothness_rad", "smoothness (rad)"),
        ("kinematic", "jerk_rad", "jerk (rad)"),
    ),
) -> Path:
    cells = list(per_cell)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 4.2))
    for ax, (family, key, title) in zip(np.atleast_1d(axes), metrics):
        values = [
            [run[key] for run in per_cell[c][family]] for c in cells
        ]
        bp = ax.boxplot(values, patch_artist=True, widths=0.55,
                        medianprops=dict(color="#0b0b0b", lw=1.8))
        for patch, cell_id in zip(bp["boxes"], cells):
            style = cell_style(cell_id)
            patch.set_facecolor(style["color"] + "99")
            patch.set_hatch(style["hatch"])
            patch.set_edgecolor(style["color"])
        short = {"cold": "cold", "injection": "inject.", "persistence": "persist."}
        labels = []
        for c in cells:
            algorithm, memory, landscape = _split(c)
            label = f"{algorithm.upper()}\n{short[memory]}"
            if landscape == "adaptive":
                label += "\n(adapt.)"
            labels.append(label)
        ax.set_xticks(range(1, len(cells) + 1))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        _style_axes(ax)
    fig.suptitle(f"Per-run metrics — {trajectory_name}", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, f"boxplots_{trajectory_name}.png")


def fig_convergence_first_point(
    per_cell: dict, trajectory_name: str, out_dir: Path, point: int = 0
) -> Path:
    """Mean incumbent fitness per iteration at one trajectory point."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cell_id, data in per_cell.items():
        style = cell_style(cell_id)
        curves = np.array([run["fitness_hist"][point] for run in data["runs"]])
        for c in curves:
            ax.plot(c, color=style["color"], alpha=0.10, lw=0.7)
        ax.plot(curves.mean(axis=0), style["linestyle"], color=style["color"],
                lw=2.2, label=style["label"])
    ax.set_yscale("log")
    ax.set_xlabel("iteration / generation")
    ax.set_ylabel("incumbent fitness (dimensionless)")
    ax.set_title(
        f"Convergence at trajectory point {point + 1} — {trajectory_name} "
        "(thin: individual runs; thick: mean)"
    )
    ax.legend(fontsize=8)
    _style_axes(ax)
    return _save(fig, out_dir, f"convergence_point{point + 1}_{trajectory_name}.png")


def fig_effort_vs_epsilon(per_cell: dict, trajectory_name: str, out_dir: Path) -> Path:
    """Median effort to reach epsilon + fraction of points reaching it."""
    fig, (ax_eff, ax_frac) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, height_ratios=[2, 1]
    )
    for cell_id, data in per_cell.items():
        style = cell_style(cell_id)
        summary = summarize_effort(data["effort"])
        eps = summary["epsilon_values"]
        ax_eff.plot(eps, summary["median_iterations"], style["linestyle"],
                    color=style["color"], lw=2, marker="o", ms=4,
                    label=style["label"])
        ax_frac.plot(eps, summary["fraction_reached"], style["linestyle"],
                     color=style["color"], lw=2, marker="o", ms=4)
    ax_eff.set_xscale("log")
    ax_eff.set_ylabel("median iterations to reach ε")
    ax_eff.set_title(
        f"Effort vs accuracy — {trajectory_name}\n"
        "(censored points counted at the iteration cap; lower panel shows coverage)"
    )
    ax_eff.legend(fontsize=8)
    ax_frac.set_xlabel("ε (dimensionless pose error, log scale)")
    ax_frac.set_ylabel("fraction reaching ε")
    ax_frac.set_ylim(-0.02, 1.02)
    _style_axes(ax_eff)
    _style_axes(ax_frac)
    fig.tight_layout()
    return _save(fig, out_dir, f"effort_vs_epsilon_{trajectory_name}.png")


def fig_weight_schedule(
    base: FitnessWeights, n_points: int, out_dir: Path
) -> Path:
    """Static vs adaptive fitness weights along the trajectory (diagnostic)."""
    t = np.linspace(0.0, 1.0, n_points)
    scheduled = [scheduled_weights(base, ti) for ti in t]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    steps = np.arange(1, n_points + 1)

    ax1.plot(steps, [base.w_pos] * n_points, "--", color="#0b0b0b", lw=1.6,
             label="static")
    ax1.plot(steps, [w.w_pos for w in scheduled], color="#e34948", lw=2,
             label="adaptive")
    ax1.set_title("accuracy weights (w_pos = w_ori shape)", fontsize=10)
    ax2.plot(steps, [base.w_joint] * n_points, "--", color="#0b0b0b", lw=1.6,
             label="static")
    ax2.plot(steps, [w.w_joint for w in scheduled], color="#e34948", lw=2,
             label="adaptive")
    ax2.set_title("continuity weight (w_joint)", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xlabel("trajectory point")
        ax.set_ylabel("weight (dimensionless)")
        ax.legend(fontsize=8)
        _style_axes(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "weight_schedule.png")


# ---------------------------------------------------------------------------

def make_all_figures(collected: dict, config, out_dir: str | Path) -> list[Path]:
    """Standard figure set for every trajectory in a collected results dict."""
    from ..trajectories.generators import build_trajectory

    out_dir = Path(out_dir)
    paths = [fig_weight_schedule(config.weights, config.trajectories[0].n_points, out_dir)]
    for spec in config.trajectories:
        trajectory = build_trajectory(spec)
        per_cell = collected[spec.name]
        paths += [
            fig_trajectory_3d(per_cell, trajectory, out_dir),
            fig_error_per_point(per_cell, trajectory, out_dir, which="pos"),
            fig_error_per_point(per_cell, trajectory, out_dir, which="ori"),
            fig_joint_profiles(per_cell, trajectory, out_dir),
            fig_metric_boxplots(per_cell, spec.name, out_dir),
            fig_convergence_first_point(per_cell, spec.name, out_dir),
            fig_effort_vs_epsilon(per_cell, spec.name, out_dir),
        ]
    return paths
