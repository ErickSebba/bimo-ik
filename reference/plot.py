"""
Visualizacao dos resultados oficiais do main.py.

[FROZEN REFERENCE COPY - see reference/README.md.]

Este modulo NAO gera dados de otimizaao.
Ele recebe os resultados prontos (res_pso, res_dfl, res_ag)
e gera os graficos comparativos.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

warnings.filterwarnings("ignore")


def _melhor_historico_q(resultados):
    idx_melhor = int(np.argmin([r["rmse_pos_mm"] for r in resultados]))
    return resultados[idx_melhor]["historico_q"]


def _limites_globais(trajs_q, posicoes_dos_elos_fn):
    todos_pts = []
    for traj_q in trajs_q:
        for q in traj_q:
            todos_pts.append(posicoes_dos_elos_fn(q))
    todos_pts = np.vstack(todos_pts)

    x_min = todos_pts[:, 0].min() - 30
    x_max = todos_pts[:, 0].max() + 30
    y_min = todos_pts[:, 1].min() - 30
    y_max = todos_pts[:, 1].max() + 30
    z_min = max(0.0, todos_pts[:, 2].min() - 30)
    z_max = todos_pts[:, 2].max() + 30
    return x_min, x_max, y_min, y_max, z_min, z_max


def gerar_animacao_3d_comparativa(
    res_pso,
    res_dfl,
    res_ag,
    trajetoria,
    posicoes_dos_elos_fn,
    cinematica_direta_fn,
    output_dir,
):
    print("\n[PLOT] Gerando animacao 3D comparativa...")

    traj_q_pso = _melhor_historico_q(res_pso)
    traj_q_dfl = _melhor_historico_q(res_dfl)
    traj_q_ag = _melhor_historico_q(res_ag)
    n_passos = len(trajetoria)

    x_min, x_max, y_min, y_max, z_min, z_max = _limites_globais(
        [traj_q_pso, traj_q_dfl, traj_q_ag], posicoes_dos_elos_fn
    )

    fig = plt.figure(figsize=(18, 5))
    ax_pso = fig.add_subplot(131, projection="3d")
    ax_dfl = fig.add_subplot(132, projection="3d")
    ax_ag = fig.add_subplot(133, projection="3d")

    trace_pso = {"x": [], "y": [], "z": []}
    trace_dfl = {"x": [], "y": [], "z": []}
    trace_ag = {"x": [], "y": [], "z": []}

    def _cfg(ax, titulo):
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=-45)

    def _plot_alg(ax, q, alvo, trace, cor, nome):
        elos = posicoes_dos_elos_fn(q)
        ee = cinematica_direta_fn(q)

        ax.plot(
            elos[:, 0],
            elos[:, 1],
            elos[:, 2],
            "o-",
            color=cor,
            linewidth=2.5,
            markersize=5,
            label="Braco",
        )
        ax.scatter(
            ee[0],
            ee[1],
            ee[2],
            color="#ff7f0e",
            s=120,
            marker="*",
            zorder=5,
            label="End-effector",
        )
        ax.scatter(
            alvo[0],
            alvo[1],
            alvo[2],
            color="lime",
            s=100,
            marker="x",
            linewidths=2.5,
            zorder=5,
            label="Alvo",
        )

        trace["x"].append(float(ee[0]))
        trace["y"].append(float(ee[1]))
        trace["z"].append(float(ee[2]))
        if len(trace["x"]) > 1:
            ax.plot(trace["x"], trace["y"], trace["z"], "--", color=cor, alpha=0.4, linewidth=1.5)
        ax.legend(loc="upper left", fontsize=8)

    def atualizar(frame):
        ax_pso.clear()
        ax_dfl.clear()
        ax_ag.clear()

        _cfg(ax_pso, f"PSO - Ponto {frame + 1}/{n_passos}")
        _cfg(ax_dfl, f"PSO-DFL - Ponto {frame + 1}/{n_passos}")
        _cfg(ax_ag, f"AG - Ponto {frame + 1}/{n_passos}")

        _plot_alg(ax_pso, traj_q_pso[frame], trajetoria[frame], trace_pso, "#1f77b4", "PSO")
        _plot_alg(ax_dfl, traj_q_dfl[frame], trajetoria[frame], trace_dfl, "#9467bd", "PSO-DFL")
        _plot_alg(ax_ag, traj_q_ag[frame], trajetoria[frame], trace_ag, "#d62728", "AG")

    anim = FuncAnimation(fig, atualizar, frames=n_passos, interval=500, repeat=True)

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "animacao_comparativa_3alg.gif")
    try:
        anim.save(gif_path, writer=PillowWriter(fps=2))
        print(f"  [OK] {gif_path}")
    except Exception as exc:
        print(f"  [ERRO] Falha ao salvar GIF: {exc}")
    plt.close()


def gerar_grafico_3d_estatico_comparativo(
    res_pso,
    res_dfl,
    res_ag,
    trajetoria,
    posicoes_dos_elos_fn,
    cinematica_direta_fn,
    output_dir,
):
    print("\n[PLOT] Gerando grafico 3D estatico comparativo...")

    traj_q_pso = _melhor_historico_q(res_pso)
    traj_q_dfl = _melhor_historico_q(res_dfl)
    traj_q_ag = _melhor_historico_q(res_ag)
    n_passos = len(trajetoria)

    x_min, x_max, y_min, y_max, z_min, z_max = _limites_globais(
        [traj_q_pso, traj_q_dfl, traj_q_ag], posicoes_dos_elos_fn
    )

    pontos = [0, n_passos // 4, n_passos // 2, (3 * n_passos) // 4, n_passos - 1]

    fig = plt.figure(figsize=(18, 12))
    for idx, ponto in enumerate(pontos):
        for row, (nome, traj_q, cor) in enumerate(
            [
                ("PSO", traj_q_pso, "#1f77b4"),
                ("PSO-DFL", traj_q_dfl, "#9467bd"),
                ("AG", traj_q_ag, "#d62728"),
            ]
        ):
            ax = fig.add_subplot(3, 5, row * 5 + idx + 1, projection="3d")
            q = traj_q[ponto]
            elos = posicoes_dos_elos_fn(q)
            ee = cinematica_direta_fn(q)
            alvo = trajetoria[ponto]

            ax.plot(elos[:, 0], elos[:, 1], elos[:, 2], "o-", color=cor, linewidth=2, markersize=5)
            ax.scatter(ee[0], ee[1], ee[2], color="#ff7f0e", s=100, marker="*", zorder=5)
            ax.scatter(alvo[0], alvo[1], alvo[2], color="lime", s=80, marker="x", linewidths=2, zorder=5)
            ax.set_title(f"{nome} - Ponto {ponto + 1}", fontsize=10)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=20, azim=-45)

    fig.suptitle("Comparativo 3D: Braco Robotico - PSO vs PSO-DFL vs AG", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "comparativo_3d_estatico_3alg.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] {fig_path}")
    plt.close()


def gerar_grafico_trajetoria(res_pso, res_dfl, res_ag, trajetoria, cinematica_direta_fn, output_dir):
    print("\n[PLOT] Gerando grafico de erro por ponto...")

    traj_q_pso = _melhor_historico_q(res_pso)
    traj_q_dfl = _melhor_historico_q(res_dfl)
    traj_q_ag = _melhor_historico_q(res_ag)
    n_passos = len(trajetoria)

    erros_pso = []
    erros_dfl = []
    erros_ag = []

    for i in range(n_passos):
        erro_pso = np.linalg.norm(cinematica_direta_fn(traj_q_pso[i]) - trajetoria[i])
        erro_dfl = np.linalg.norm(cinematica_direta_fn(traj_q_dfl[i]) - trajetoria[i])
        erro_ag = np.linalg.norm(cinematica_direta_fn(traj_q_ag[i]) - trajetoria[i])
        erros_pso.append(erro_pso)
        erros_dfl.append(erro_dfl)
        erros_ag.append(erro_ag)

    pontos = np.arange(1, n_passos + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pontos, erros_pso, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label=f"PSO (medio={np.mean(erros_pso):.2f} mm)")
    ax.plot(pontos, erros_dfl, "s-", color="#9467bd", linewidth=2.5, markersize=8, label=f"PSO-DFL (medio={np.mean(erros_dfl):.2f} mm)")
    ax.plot(pontos, erros_ag, "^-", color="#d62728", linewidth=2.5, markersize=8, label=f"AG (medio={np.mean(erros_ag):.2f} mm)")

    ax.set_title("Erro de Rastreamento por Ponto da Trajetoria", fontsize=14, fontweight="bold")
    ax.set_xlabel("Ponto da Trajetoria", fontsize=12)
    ax.set_ylabel("Erro de Posicao (mm)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "erro_rastreamento_3alg.png")
    plt.savefig(fig_path, dpi=300)
    print(f"  [OK] {fig_path}")
    plt.close()
