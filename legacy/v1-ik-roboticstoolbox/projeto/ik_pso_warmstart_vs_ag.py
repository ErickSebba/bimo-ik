"""
=============================================================================
  CINEMÁTICA INVERSA DO ROBÔ BIMO (6-DOF)
  Comparativo Experimental: PSO com Warm-Start vs Algoritmo Genético
  
  Sem dependência de RoboticToolbox - Implementação manual de DH
  
  Artigo: "Comparativo entre PSO e AG na Cinemática Inversa de um
          Braço Robótico Educacional de 6-DoF"
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import warnings
warnings.filterwarnings("ignore")

# Criar pasta de saída
OUTPUT_DIR = "OutputsResultados"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO GLOBAL DO EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────
N_RUNS       = 10      # execuções independentes
N_PASSOS     = 10      # pontos na trajetória
PSO_POP      = 60      # partículas PSO
PSO_ITER     = 80      # iterações PSO
AG_POP       = 60      # indivíduos AG
AG_GENS      = 80      # gerações AG
TAXA_CROSS   = 0.7     # probabilidade de crossover
TAXA_MUT     = 0.15    # probabilidade de mutação por gene
SEED_BASE    = 42      # semente-base

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARÂMETROS DH - ROBÔ BIMO (ARQUEADO)
# ─────────────────────────────────────────────────────────────────────────────
DH_params = [
    [0, 5, 0, np.pi/2],     # Junta 1
    [0, 0, 6, 0],           # Junta 2
    [0, 0, 5, 0],           # Junta 3
    [0, 0, 0, np.pi/2],     # Junta 4
    [0, 0, 0, -np.pi/2],    # Junta 5
    [0, 2, 0, 0]            # Junta 6
]

# Limites articulares individuais (rad)
LIMITES_INF = np.array([-np.pi, -np.pi/2, -np.pi/2, -np.pi, -np.pi/2, -np.pi])
LIMITES_SUP = np.array([np.pi, np.pi/2, np.pi/2, np.pi, np.pi/2, np.pi])

# ─────────────────────────────────────────────────────────────────────────────
# 2. MATRIZ DH
# ─────────────────────────────────────────────────────────────────────────────
def dh_matrix(theta, d, a, alpha):
    """Calcula a matriz de transformação DH."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

# ─────────────────────────────────────────────────────────────────────────────
# 3. CINEMÁTICA DIRETA
# ─────────────────────────────────────────────────────────────────────────────
def forward_kinematics(thetas):
    """Retorna posições dos elos e matriz homogênea final."""
    T = np.eye(4)
    positions = [[0, 0, 0]]
    
    for i in range(6):
        _, d, a, alpha = DH_params[i]
        T = T @ dh_matrix(thetas[i], d, a, alpha)
        positions.append(T[:3, 3])
    
    return np.array(positions), T

# ─────────────────────────────────────────────────────────────────────────────
# 4. FITNESS - Com elbow penalty e suavidade
# ─────────────────────────────────────────────────────────────────────────────
def fitness(theta, target, q_anterior=None):
    """
    Função de fitness multi-objetivo:
    - Erro de posição (cartesiano)
    - Penalidade de cotovelo
    - Penalidade de suavidade articular
    """
    positions, T = forward_kinematics(theta)
    
    # Erro de posição
    pos_error = np.linalg.norm(T[:3, 3] - target)
    
    # Penalidade de cotovelo (altura desejada = 0.8 * altura do alvo)
    elbow_z = positions[3][2]
    desired_height = target[2] * 0.8
    elbow_penalty = max(0, desired_height - elbow_z) * 0.4
    
    # Penalidade de suavidade (mudança articular)
    if q_anterior is not None:
        smooth_penalty = np.linalg.norm(theta - q_anterior) * 0.05
    else:
        smooth_penalty = 0
    
    return pos_error + elbow_penalty + smooth_penalty

# ─────────────────────────────────────────────────────────────────────────────
# 5. PSO - Particle Swarm Optimization (personalizado)
# ─────────────────────────────────────────────────────────────────────────────
def solve_ik_pso(target, initial_guess=None, q_anterior=None):
    """
    Resolve IK usando PSO com doping (warm-start).
    Retorna solução, histórico de fitness e histórico de posições de partículas.
    """
    num_particles = PSO_POP
    max_iter = PSO_ITER
    w, c1, c2 = 0.5, 1.5, 1.5
    
    # Inicialização com doping
    if initial_guess is None:
        n_global = int(num_particles * 0.6)
        n_local = num_particles - n_global
        global_particles = np.random.uniform(LIMITES_INF, LIMITES_SUP, (n_global, 6))
        local_particles = np.random.uniform(-1, 1, (n_local, 6))
        particles = np.vstack([global_particles, local_particles])
    else:
        n_local = int(num_particles * 0.5)
        n_global = num_particles - n_local
        local_particles = initial_guess + np.random.normal(0, 0.1, (n_local, 6))
        global_particles = np.random.uniform(LIMITES_INF, LIMITES_SUP, (n_global, 6))
        particles = np.vstack([local_particles, global_particles])
    
    np.random.shuffle(particles)
    
    velocities = np.zeros((num_particles, 6))
    pbest = particles.copy()
    pbest_scores = np.array([fitness(p, target, q_anterior) for p in particles])
    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)
    
    history = []
    particles_history = []  # Histórico de posições cartesianas das partículas
    
    for it in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (pbest[i] - particles[i]) +
                           c2 * r2 * (gbest - particles[i]))
            
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], LIMITES_INF, LIMITES_SUP)
            
            score = fitness(particles[i], target, q_anterior)
            
            if score < pbest_scores[i]:
                pbest[i] = particles[i]
                pbest_scores[i] = score
        
        best_idx = np.argmin(pbest_scores)
        if pbest_scores[best_idx] < gbest_score:
            gbest = pbest[best_idx]
            gbest_score = pbest_scores[best_idx]
        
        history.append(gbest_score)
        
        # Salva posições cartesianas de todas as partículas nesta iteração
        particle_positions = []
        for p in particles:
            _, T = forward_kinematics(p)
            particle_positions.append(T[:3, 3])
        particles_history.append(np.array(particle_positions))
    
    return gbest, history, particles_history

# ─────────────────────────────────────────────────────────────────────────────
# 6. AG - Algoritmo Genético
# ─────────────────────────────────────────────────────────────────────────────
def _selecao_roleta(pop, scores):
    """Seleção por roleta."""
    inv = 1.0 / (scores + 1e-12)
    probs = inv / inv.sum()
    idx = np.random.choice(len(pop), size=len(pop), replace=True, p=probs)
    return pop[idx]

def _crossover(p1, p2, taxa=TAXA_CROSS):
    """Crossover aritmético."""
    if np.random.rand() < taxa:
        alpha = np.random.rand()
        f1 = alpha * p1 + (1 - alpha) * p2
        f2 = (1 - alpha) * p1 + alpha * p2
    else:
        f1, f2 = p1.copy(), p2.copy()
    return f1, f2

def _mutacao(ind, taxa=TAXA_MUT):
    """Mutação gaussiana."""
    novo = ind.copy()
    mask = np.random.rand(6) < taxa
    novo[mask] += np.random.normal(0, 0.1, mask.sum())
    return np.clip(novo, LIMITES_INF, LIMITES_SUP)

def solve_ik_ag(target, initial_guess=None, q_anterior=None):
    """
    Resolve IK usando AG com doping.
    Retorna solução, histórico de fitness e histórico de posições da população.
    """
    # Inicialização
    pop = np.random.uniform(LIMITES_INF, LIMITES_SUP, (AG_POP, 6))
    
    # Doping
    if initial_guess is not None:
        pop[0] = np.clip(initial_guess, LIMITES_INF, LIMITES_SUP)
        pop[1] = np.clip(initial_guess + np.random.uniform(-0.05, 0.05, 6),
                        LIMITES_INF, LIMITES_SUP)
    
    history = []
    population_history = []  # Histórico de posições cartesianas da população
    
    for gen in range(AG_GENS):
        scores = np.array([fitness(ind, target, q_anterior) for ind in pop])
        
        # Elitismo
        idx_elite = np.argsort(scores)[:2]
        elite = pop[idx_elite].copy()
        
        # Seleção
        pais = _selecao_roleta(pop, scores)
        
        # Crossover + Mutação
        filhos = []
        for j in range(0, AG_POP - 2, 2):
            f1, f2 = _crossover(pais[j], pais[j+1])
            filhos.append(_mutacao(f1))
            filhos.append(_mutacao(f2))
        
        while len(filhos) < AG_POP - 2:
            filhos.append(_mutacao(pais[-1]))
        
        pop = np.vstack([elite, np.array(filhos[:AG_POP-2])])
        
        scores_new = np.array([fitness(ind, target, q_anterior) for ind in pop])
        history.append(np.min(scores_new))
        
        # Salva posições cartesianas de todos os indivíduos nesta geração
        population_positions = []
        for ind in pop:
            _, T = forward_kinematics(ind)
            population_positions.append(T[:3, 3])
        population_history.append(np.array(population_positions))
    
    idx_best = np.argmin(scores_new)
    return pop[idx_best], history, population_history

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAJETÓRIA
# ─────────────────────────────────────────────────────────────────────────────
TRAJETORIA = np.array([
    np.array([6 + 2*np.cos(t/2), 4 + 2*np.sin(t/2), 4 + 1*np.sin(t/3)])
    for t in range(N_PASSOS)
])

# ─────────────────────────────────────────────────────────────────────────────
# 8. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
def calcular_metricas(historico_q, historico_erro):
    """Calcula métricas para o algoritmo."""
    erros_pos = []
    for i in range(N_PASSOS):
        _, T = forward_kinematics(historico_q[i])
        pos_atual = T[:3, 3]
        erro = np.linalg.norm(pos_atual - TRAJETORIA[i])
        erros_pos.append(erro)
    
    erros_pos = np.array(erros_pos)
    
    return {
        "rmse_pos": np.sqrt(np.mean(erros_pos**2)),
        "erro_medio": np.mean(erros_pos),
        "erro_max": np.max(erros_pos),
        "suavidade": np.sum(np.abs(np.diff(historico_q, axis=0))),
        "fitness_medio": np.mean(historico_erro),
        "fitness_max": np.max(historico_erro),
        "erros_pos": erros_pos,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────
def rodar_experimento():
    print("=" * 60)
    print(f"EXPERIMENTO: PSO vs AG – {N_RUNS} execuções")
    print(f"Trajetória: {N_PASSOS} pontos")
    print("=" * 60)
    
    resultados_pso = []
    resultados_ag = []
    historico_pso_evolucao = []  # Para visualizar evolução
    historico_ag_evolucao = []
    
    for run in range(N_RUNS):
        seed = SEED_BASE + run
        np.random.seed(seed)
        
        print(f"\n▶ Run {run+1}/{N_RUNS}  (seed={seed})")
        
        # PSO
        q_ant_pso = np.zeros(6)
        hq_pso, he_pso = [], []
        pso_evol = []  # Evolução para primeiro ponto apenas
        t0_pso = time.time()
        
        for i, alvo in enumerate(TRAJETORIA):
            sol, hist, part_hist = solve_ik_pso(alvo, q_anterior=q_ant_pso)
            hq_pso.append(sol)
            he_pso.append(hist[-1])
            if i == 0:  # Salva evolução do primeiro ponto
                pso_evol = part_hist
            q_ant_pso = sol
        
        t_pso = time.time() - t0_pso
        m_pso = calcular_metricas(np.array(hq_pso), np.array(he_pso))
        m_pso["tempo_s"] = t_pso
        resultados_pso.append(m_pso)
        historico_pso_evolucao.append(pso_evol)
        print(f"  [PSO] RMSE={m_pso['rmse_pos']:.3f}  MaxErr={m_pso['erro_max']:.3f}  t={t_pso:.1f}s")
        
        # AG
        q_ant_ag = np.zeros(6)
        hq_ag, he_ag = [], []
        ag_evol = []  # Evolução para primeiro ponto apenas
        t0_ag = time.time()
        
        for i, alvo in enumerate(TRAJETORIA):
            sol, hist, pop_hist = solve_ik_ag(alvo, q_anterior=q_ant_ag)
            hq_ag.append(sol)
            he_ag.append(hist[-1])
            if i == 0:  # Salva evolução do primeiro ponto
                ag_evol = pop_hist
            q_ant_ag = sol
        
        t_ag = time.time() - t0_ag
        m_ag = calcular_metricas(np.array(hq_ag), np.array(he_ag))
        m_ag["tempo_s"] = t_ag
        resultados_ag.append(m_ag)
        historico_ag_evolucao.append(ag_evol)
        print(f"  [AG]  RMSE={m_ag['rmse_pos']:.3f}  MaxErr={m_ag['erro_max']:.3f}  t={t_ag:.1f}s")
    
    return resultados_pso, resultados_ag, historico_pso_evolucao, historico_ag_evolucao

# ─────────────────────────────────────────────────────────────────────────────
# 10. TABELA COMPARATIVA
# ─────────────────────────────────────────────────────────────────────────────
def imprimir_tabela(resultados_pso, resultados_ag):
    metricas_labels = [
        ("rmse_pos", "RMSE Posição"),
        ("erro_medio", "Erro Médio"),
        ("erro_max", "Erro Máximo"),
        ("suavidade", "Suavidade"),
        ("fitness_medio", "Fitness Médio"),
        ("tempo_s", "Tempo Exec. (s)"),
    ]
    
    print("\n" + "=" * 65)
    print(f"{'MÉTRICA':<25} {'PSO (μ ± σ)':>18}  {'AG  (μ ± σ)':>18}")
    print("=" * 65)
    
    for chave, nome in metricas_labels:
        vals_pso = np.array([r[chave] for r in resultados_pso])
        vals_ag = np.array([r[chave] for r in resultados_ag])
        mu_pso, sd_pso = vals_pso.mean(), vals_pso.std()
        mu_ag, sd_ag = vals_ag.mean(), vals_ag.std()
        print(f"{nome:<25} {mu_pso:8.4f} ± {sd_pso:.4f}  {mu_ag:8.4f} ± {sd_ag:.4f}")
    
    print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 11. VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────────────────────────
def gerar_graficos(resultados_pso, resultados_ag):
    cor_pso = "#1f77b4"   # azul
    cor_ag = "#d62728"    # vermelho
    
    # Gráfico: Erro de rastreamento por ponto
    fig, ax = plt.subplots(figsize=(9, 4))
    pontos = np.arange(1, N_PASSOS + 1)
    
    erros_pso_mat = np.array([r["erros_pos"] for r in resultados_pso])
    erros_ag_mat = np.array([r["erros_pos"] for r in resultados_ag])
    mu_pso = erros_pso_mat.mean(axis=0)
    sd_pso = erros_pso_mat.std(axis=0)
    mu_ag = erros_ag_mat.mean(axis=0)
    sd_ag = erros_ag_mat.std(axis=0)
    
    ax.fill_between(pontos, mu_pso - sd_pso, mu_pso + sd_pso,
                    alpha=0.2, color=cor_pso)
    ax.plot(pontos, mu_pso, "-o", ms=4, lw=2, color=cor_pso,
            label=f"PSO  μ={mu_pso.mean():.2f}")
    ax.fill_between(pontos, mu_ag - sd_ag, mu_ag + sd_ag,
                    alpha=0.2, color=cor_ag)
    ax.plot(pontos, mu_ag, "-s", ms=4, lw=2, color=cor_ag,
            label=f"AG   μ={mu_ag.mean():.2f}")
    
    ax.set_title(f"Erro de Rastreamento por Ponto (N={N_RUNS} execuções, μ ± σ)")
    ax.set_xlabel("Ponto da Trajetória")
    ax.set_ylabel("Erro de Posição")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig_erro_rastreamento.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"  → {fig_path} salva")
    
    # Boxplot comparativo
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    metricas_box = [
        ("rmse_pos", "RMSE Posição"),
        ("suavidade", "Suavidade"),
        ("tempo_s", "Tempo Exec. (s)"),
    ]
    
    for ax, (chave, titulo) in zip(axes, metricas_box):
        vals_pso = [r[chave] for r in resultados_pso]
        vals_ag = [r[chave] for r in resultados_ag]
        bp = ax.boxplot([vals_pso, vals_ag], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor(cor_pso + "88")
        bp["boxes"][1].set_facecolor(cor_ag + "88")
        ax.set_xticklabels(["PSO", "AG"])
        ax.set_title(titulo)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    fig.suptitle(f"Comparativo PSO vs AG (N={N_RUNS} execuções)")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig_boxplot.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"  → {fig_path} salva")

# ─────────────────────────────────────────────────────────────────────────────
# 12. VISUALIZAÇÃO 3D - MOVIMENTO DO BRAÇO
# ─────────────────────────────────────────────────────────────────────────────
def gerar_animacao_3d(resultados_pso, resultados_ag):
    """Gera animação 3D lado-a-lado do movimento do braço PSO vs AG."""
    
    # Identifica melhor run de cada algoritmo
    idx_best_pso = np.argmin([r["rmse_pos"] for r in resultados_pso])
    idx_best_ag = np.argmin([r["rmse_pos"] for r in resultados_ag])
    
    # Executa novamente para obter ângulos completos (warm-start)
    np.random.seed(SEED_BASE + idx_best_pso)
    q_ant_pso = np.zeros(6)
    traj_pso = []
    for alvo in TRAJETORIA:
        sol, _, _ = solve_ik_pso(alvo, q_anterior=q_ant_pso)
        traj_pso.append(sol)
        q_ant_pso = sol
    
    np.random.seed(SEED_BASE + idx_best_ag)
    q_ant_ag = np.zeros(6)
    traj_ag = []
    for alvo in TRAJETORIA:
        sol, _, _ = solve_ik_ag(alvo, q_anterior=q_ant_ag)
        traj_ag.append(sol)
        q_ant_ag = sol
    
    # Cria figura com dois subplots 3D
    fig = plt.figure(figsize=(14, 6))
    ax_pso = fig.add_subplot(121, projection='3d')
    ax_ag = fig.add_subplot(122, projection='3d')
    
    # Função para desenhar braço
    def desenhar_braço(ax, q, cor='blue', alpha=1.0):
        """Desenha o braço dado seus ângulos articulares."""
        positions, _ = forward_kinematics(q)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'o-', color=cor, linewidth=2.5, markersize=6, alpha=alpha)
    
    # Configuração dos eixos
    for ax, titulo in [(ax_pso, "PSO"), (ax_ag, "AG")]:
        ax.set_title(f"Movimento do Braço - {titulo}", fontsize=12, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-5, 10)
        ax.set_ylim(-5, 10)
        ax.set_zlim(0, 12)
    
    # Rastreio de trajetória
    trace_pso_x = []
    trace_pso_y = []
    trace_pso_z = []
    trace_ag_x = []
    trace_ag_y = []
    trace_ag_z = []
    
    def atualizar(frame):
        """Atualiza animação para cada frame."""
        ax_pso.clear()
        ax_ag.clear()
        
        # Configurar eixos novamente
        for ax, titulo in [(ax_pso, "PSO"), (ax_ag, "AG")]:
            ax.set_title(f"Movimento do Braço - {titulo} | Ponto {frame+1}/{N_PASSOS}", 
                        fontsize=11)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(-5, 10)
            ax.set_ylim(-5, 10)
            ax.set_zlim(0, 12)
        
        # PSO
        positions_pso, T_pso = forward_kinematics(traj_pso[frame])
        ax_pso.plot(positions_pso[:, 0], positions_pso[:, 1], positions_pso[:, 2],
                   'o-', color='#1f77b4', linewidth=2.5, markersize=6)
        
        # Ponto final (end-effector)
        ax_pso.scatter(T_pso[0, 3], T_pso[1, 3], T_pso[2, 3],
                      color='#ff7f0e', s=100, marker='*', zorder=5, label='End-effector')
        
        # Ponto alvo
        ax_pso.scatter(TRAJETORIA[frame][0], TRAJETORIA[frame][1], TRAJETORIA[frame][2],
                      color='green', s=80, marker='x', linewidths=2, zorder=5, label='Target')
        
        # Rastreio até agora
        trace_pso_x.append(T_pso[0, 3])
        trace_pso_y.append(T_pso[1, 3])
        trace_pso_z.append(T_pso[2, 3])
        if len(trace_pso_x) > 1:
            ax_pso.plot(trace_pso_x, trace_pso_y, trace_pso_z,
                       'b--', alpha=0.5, linewidth=1, label='Trajectory')
        
        ax_pso.legend(loc='upper left', fontsize=8)
        
        # AG
        positions_ag, T_ag = forward_kinematics(traj_ag[frame])
        ax_ag.plot(positions_ag[:, 0], positions_ag[:, 1], positions_ag[:, 2],
                  'o-', color='#d62728', linewidth=2.5, markersize=6)
        
        # Ponto final
        ax_ag.scatter(T_ag[0, 3], T_ag[1, 3], T_ag[2, 3],
                     color='#ff7f0e', s=100, marker='*', zorder=5, label='End-effector')
        
        # Ponto alvo
        ax_ag.scatter(TRAJETORIA[frame][0], TRAJETORIA[frame][1], TRAJETORIA[frame][2],
                     color='green', s=80, marker='x', linewidths=2, zorder=5, label='Target')
        
        # Rastreio
        trace_ag_x.append(T_ag[0, 3])
        trace_ag_y.append(T_ag[1, 3])
        trace_ag_z.append(T_ag[2, 3])
        if len(trace_ag_x) > 1:
            ax_ag.plot(trace_ag_x, trace_ag_y, trace_ag_z,
                      'r--', alpha=0.5, linewidth=1, label='Trajectory')
        
        ax_ag.legend(loc='upper left', fontsize=8)
        
        return ax_pso, ax_ag
    
    # Cria animação
    anim = FuncAnimation(fig, atualizar, frames=N_PASSOS, interval=500, repeat=True)
    
    # Salva como GIF
    try:
        gif_path = os.path.join(OUTPUT_DIR, "braço_movimento_pso_vs_ag.gif")
        anim.save(gif_path, writer=PillowWriter(fps=2))
        print(f"  → {gif_path} salva")
    except Exception as e:
        print(f"  ! Erro ao salvar GIF: {e}")
    
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 13. VISUALIZAÇÃO 3D - APENAS UM PONTO (ESTÁTICO)
# ─────────────────────────────────────────────────────────────────────────────
def gerar_grafico_3d_estático(resultados_pso, resultados_ag):
    """Gera gráficos 3D estáticos mostrando melhor e pior caso."""
    
    # Identifica melhor run
    idx_best_pso = np.argmin([r["rmse_pos"] for r in resultados_pso])
    idx_best_ag = np.argmin([r["rmse_pos"] for r in resultados_ag])
    
    # Executa
    np.random.seed(SEED_BASE + idx_best_pso)
    q_ant_pso = np.zeros(6)
    traj_pso = []
    for alvo in TRAJETORIA:
        sol, _, _ = solve_ik_pso(alvo, q_anterior=q_ant_pso)
        traj_pso.append(sol)
        q_ant_pso = sol
    
    np.random.seed(SEED_BASE + idx_best_ag)
    q_ant_ag = np.zeros(6)
    traj_ag = []
    for alvo in TRAJETORIA:
        sol, _, _ = solve_ik_ag(alvo, q_anterior=q_ant_ag)
        traj_ag.append(sol)
        q_ant_ag = sol
    
    # Cria figura com múltiplos pontos
    fig = plt.figure(figsize=(14, 5))
    
    # PSO - Ponto inicial, meio e final
    for idx_plot, frame in enumerate([0, N_PASSOS//2, N_PASSOS-1]):
        ax = fig.add_subplot(2, 3, idx_plot + 1, projection='3d')
        
        positions_pso, T_pso = forward_kinematics(traj_pso[frame])
        ax.plot(positions_pso[:, 0], positions_pso[:, 1], positions_pso[:, 2],
               'o-', color='#1f77b4', linewidth=2.5, markersize=6, label='Braço')
        
        ax.scatter(T_pso[0, 3], T_pso[1, 3], T_pso[2, 3],
                  color='#ff7f0e', s=100, marker='*', zorder=5, label='End-effector')
        ax.scatter(TRAJETORIA[frame][0], TRAJETORIA[frame][1], TRAJETORIA[frame][2],
                  color='green', s=80, marker='x', linewidths=2, zorder=5, label='Target')
        
        ax.set_title(f"PSO - Ponto {frame+1}", fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-5, 10)
        ax.set_ylim(-5, 10)
        ax.set_zlim(0, 12)
        ax.legend(fontsize=8)
    
    # AG
    for idx_plot, frame in enumerate([0, N_PASSOS//2, N_PASSOS-1]):
        ax = fig.add_subplot(2, 3, idx_plot + 4, projection='3d')
        
        positions_ag, T_ag = forward_kinematics(traj_ag[frame])
        ax.plot(positions_ag[:, 0], positions_ag[:, 1], positions_ag[:, 2],
               'o-', color='#d62728', linewidth=2.5, markersize=6, label='Braço')
        
        ax.scatter(T_ag[0, 3], T_ag[1, 3], T_ag[2, 3],
                  color='#ff7f0e', s=100, marker='*', zorder=5, label='End-effector')
        ax.scatter(TRAJETORIA[frame][0], TRAJETORIA[frame][1], TRAJETORIA[frame][2],
                  color='green', s=80, marker='x', linewidths=2, zorder=5, label='Target')
        
        ax.set_title(f"AG - Ponto {frame+1}", fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-5, 10)
        ax.set_ylim(-5, 10)
        ax.set_zlim(0, 12)
        ax.legend(fontsize=8)
    
    fig.suptitle("Posições do Braço - PSO vs AG (Início, Meio, Fim)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig_braço_3d_estático.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"  → {fig_path} salva")

# ─────────────────────────────────────────────────────────────────────────────
# 14. VISUALIZAÇÃO 3D - EVOLUÇÃO DO PSO
# ─────────────────────────────────────────────────────────────────────────────
def gerar_grafico_evolucao_pso(historico_pso_evolucao, target):
    """Mostra evolução das partículas do PSO através das iterações."""
    
    # Usa primeira execução (melhor run)
    evolucao = historico_pso_evolucao[0]
    n_iteracoes = len(evolucao)
    
    # Seleciona iterações para visualizar (início, 25%, 50%, 75%, fim)
    indices = [0, n_iteracoes//4, n_iteracoes//2, 3*n_iteracoes//4, n_iteracoes-1]
    
    fig = plt.figure(figsize=(16, 4))
    
    for idx_plot, iter_idx in enumerate(indices):
        ax = fig.add_subplot(1, 5, idx_plot + 1, projection='3d')
        
        # Posições das partículas nesta iteração
        particles_pos = evolucao[iter_idx]
        
        # Plota partículas
        ax.scatter(particles_pos[:, 0], particles_pos[:, 1], particles_pos[:, 2],
                  c='#1f77b4', s=50, alpha=0.6, marker='o', edgecolors='navy', linewidth=0.5)
        
        # Plota alvo
        ax.scatter(target[0], target[1], target[2],
                  color='green', s=150, marker='*', zorder=5, edgecolors='darkgreen', linewidth=2)
        
        ax.set_title(f"Iteração {iter_idx+1}/{n_iteracoes}", fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 10)
        ax.set_zlim(0, 10)
    
    fig.suptitle(f"Evolução das Partículas PSO (Primeira Execução)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig_evolucao_pso_3d.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"  → {fig_path} salva")

# ─────────────────────────────────────────────────────────────────────────────
# 15. VISUALIZAÇÃO 3D - EVOLUÇÃO DO AG
# ─────────────────────────────────────────────────────────────────────────────
def gerar_grafico_evolucao_ag(historico_ag_evolucao, target):
    """Mostra evolução da população do AG através das gerações."""
    
    # Usa primeira execução
    evolucao = historico_ag_evolucao[0]
    n_geracoes = len(evolucao)
    
    # Seleciona gerações para visualizar (início, 25%, 50%, 75%, fim)
    indices = [0, n_geracoes//4, n_geracoes//2, 3*n_geracoes//4, n_geracoes-1]
    
    fig = plt.figure(figsize=(16, 4))
    
    for idx_plot, gen_idx in enumerate(indices):
        ax = fig.add_subplot(1, 5, idx_plot + 1, projection='3d')
        
        # Posições da população nesta geração
        population_pos = evolucao[gen_idx]
        
        # Plota população
        ax.scatter(population_pos[:, 0], population_pos[:, 1], population_pos[:, 2],
                  c='#d62728', s=50, alpha=0.6, marker='s', edgecolors='darkred', linewidth=0.5)
        
        # Plota alvo
        ax.scatter(target[0], target[1], target[2],
                  color='green', s=150, marker='*', zorder=5, edgecolors='darkgreen', linewidth=2)
        
        ax.set_title(f"Geração {gen_idx+1}/{n_geracoes}", fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 10)
        ax.set_zlim(0, 10)
    
    fig.suptitle(f"Evolução da População AG (Primeira Execução)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig_evolucao_ag_3d.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"  → {fig_path} salva")

# ─────────────────────────────────────────────────────────────────────────────
# 16. COMPARAÇÃO 3D - PSO VS AG (MESMA ITERAÇÃO)
# ─────────────────────────────────────────────────────────────────────────────
def gerar_grafico_comparacao_evolucao(historico_pso_evolucao, historico_ag_evolucao, target):
    """Compara lado-a-lado a evolução de PSO vs AG."""
    
    pso_evol = historico_pso_evolucao[0]
    ag_evol = historico_ag_evolucao[0]
    
    n_pso = len(pso_evol)
    n_ag = len(ag_evol)
    
    # Seleciona iterações/gerações para visualizar
    indices_pso = [0, n_pso//4, n_pso//2, 3*n_pso//4, n_pso-1]
    indices_ag = [0, n_ag//4, n_ag//2, 3*n_ag//4, n_ag-1]
    
    fig = plt.figure(figsize=(16, 8))
    
    # PSO
    for idx_plot, iter_idx in enumerate(indices_pso):
        ax = fig.add_subplot(2, 5, idx_plot + 1, projection='3d')
        particles_pos = pso_evol[iter_idx]
        ax.scatter(particles_pos[:, 0], particles_pos[:, 1], particles_pos[:, 2],
                  c='#1f77b4', s=50, alpha=0.6, marker='o', edgecolors='navy', linewidth=0.5)
        ax.scatter(target[0], target[1], target[2],
                  color='green', s=150, marker='*', zorder=5, edgecolors='darkgreen', linewidth=2)
        ax.set_title(f"PSO - Iter {iter_idx+1}/{n_pso}", fontsize=9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 10)
        ax.set_zlim(0, 10)
    
    # AG
    for idx_plot, gen_idx in enumerate(indices_ag):
        ax = fig.add_subplot(2, 5, idx_plot + 6, projection='3d')
        population_pos = ag_evol[gen_idx]
        ax.scatter(population_pos[:, 0], population_pos[:, 1], population_pos[:, 2],
                  c='#d62728', s=50, alpha=0.6, marker='s', edgecolors='darkred', linewidth=0.5)
        ax.scatter(target[0], target[1], target[2],
                  color='green', s=150, marker='*', zorder=5, edgecolors='darkgreen', linewidth=2)
        ax.set_title(f"AG - Gen {gen_idx+1}/{n_ag}", fontsize=9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 10)
        ax.set_zlim(0, 10)
    
    fig.suptitle("Comparação: Evolução PSO (azul) vs AG (vermelho) - Primeira Execução", 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig_comparacao_evolucao_pso_ag.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"  → {fig_path} salva")

# ─────────────────────────────────────────────────────────────────────────────
# 17. PONTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nINICIANDO EXPERIMENTO COMPLETO")
    
    resultados_pso, resultados_ag, hist_pso_evol, hist_ag_evol = rodar_experimento()
    
    print("\n\n─── TABELA COMPARATIVA ───")
    imprimir_tabela(resultados_pso, resultados_ag)
    
    print("\n\n─── GERANDO GRÁFICOS ───")
    gerar_graficos(resultados_pso, resultados_ag)
    
    print("\n─── GERANDO ANIMAÇÃO 3D ───")
    gerar_animacao_3d(resultados_pso, resultados_ag)
    
    print("\n─── GERANDO GRÁFICO 3D ESTÁTICO ───")
    gerar_grafico_3d_estático(resultados_pso, resultados_ag)
    
    print("\n─── GERANDO EVOLUÇÃO 3D PSO ───")
    gerar_grafico_evolucao_pso(hist_pso_evol, TRAJETORIA[0])
    
    print("\n─── GERANDO EVOLUÇÃO 3D AG ───")
    gerar_grafico_evolucao_ag(hist_ag_evol, TRAJETORIA[0])
    
    print("\n─── GERANDO COMPARAÇÃO EVOLUÇÃO 3D ───")
    gerar_grafico_comparacao_evolucao(hist_pso_evol, hist_ag_evol, TRAJETORIA[0])
    
    print("\n✓ Experimento concluído.")