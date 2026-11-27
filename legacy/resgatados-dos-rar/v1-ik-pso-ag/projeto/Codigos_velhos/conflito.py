import numpy as np
import scipy

# Compatibilidade: versões recentes do SciPy removeram `scipy.randn`,
# mas o `roboticstoolbox` ainda tenta importar esse símbolo.
if not hasattr(scipy, "randn"):
    scipy.randn = np.random.randn

import roboticstoolbox as rtb
import time
import spatialmath as sm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sko.GA import GA
from sko.PSO import PSO

# =====================================================================
# 1. MODELAGEM DO ROBÔ BIMO (D-H Validado)
# =====================================================================
bimo = rtb.DHRobot([
    rtb.RevoluteDH(d=184, a=0,   alpha=np.pi/2),   # Junta 1
    rtb.RevoluteDH(d=0,   a=222, alpha=0),         # Junta 2
    rtb.RevoluteDH(d=0,   a=205, alpha=0),         # Junta 3
    rtb.RevoluteDH(d=0,   a=0,   alpha=-np.pi/2),  # Junta 4
    rtb.RevoluteDH(d=0,   a=0,   alpha=np.pi/2),   # Junta 5
    rtb.RevoluteDH(d=75,  a=0,   alpha=0)          # Junta 6
], name="BIMO")

def cinematica_direta(q):
    return bimo.fkine(q).t


def posicoes_dos_elos(q):
    transformacao = sm.SE3()
    pontos = [transformacao.t]

    for link, angulo in zip(bimo.links, q):
        transformacao = transformacao * link.A(angulo)
        pontos.append(transformacao.t)

    return np.array(pontos)


def para_escalar(valor):
    return float(np.asarray(valor).reshape(-1)[0])


def salvar_animacao_trajetoria(trajetoria_q, alvo_xyz, nome_arquivo, titulo, fps=12):
    trajetoria_q = np.asarray(trajetoria_q, dtype=float)
    if trajetoria_q.ndim == 1:
        trajetoria_q = trajetoria_q.reshape(1, -1)

    caminhos = [posicoes_dos_elos(q) for q in trajetoria_q]
    pontas = np.array([caminho[-1] for caminho in caminhos])
    erros = np.linalg.norm(pontas - alvo_xyz, axis=1)
    pontos = np.concatenate(caminhos, axis=0)

    limite_xy = np.max(np.abs(pontos[:, :2])) + 50
    limite_z = max(200, np.max(pontos[:, 2]) + 50)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    linha, = ax.plot([], [], [], "o-", lw=2, color="tab:blue")
    rastro, = ax.plot([], [], [], "-", lw=1.6, color="tab:orange", alpha=0.85)
    alvo, = ax.plot([], [], [], "x", markersize=10, color="tab:red")
    ponta_atual, = ax.plot([], [], [], "o", markersize=7, color="tab:green")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-limite_xy, limite_xy)
    ax.set_ylim(-limite_xy, limite_xy)
    ax.set_zlim(0, limite_z)

    def atualizar(frame_index):
        pontos_frame = caminhos[frame_index]
        linha.set_data(pontos_frame[:, 0], pontos_frame[:, 1])
        linha.set_3d_properties(pontos_frame[:, 2])

        rastro_pontas = pontas[: frame_index + 1]
        rastro.set_data(rastro_pontas[:, 0], rastro_pontas[:, 1])
        rastro.set_3d_properties(rastro_pontas[:, 2])

        ponta_atual.set_data([rastro_pontas[-1, 0]], [rastro_pontas[-1, 1]])
        ponta_atual.set_3d_properties([rastro_pontas[-1, 2]])

        ax.set_title(
            f"{titulo} | Frame {frame_index + 1}/{len(caminhos)} | Erro: {erros[frame_index]:.2f}"
        )
        return linha, rastro, alvo, ponta_atual

    alvo.set_data([alvo_xyz[0]], [alvo_xyz[1]])
    alvo.set_3d_properties([alvo_xyz[2]])

    animacao = FuncAnimation(fig, atualizar, frames=len(caminhos), interval=1000 / fps, blit=False)
    animacao.save(nome_arquivo, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return nome_arquivo


def salvar_plot_comparativo(trajetoria_ga, trajetoria_pso, alvo_xyz, nome_arquivo="trajetorias_ga_pso.png"):
    traj_ga = np.asarray(trajetoria_ga, dtype=float)
    traj_pso = np.asarray(trajetoria_pso, dtype=float)

    pontas_ga = np.array([cinematica_direta(q) for q in traj_ga])
    pontas_pso = np.array([cinematica_direta(q) for q in traj_pso])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pontas_ga[:, 0], pontas_ga[:, 1], pontas_ga[:, 2], "-", lw=2, color="tab:blue", label="GA")
    ax.plot(pontas_pso[:, 0], pontas_pso[:, 1], pontas_pso[:, 2], "-", lw=2, color="tab:orange", label="PSO")

    ax.scatter(*pontas_ga[0], color="tab:blue", marker="o", s=50)
    ax.scatter(*pontas_pso[0], color="tab:orange", marker="o", s=50)
    ax.scatter(*pontas_ga[-1], color="tab:blue", marker="^", s=70)
    ax.scatter(*pontas_pso[-1], color="tab:orange", marker="^", s=70)
    ax.scatter(*alvo_xyz, color="tab:red", marker="x", s=110, label="Alvo")

    todos = np.vstack([pontas_ga, pontas_pso, alvo_xyz.reshape(1, 3)])
    limite_xy = np.max(np.abs(todos[:, :2])) + 50
    limite_z = max(200, np.max(todos[:, 2]) + 50)
    ax.set_xlim(-limite_xy, limite_xy)
    ax.set_ylim(-limite_xy, limite_xy)
    ax.set_zlim(0, limite_z)

    ax.set_title("Trajetórias da Melhor Solução por Iteração (GA vs PSO)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(nome_arquivo, dpi=150)
    plt.close(fig)
    return nome_arquivo

# =====================================================================
# 2. FÁBRICA DA FUNÇÃO FITNESS MULTI-OBJETIVO
# =====================================================================
# O scikit-opt exige uma função que receba apenas 'q' (os ângulos).
# Usamos essa "fábrica" para injetar o alvo e a posição anterior.
def criar_funcao_fitness(alvo_xyz, q_anterior, peso_posicao=1.0, peso_juntas=0.1):
    
    def fitness(q):
        # 1. Erro de Posição (Chegar no alvo)
        posicao_atual = cinematica_direta(q)
        erro_posicao = np.linalg.norm(posicao_atual - alvo_xyz)
        
        # 2. Erro de Esforço (Minimizar o movimento em relação ao ponto anterior)
        erro_juntas = np.linalg.norm(q - q_anterior)
        
        # O objetivo do algoritmo é fazer esse valor final chegar o mais perto de ZERO possível
        return (peso_posicao * erro_posicao) + (peso_juntas * erro_juntas)
    
    return fitness

# =====================================================================
# 3. CONFIGURAÇÃO DA ARENA (AG vs PSO)
# =====================================================================
if __name__ == "__main__":
    # --- Cenário do Teste ---
    # Queremos que o robô vá da posição "Zero" para um alvo específico
    q_inicial_robo = np.zeros(6) # Posição anterior (onde o robô está agora)
    alvo_desejado = np.array([250.0, 150.0, 200.0]) # Coordenada (x, y, z) alvo
    
    # Criamos a nossa função "Juiz" específica para este movimento
    juiz_fitness = criar_funcao_fitness(alvo_desejado, q_inicial_robo)
    
    # Limites físicos das juntas (Vamos assumir de -180 a +180 graus para todas)
    # No scikit-opt, precisamos passar os limites inferiores (lb) e superiores (ub)
    limite_inferior = [-np.pi] * 6
    limite_superior = [np.pi] * 6
    
    # Parâmetros de Justiça (Ambos terão a mesma "energia" para gastar)
    tamanho_populacao = 50
    max_iteracoes = 150

    print(f"--- INICIANDO DUELO BIMO: ALVO {alvo_desejado} ---\n")

    # ---------------------------------------------------------
    # TESTE 1: ALGORITMO GENÉTICO (GA)
    # ---------------------------------------------------------
    print(">> Executando Algoritmo Genético (GA)...")
    tempo_inicio_ga = time.time()
    
    ga = GA(func=juiz_fitness, n_dim=6, size_pop=tamanho_populacao, max_iter=max_iteracoes, 
            prob_mut=0.1, lb=limite_inferior, ub=limite_superior)
    
    melhor_q_ga, melhor_erro_ga = ga.run()
    trajetoria_q_ga = np.array(ga.generation_best_X, dtype=float)
    tempo_total_ga = time.time() - tempo_inicio_ga
    melhor_q_ga = np.asarray(melhor_q_ga, dtype=float).reshape(-1)
    pos_final_ga = cinematica_direta(melhor_q_ga)
    
    print(f"GA - Tempo: {tempo_total_ga:.3f} s")
    print(f"GA - Erro Final (Fitness): {para_escalar(melhor_erro_ga):.2f}")
    print(f"GA - Posição Alcançada: {np.round(pos_final_ga, 2)}\n")

    # ---------------------------------------------------------
    # TESTE 2: PARTICLE SWARM OPTIMIZATION (PSO)
    # ---------------------------------------------------------
    print(">> Executando Enxame de Partículas (PSO)...")
    tempo_inicio_pso = time.time()
    
    pso = PSO(func=juiz_fitness, n_dim=6, pop=tamanho_populacao, max_iter=max_iteracoes, 
              lb=limite_inferior, ub=limite_superior)

    historico_gbest_pso = [np.asarray(pso.gbest_x, dtype=float).reshape(-1)]
    update_gbest_original = pso.update_gbest

    def update_gbest_com_historico():
        update_gbest_original()
        historico_gbest_pso.append(np.asarray(pso.gbest_x, dtype=float).reshape(-1).copy())

    pso.update_gbest = update_gbest_com_historico
    
    melhor_q_pso, melhor_erro_pso = pso.run()
    trajetoria_q_pso = np.array(historico_gbest_pso, dtype=float)
    tempo_total_pso = time.time() - tempo_inicio_pso
    melhor_q_pso = np.asarray(melhor_q_pso, dtype=float).reshape(-1)
    pos_final_pso = cinematica_direta(melhor_q_pso)
    
    print(f"PSO - Tempo: {tempo_total_pso:.3f} s")
    print(f"PSO - Erro Final (Fitness): {para_escalar(melhor_erro_pso):.2f}")
    print(f"PSO - Posição Alcançada: {np.round(pos_final_pso, 2)}\n")
    
    # --- RESULTADO RÁPIDO ---
    print("=== PLACAR FINAL ===")
    vencedor_tempo = "PSO" if tempo_total_pso < tempo_total_ga else "GA"
    vencedor_erro = "PSO" if para_escalar(melhor_erro_pso) < para_escalar(melhor_erro_ga) else "GA"
    print(f"Mais Rápido: {vencedor_tempo}")
    print(f"Mais Preciso: {vencedor_erro}")

    # ---------------------------------------------------------
    # 4. PLOTAGEM DAS MELHORES TRAJETÓRIAS (BASEADO NO coisa.py)
    # ---------------------------------------------------------
    print("\n>> Gerando plotagens das melhores trajetórias...")

    caminho_gif_ga = salvar_animacao_trajetoria(
        trajetoria_q_ga,
        alvo_desejado,
        nome_arquivo="trajetoria_melhor_ga.gif",
        titulo="GA - Melhor solução por geração",
    )

    caminho_gif_pso = salvar_animacao_trajetoria(
        trajetoria_q_pso,
        alvo_desejado,
        nome_arquivo="trajetoria_melhor_pso.gif",
        titulo="PSO - Melhor solução global por iteração",
    )

    caminho_plot = salvar_plot_comparativo(
        trajetoria_q_ga,
        trajetoria_q_pso,
        alvo_desejado,
        nome_arquivo="trajetorias_ga_pso.png",
    )

    print(f"GIF GA salvo em: {caminho_gif_ga}")
    print(f"GIF PSO salvo em: {caminho_gif_pso}")
    print(f"Plot comparativo salvo em: {caminho_plot}")