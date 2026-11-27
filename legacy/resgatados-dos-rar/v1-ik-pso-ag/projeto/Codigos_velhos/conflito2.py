import numpy as np
import scipy

# Compatibilidade: versões recentes do SciPy removeram `scipy.randn`,
# mas o `roboticstoolbox` ainda tenta importar esse símbolo.
if not hasattr(scipy, "randn"):
    scipy.randn = np.random.randn

import roboticstoolbox as rtb

import matplotlib.pyplot as plt
import time
from sko.PSO import PSO

# =====================================================================
# 1. MODELAGEM DO ROBÔ BIMO
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

# =====================================================================
# 2. FÁBRICA DA FUNÇÃO FITNESS MULTI-OBJETIVO
# =====================================================================
def criar_funcao_fitness(alvo_xyz, q_anterior, peso_posicao=1.0, peso_juntas=0.3):
    def fitness(q):
        posicao_atual = cinematica_direta(q)
        erro_posicao = np.linalg.norm(posicao_atual - alvo_xyz)
        erro_juntas = np.linalg.norm(q - q_anterior)
        # Penaliza o erro de posição e os solavancos nas juntas
        #penaliza pela distância ao alvo e pela mudança brusca nos ângulos das juntas
        #penalidade_juntas = np.sum(np.abs(q - q_anterior))
        #return (peso_posicao * erro_posicao) + (peso_juntas * penalidade_juntas)
        return (peso_posicao * erro_posicao) + (peso_juntas * erro_juntas)
    return fitness

# =====================================================================
# 3. GERAÇÃO DA TRAJETÓRIA (Uma linha reta no espaço 3D)
# =====================================================================
passos = 30 # Quantidade de pontos na trajetória
x_coords = np.linspace(150, 300, passos)  # Move-se de X=150 para X=300
y_coords = np.linspace(100, -100, passos) # Move-se de Y=100 para Y=-100
z_coords = np.linspace(200, 200, passos)  # Mantém a altura fixa em Z=200
trajetoria_alvos = np.vstack((x_coords, y_coords, z_coords)).T

# =====================================================================
# 4. O LOOP PRINCIPAL COM DOPING (WARM-START)
# =====================================================================
if __name__ == "__main__":
    q_anterior = np.zeros(6) # Começa com o robô na posição inicial (Zero)
    limites = np.pi # Limite de -180 a 180 graus
    
    # Listas para guardar os dados para os gráficos do artigo
    historico_q = []
    historico_erro = []
    
    print("A iniciar cálculo da trajetória com PSO + Doping...\n")
    
    for i, alvo in enumerate(trajetoria_alvos):
        # 1. Cria o juiz para o ponto atual da trajetória
        juiz = criar_funcao_fitness(alvo, q_anterior)
        
        # 2. Inicializa o PSO (30 partículas, 100 iterações é suficiente com doping)
        pso = PSO(func=juiz, n_dim=6, pop=30, max_iter=100, 
                  lb=[-limites]*6, ub=[limites]*6)
        
        # === A MÁGICA DO DOPING (WARM-START) ===
        # A matriz pso.X guarda a posição inicial de todas as partículas.
        # Nós substituímos as primeiras partículas pela posição onde o robô já está!
        pso.X[0] = q_anterior 
        pso.X[1] = q_anterior + np.random.uniform(-0.01, 0.01, 6) # Uma micro mutação
        pso.X[2] = q_anterior - np.random.uniform(-0.01, 0.01, 6) # Outra micro mutação
        # =======================================

        # 3. Executa o algoritmo
        melhor_q, melhor_erro = pso.run()
        
        # 4. Regista os dados e ATUALIZA o q_anterior para o próximo ciclo
        historico_q.append(melhor_q)
        historico_erro.append(melhor_erro[0])
        q_anterior = melhor_q # <--- O passo mais importante para a continuidade!
        
        print(f"Ponto {i+1}/{passos} | Alvo: {np.round(alvo,1)} | Erro: {melhor_erro[0]:.4f}")

    # =====================================================================
    # 5. GERAÇÃO DOS GRÁFICOS CIENTÍFICOS PARA O ARTIGO (CBA)
    # =====================================================================
    historico_q = np.array(historico_q)
    historico_pos = np.array([cinematica_direta(q) for q in historico_q])
    historico_erro = np.array(historico_erro)
    indice_melhor = int(np.argmin(historico_erro))
    
    # Gráfico 1: Comportamento das Juntas ao Longo da Trajetória
    plt.figure(figsize=(10, 5))
    for junta in range(6):
        # Convertendo radianos para graus para ficar mais legível no artigo
        plt.plot(np.degrees(historico_q[:, junta]), label=f'Junta {junta+1}', marker='o', markersize=3)
        
    plt.title('Movimento das Juntas durante a Trajetória (PSO com Doping)')
    plt.xlabel('Pontos da Trajetória')
    plt.ylabel('Ângulo (Graus)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('grafico_juntas_suaves.png', dpi=300) # Salva com qualidade para o artigo
    plt.show()

    # Gráfico 2: Erro de Rastreamento (Fitness)
    plt.figure(figsize=(8, 4))
    plt.plot(historico_erro, color='red', linewidth=2, marker='s', markersize=4)
    plt.title('Erro de Convergência (Função Multi-Objetivo) por Ponto')
    plt.xlabel('Pontos da Trajetória')
    plt.ylabel('Valor do Erro (Fitness)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('grafico_erro_trajetoria.png', dpi=300)
    plt.show()

    # Gráfico 3: Melhor movimento da garra no espaço (3D)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Trajetória desejada (alvos)
    ax.plot(
        trajetoria_alvos[:, 0], trajetoria_alvos[:, 1], trajetoria_alvos[:, 2],
        '--', color='gray', linewidth=1.6, label='Trajetória desejada'
    )

    # Trajetória alcançada pela melhor solução em cada ponto
    ax.plot(
        historico_pos[:, 0], historico_pos[:, 1], historico_pos[:, 2],
        '-', color='tab:blue', linewidth=2.2, marker='o', markersize=3,
        label='Melhor movimento (PSO + Doping)'
    )

    # Destaque do melhor ponto (menor erro)
    ax.scatter(
        historico_pos[indice_melhor, 0],
        historico_pos[indice_melhor, 1],
        historico_pos[indice_melhor, 2],
        color='tab:green', s=90, marker='*',
        label=f'Melhor ponto (erro={historico_erro[indice_melhor]:.4f})'
    )

    # Marca alvo inicial e final para referência visual
    ax.scatter(*trajetoria_alvos[0], color='tab:orange', marker='^', s=60, label='Alvo inicial')
    ax.scatter(*trajetoria_alvos[-1], color='tab:red', marker='x', s=80, label='Alvo final')

    ax.set_title('Melhor Movimento da Garra no Espaço 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('grafico_melhor_movimento_3d.png', dpi=300)
    plt.show()