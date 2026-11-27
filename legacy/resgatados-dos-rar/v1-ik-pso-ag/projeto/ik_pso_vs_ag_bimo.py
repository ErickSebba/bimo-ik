"""
=============================================================================
  CINEMÁTICA INVERSA DO ROBÔ BIMO (6-DOF)
  Comparativo Experimental: PSO com Warm-Start vs Algoritmo Genético
  
  Artigo: "Comparativo entre PSO e AG na Cinemática Inversa de um
          Braço Robótico Educacional de 6-DoF"
  
  Autores: [Seus nomes aqui]
  
  Referências principais:
    - Luo et al. (2022) – MOFOPSO, Front. Neurorobot. 16:791796
    - Farnese et al. – AG para IK de robô 6-DoF (IRB120)
    - Duymazlar & Gönenli – GA vs PSO para delta robot IK (Cap. 17)
=============================================================================

ANÁLISE CRÍTICA DO CÓDIGO ORIGINAL (resumo):
  1. [CORRIGIDO] A fitness usava apenas erro de POSIÇÃO (3 graus) e penalidade
     de juntas – correto para trajetória sem restrição de orientação, mas os
     PESOS não eram normalizados, causando dominância do termo de posição e
     comportamento imprevisível nos últimos pontos.
  2. [CORRIGIDO] Sem limites articulares por junta – o original usava ±π
     uniforme para todas as 6 juntas, ignorando a mecânica real do BIMO.
  3. [CORRIGIDO] Sem múltiplas execuções – impossível fazer análise estatística
     (média, desvio-padrão) exigida em artigos científicos comparativos.
  4. [ADICIONADO] Implementação do AG com estrutura simétrica ao PSO (mesma
     função fitness, mesmos limites, mesmo doping) para comparação justa.
  5. [CORRIGIDO] Métricas ampliadas: RMSE de posição, erro máximo, suavidade
     das juntas (Σ|Δθ|), tempo de execução, taxa de convergência.
  6. [ADICIONADO] Semente aleatória controlada para reprodutibilidade.
  7. [CORRIGIDO] Gráficos de comparação direta (erro PSO vs AG sobrepostos).
"""

import numpy as np
import scipy
if not hasattr(scipy, "randn"):
    scipy.randn = np.random.randn

import roboticstoolbox as rtb
import spatialmath as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import warnings
import os
warnings.filterwarnings("ignore")

from sko.PSO import PSO

# Criar pasta de saída
OUTPUT_DIR = "OutputsResultados"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO GLOBAL DO EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────
N_RUNS       = 5      # execuções independentes por algoritmo (rigor estatístico)
N_PASSOS     = 30      # pontos na trajetória
PSO_POP      = 40      # partículas PSO  (≥30 recomendado pela literatura)
PSO_ITER     = 150     # iterações PSO   (aumentado para reduzir erro final)
PSO_NOVO_POP = 60      # partículas PSO nova (exploração maior)
PSO_NOVO_ITER = 80     # iterações PSO nova (mais rápido)
AG_POP       = 40      # indivíduos AG   (equivalente ao PSO para comparação justa)
AG_GENS      = 150     # gerações AG
TAXA_CROSS   = 0.7     # probabilidade de crossover (Farnese et al. usaram 0.5–0.8)
TAXA_MUT     = 0.15    # probabilidade de mutação por gene
PESO_POS     = 1.0     # peso do erro de posição na fitness  [normalizado]
PESO_JTA     = 0.05    # peso da penalidade articular  ← CORRIGIDO: era 0.3, causava
                       # o algoritmo sacrificar precisão para suavizar demais
SEED_BASE    = 42      # semente-base; cada run usa SEED_BASE + run_id

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODELAGEM DO ROBÔ BIMO  (parâmetros DH)
#    NOTA: Os parâmetros abaixo foram mantidos do código original.
#    Para o artigo, verificar no manual físico do BIMO os limites reais de
#    cada junta. Os limites ±π são conservadores – recomenda-se ajustar.
# ─────────────────────────────────────────────────────────────────────────────
bimo = rtb.DHRobot([
    rtb.RevoluteDH(d=184, a=0,   alpha=np.pi/2),   # Junta 1
    rtb.RevoluteDH(d=0,   a=222, alpha=0),          # Junta 2
    rtb.RevoluteDH(d=0,   a=205, alpha=0),          # Junta 3
    rtb.RevoluteDH(d=0,   a=0,   alpha=-np.pi/2),   # Junta 4
    rtb.RevoluteDH(d=0,   a=0,   alpha=np.pi/2),    # Junta 5
    rtb.RevoluteDH(d=75,  a=0,   alpha=0)           # Junta 6
], name="BIMO")

# Limites articulares individuais (rad)
# RECOMENDAÇÃO: substituir pelos valores reais do manual do BIMO
# Exemplo inspirado no IRB120 (Farnese et al.) convertido para rad:
LIMITES_INF = np.array([-np.pi, -np.pi*110/180, -np.pi*80/180,
                         -np.pi, -np.pi*50/180,  -np.pi])
LIMITES_SUP = np.array([ np.pi,  np.pi*155/180,  np.pi*210/180,
                          np.pi,  np.pi*230/180,   np.pi])

def cinematica_direta(q):
    """Retorna posição XYZ (mm) do efetuador dado o vetor de juntas q (rad)."""
    return bimo.fkine(q).t  # vetor [x, y, z]

# ─────────────────────────────────────────────────────────────────────────────
# 2. FUNÇÃO FITNESS MULTI-OBJETIVO  (compartilhada por PSO e AG)
#
#    F(q) = w_pos * ||p_alvo - p_atual|| + w_jta * ||q - q_anterior||
#
#    Justificativa matemática:
#      - f1: distância euclidiana (mm) entre efetuador e alvo  → Eq.(3) de Farnese
#      - f3: norma euclidiana da variação articular             → Eq.(5) de Farnese
#      - Pesos normalizados: w_pos >> w_jta para garantir que precisão
#        de posição não seja sacrificada por suavidade (erro observado
#        no código original com peso_juntas=0.3).
#
#    IMPORTANTE: Para trajetórias onde a ORIENTAÇÃO é relevante, adicionar
#      f2 = ||e_alvo - e_atual||  (ângulos de Euler)           → Eq.(4) de Farnese
#    e incluí-la na fitness com peso adequado.
# ─────────────────────────────────────────────────────────────────────────────
def obter_matriz_se3(q):
    """Retorna a matriz homogênea 4x4 completa de translação e rotação."""
    max_tentativas = 3
    for tentativa in range(max_tentativas):
        try:
            return bimo.fkine(q).A
        except Exception as exc:
            if tentativa < max_tentativas - 1:
                continue  # Tenta novamente
            else:
                # Se falhar 3 vezes, retorna identidade
                print(f"Aviso: fkine falhou 3x para q={q}")
                return np.eye(4)

def criar_fitness(matriz_alvo, q_anterior, 
                                peso_trans=1.0, peso_rot=0.5, peso_junta=0.005):
    """
    Constrói a função de aptidão em tempo de execução incorporando:
    1. Distância Euclidiana (mm) para controle translacional.
    2. Distância de Traço Matricial (radianos) para controle orientacional.
    3. Fator de Regularização para minimizar a descontinuidade articular.
    
    NOTA: O peso_junta é significativamente reduzido para promover
          o dimensionamento escalar correto frente às grandezas em mm.
    """
    # Aceita dois formatos de alvo para manter compatibilidade:
    # 1) vetor XYZ (shape=(3,))  -> otimização apenas translacional
    # 2) matriz homogênea 4x4     -> otimização translacional + orientacional
    alvo = np.asarray(matriz_alvo)
    if alvo.ndim == 1:
        if alvo.shape[0] < 3:
            raise ValueError("alvo 1D deve ter ao menos 3 elementos (x, y, z)")
        posicao_alvo = alvo[:3]
        rotacao_alvo = None
    elif alvo.ndim == 2 and alvo.shape[0] >= 3 and alvo.shape[1] >= 4:
        posicao_alvo = alvo[:3, 3]
        rotacao_alvo = alvo[:3, :3]
    else:
        raise ValueError(
            "matriz_alvo deve ser um vetor XYZ (3,) ou matriz homogênea 4x4"
        )
    
    def fitness_computada(q):
        matriz_atual = obter_matriz_se3(q)
        posicao_atual = matriz_atual[:3, 3]
        rotacao_atual = matriz_atual[:3, :3]
        
        # Erro Translacional Euclidiano (L2 Norm)
        erro_cartesiano = np.linalg.norm(posicao_atual - posicao_alvo)
        
        # Erro de Orientação Computada via Traço (Trace)
        # Extrai o ângulo real de diferença entre duas matrizes SO(3).
        # A limitação do clip previne NaNs ocasionados por falhas de precisão do float64.
        if rotacao_alvo is None:
            erro_angular = 0.0
        else:
            matriz_diferenca = np.dot(rotacao_alvo.T, rotacao_atual)
            traco_operativo = np.clip((np.trace(matriz_diferenca) - 1.0) / 2.0, -1.0, 1.0)
            erro_angular = np.arccos(traco_operativo)
        
        # Penalidade Evolutiva para preservação da inércia articular
        erro_continuo = np.linalg.norm(q - q_anterior)
        
        # Agregação Multiobjetivo Normalizada (Escalarização)
        # O erro angular (em radianos, máx ~3.14) é multiplicador por 100 para competir
        # de forma equitativa na busca topológica contra a variação milimétrica espacial.
        custo_global = (peso_trans * erro_cartesiano) + \
                       (peso_rot * erro_angular * 100.0) + \
                       (peso_junta * erro_continuo * 10.0)
        return custo_global
        
    return fitness_computada
# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAJETÓRIA DE TESTE  (idêntica para PSO e AG)
#    Linha reta no espaço 3D – mesma do código original para manter comparação
#    com resultados já obtidos.
# ─────────────────────────────────────────────────────────────────────────────
x_traj = np.linspace(150, 300, N_PASSOS)
y_traj = np.linspace(100, -100, N_PASSOS)
z_traj = np.ones(N_PASSOS) * 200.0
TRAJETORIA = np.column_stack([x_traj, y_traj, z_traj])

# ─────────────────────────────────────────────────────────────────────────────
# 4. ALGORITMO PSO COM WARM-START (DOPING)
# ─────────────────────────────────────────────────────────────────────────────
# def rodar_pso(seed=42, verbose=False):
#     """
#     Executa PSO com doping ao longo da trajetória completa.
    
#     Returns:
#         historico_q    : (N_PASSOS, 6) ângulos ótimos por ponto
#         historico_erro : (N_PASSOS,)   fitness ótima por ponto 
#         tempo_total    : float  segundos
#     """
#     np.random.seed(seed)
#     q_ant = np.zeros(6)
#     hist_q    = []
#     hist_erro = []
    
#     t0 = time.time()
#     for i, alvo in enumerate(TRAJETORIA):
#         juiz = criar_fitness(alvo, q_ant)
        
#         pso = PSO(func=juiz, n_dim=6, pop=40, max_iter=200,
#                   lb=LIMITES_INF.tolist(), ub=LIMITES_SUP.tolist()) #func=juiz, n_dim=6, pop=PSO_POP, max_iter=PSO_ITER, lb=LIMITES_INF.tolist(), ub=LIMITES_SUP.tolist()
        
#         # === DOPING (warm-start) ===
#         # Planta solução anterior + variações pequenas como partículas iniciais.
#         # Técnica validada em Farnese et al. – garantiu 100% convergência.
#         pso.X[0] = np.clip(q_ant, LIMITES_INF, LIMITES_SUP)
#         for j in range(1, 6):
#             pso.X[j] = np.clip(q_ant + np.random.uniform(-0.05, 0.05, 6),
#                                 LIMITES_INF, LIMITES_SUP)
#         # # ===========================
        
#         melhor_q, melhor_err = pso.run()
#         hist_q.append(melhor_q.copy())
#         hist_erro.append(float(melhor_err[0]))
#         q_ant = melhor_q.copy()
        
#         if verbose:
#             print(f"  PSO ponto {i+1:2d}/{N_PASSOS} | "
#                   f"alvo={np.round(alvo,1)} | erro={melhor_err[0]:.4f} mm")
    
#     tempo = time.time() - t0
#     return np.array(hist_q), np.array(hist_erro), tempo

#PSO2
# ─────────────────────────────────────────────────────────────────────────────
# 4. ALGORITMO PSO COM WARM-START (DOPING) - VERSÃO ORIGINAL
# ─────────────────────────────────────────────────────────────────────────────
def rodar_pso(seed=42, verbose=False):
    """
    Executa PSO com doping ao longo da trajetória completa.
    
    Returns:
        historico_q    : (N_PASSOS, 6) ângulos ótimos por ponto
        historico_erro : (N_PASSOS,)   fitness ótima por ponto
        tempo_total    : float  segundos
    """
    np.random.seed(seed)
    q_ant = np.zeros(6)
    hist_q    = []
    hist_erro = []
    
    t0 = time.time()
    for i, alvo in enumerate(TRAJETORIA):
        juiz = criar_fitness(alvo, q_ant)
        
        pso = PSO(func=juiz, n_dim=6, pop=PSO_POP, max_iter=PSO_ITER,
                  lb=LIMITES_INF.tolist(), ub=LIMITES_SUP.tolist())
        
        # === DOPING (warm-start) ===
        pso.X[0] = np.clip(q_ant, LIMITES_INF, LIMITES_SUP)
        pso.X[1] = np.clip(q_ant + np.random.uniform(-0.05, 0.05, 6),
                            LIMITES_INF, LIMITES_SUP)
        pso.X[2] = np.clip(q_ant - np.random.uniform(-0.05, 0.05, 6),
                            LIMITES_INF, LIMITES_SUP)
        
        melhor_q, melhor_err = pso.run()
        hist_q.append(melhor_q.copy())
        hist_erro.append(float(melhor_err[0]))
        q_ant = melhor_q.copy()
        
        if verbose:
            print(f"  PSO ponto {i+1:2d}/{N_PASSOS} | "
                  f"alvo={np.round(alvo,1)} | erro={melhor_err[0]:.4f} mm")
    
    tempo = time.time() - t0
    return np.array(hist_q), np.array(hist_erro), tempo


# ─────────────────────────────────────────────────────────────────────────────
# 4b. ALGORITMO PSO CUSTOMIZADO (NOVO) - COM EXPLORAÇÃO/EXPLORAÇÃO BALANCEADA
# ─────────────────────────────────────────────────────────────────────────────
def rodar_pso_novo(seed=42, verbose=False):
    """
    Executa PSO customizado com inicialização inteligente (exploração/refinamento).
    Parâmetros tunados: w=0.5, c1=1.5, c2=1.5, pop=60, iter=80
    
    Returns:
        historico_q    : (N_PASSOS, 6) ângulos ótimos por ponto
        historico_erro : (N_PASSOS,)   fitness ótima por ponto
        tempo_total    : float  segundos
    """
    np.random.seed(seed)
    q_ant = np.zeros(6)
    hist_q    = []
    hist_erro = []
    
    w, c1, c2 = 0.5, 1.5, 1.5
    num_particles = PSO_NOVO_POP
    max_iter = PSO_NOVO_ITER
    
    t0 = time.time()
    for i, alvo in enumerate(TRAJETORIA):
        juiz = criar_fitness(alvo, q_ant)
        
        # 🔥 INICIALIZAÇÃO INTELIGENTE
        if i == 0:
            # PRIMEIRA EXECUÇÃO → mais exploração
            n_global = int(num_particles * 0.6)
            n_local  = num_particles - n_global
            
            global_particles = np.random.uniform(LIMITES_INF, LIMITES_SUP, (n_global, 6))
            local_particles  = np.random.uniform(-1, 1, (n_local, 6))
            particles = np.vstack([global_particles, local_particles])
        else:
            # EXECUÇÕES SEGUINTES → refinamento dominante
            n_local  = int(num_particles * 0.5)
            n_global = num_particles - n_local
            
            local_particles  = q_ant + np.random.normal(0, 0.1, (n_local, 6))
            local_particles  = np.clip(local_particles, LIMITES_INF, LIMITES_SUP)
            global_particles = np.random.uniform(LIMITES_INF, LIMITES_SUP, (n_global, 6))
            
            particles = np.vstack([local_particles, global_particles])
        
        np.random.shuffle(particles)
        
        velocities = np.zeros((num_particles, 6))
        pbest = particles.copy()
        pbest_scores = np.array([juiz(p) for p in particles])
        
        gbest = pbest[np.argmin(pbest_scores)]
        gbest_score = np.min(pbest_scores)
        
        # Iterações PSO
        for it in range(max_iter):
            for j in range(num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                
                velocities[j] = (
                    w * velocities[j]
                    + c1 * r1 * (pbest[j] - particles[j])
                    + c2 * r2 * (gbest - particles[j])
                )
                
                particles[j] += velocities[j]
                particles[j] = np.clip(particles[j], LIMITES_INF, LIMITES_SUP)
                
                score = juiz(particles[j])
                
                if score < pbest_scores[j]:
                    pbest[j] = particles[j].copy()
                    pbest_scores[j] = score
            
            best_idx = np.argmin(pbest_scores)
            if pbest_scores[best_idx] < gbest_score:
                gbest = pbest[best_idx].copy()
                gbest_score = pbest_scores[best_idx]
        
        hist_q.append(gbest.copy())
        hist_erro.append(float(gbest_score))
        q_ant = gbest.copy()
        
        if verbose:
            print(f"  PSO_NOVO ponto {i+1:2d}/{N_PASSOS} | "
                  f"alvo={np.round(alvo,1)} | erro={gbest_score:.4f} mm")
    
    tempo = time.time() - t0
    return np.array(hist_q), np.array(hist_erro), tempo


# ─────────────────────────────────────────────────────────────────────────────
# 5. ALGORITMO GENÉTICO COM DOPING
#    Implementação própria para total controle e simetria com o PSO.
#    Estrutura baseada em Farnese et al. (seleção, crossover, mutação).
# ─────────────────────────────────────────────────────────────────────────────

def _inicializar_populacao(n_ind, q_doping=None, seed=None):
    """
    Cria população inicial aleatória com opção de doping.
    
    O primeiro indivíduo (e variações pequenas) recebe a solução anterior,
    espelhando exatamente o warm-start do PSO para comparação justa.
    """
    if seed is not None:
        np.random.seed(seed)
    
    pop = np.random.uniform(LIMITES_INF, LIMITES_SUP, (n_ind, 6))
    
    if q_doping is not None:
        pop[0] = np.clip(q_doping, LIMITES_INF, LIMITES_SUP)
        pop[1] = np.clip(q_doping + np.random.uniform(-0.05, 0.05, 6),
                         LIMITES_INF, LIMITES_SUP)
        pop[2] = np.clip(q_doping - np.random.uniform(-0.05, 0.05, 6),
                         LIMITES_INF, LIMITES_SUP)
    return pop


def _avaliar(pop, fitness_fn):
    """Avalia todos os indivíduos da população."""
    return np.array([fitness_fn(ind) for ind in pop])


def _selecao_roleta(pop, scores):
    """
    Seleção por roleta (proporcional ao fitness inverso).
    Melhor desempenho comparativo que torneio em Farnese et al. (AG4/AG5).
    """
    inv = 1.0 / (scores + 1e-12)
    probs = inv / inv.sum()
    idx = np.random.choice(len(pop), size=len(pop), replace=True, p=probs)
    return pop[idx]


def _crossover_aritmetico(p1, p2, taxa=TAXA_CROSS):
    """
    Crossover aritmético (blend): filho = α*p1 + (1-α)*p2.
    Melhor para espaços contínuos (Farnese et al. AG4 – melhor configuração).
    """
    if np.random.rand() < taxa:
        alpha = np.random.rand()
        filho1 = alpha * p1 + (1 - alpha) * p2
        filho2 = (1 - alpha) * p1 + alpha * p2
    else:
        filho1, filho2 = p1.copy(), p2.copy()
    return filho1, filho2


def _mutacao_gaussiana(ind, taxa=TAXA_MUT, sigma=0.1):
    """
    Mutação gaussiana adaptativa.
    Cada gene muta com probabilidade `taxa`, perturbado por N(0, sigma).
    """
    novo = ind.copy()
    mascara = np.random.rand(6) < taxa
    novo[mascara] += np.random.normal(0, sigma, mascara.sum())
    return np.clip(novo, LIMITES_INF, LIMITES_SUP)


def _elitismo(pop, scores, n_elite=2):
    """Retorna os n_elite melhores indivíduos (garantia de não-regressão)."""
    idx = np.argsort(scores)[:n_elite]
    return pop[idx].copy(), scores[idx].copy()


def rodar_ag(seed=42, verbose=False):
    """
    Executa o AG com doping ao longo da trajetória completa.
    
    Parâmetros espelhados ao PSO para comparação justa:
      - mesmo número de indivíduos/partículas
      - mesmo número de gerações/iterações
      - mesma função fitness
      - mesmo mecanismo de doping
    
    Returns:
        historico_q    : (N_PASSOS, 6)
        historico_erro : (N_PASSOS,)
        tempo_total    : float
    """
    np.random.seed(seed)
    q_ant    = np.zeros(6)
    hist_q   = []
    hist_err = []
    
    t0 = time.time()
    for i, alvo in enumerate(TRAJETORIA):
        fitness_fn = criar_fitness(alvo, q_ant)
        
        # Inicialização com doping
        pop    = _inicializar_populacao(AG_POP, q_doping=q_ant)
        scores = _avaliar(pop, fitness_fn)
        
        if verbose:
            print(f"    [AG INIT] ponto {i+1}/{N_PASSOS}, iniciando evolução...")
        
        for gen in range(AG_GENS):
            # Elitismo: preserva os 2 melhores
            elite_inds, elite_scores = _elitismo(pop, scores, n_elite=2)
            
            # Seleção por roleta
            pais = _selecao_roleta(pop, scores)
            
            # Crossover aritmético + mutação gaussiana → nova geração
            filhos = []
            for j in range(0, AG_POP - 2, 2):
                f1, f2 = _crossover_aritmetico(pais[j], pais[j+1])
                filhos.append(_mutacao_gaussiana(f1))
                filhos.append(_mutacao_gaussiana(f2))
            # Completa se AG_POP for ímpar
            while len(filhos) < AG_POP - 2:
                filhos.append(_mutacao_gaussiana(pais[-1]))
            
            # Monta nova população: elite + filhos
            nova_pop = np.vstack([elite_inds, np.array(filhos[:AG_POP-2])])
            
            # Sobreposição de população: 20% dos piores são substituídos
            # por novas amostras aleatórias (evita estagnação em mínimos locais)
            n_renov = max(1, int(0.20 * AG_POP))
            idx_piores = np.argsort(_avaliar(nova_pop, fitness_fn))[-n_renov:]
            nova_pop[idx_piores] = np.random.uniform(
                LIMITES_INF, LIMITES_SUP, (n_renov, 6))
            
            pop    = nova_pop
            scores = _avaliar(pop, fitness_fn)
            
            if verbose and gen % 50 == 0:
                print(f"      gen {gen:3d}: melhor fitness = {np.min(scores):.6f}")
        
        # Melhor indivíduo da geração final
        idx_melhor = np.argmin(scores)
        melhor_q   = pop[idx_melhor].copy()
        melhor_err = scores[idx_melhor]
        
        hist_q.append(melhor_q)
        hist_err.append(float(melhor_err))
        q_ant = melhor_q.copy()
        
        if verbose:
            print(f"  AG  ponto {i+1:2d}/{N_PASSOS} | "
                  f"alvo={np.round(alvo,1)} | erro={melhor_err:.4f} mm")
    
    tempo = time.time() - t0
    return np.array(hist_q), np.array(hist_err), tempo


# ─────────────────────────────────────────────────────────────────────────────
# 6. MÉTRICAS DE AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
def calcular_metricas(historico_q, historico_erro, historico_pos=None):
    """
    Calcula as métricas científicas para o artigo.
    
    Métricas (baseadas em Luo et al. 2022, Duymazlar & Gönenli, Farnese et al.):
      - RMSE de posição     : raiz do erro quadrático médio (mm)
      - Erro máximo         : pior ponto da trajetória (mm)
      - Suavidade articular : soma total das variações angulares (rad)
                              → menor = mais suave
      - Erro final fitness  : valor da função objetivo no último ponto
    """
    if historico_pos is None:
        historico_pos = np.array([cinematica_direta(q) for q in historico_q])
    
    erros_pos = np.array([
        np.linalg.norm(historico_pos[i] - TRAJETORIA[i])
        for i in range(N_PASSOS)
    ])
    
    rmse        = np.sqrt(np.mean(erros_pos**2))
    erro_max    = np.max(erros_pos)
    erro_medio  = np.mean(erros_pos)
    
    # Suavidade: soma das variações absolutas entre pontos consecutivos
    suavidade = np.sum(np.abs(np.diff(historico_q, axis=0)))
    
    # Fitness final (inclui penalidade articular)
    fitness_medio = np.mean(historico_erro)
    fitness_max   = np.max(historico_erro)
    
    return {
        "rmse_pos_mm"    : rmse,
        "erro_medio_mm"  : erro_medio,
        "erro_max_mm"    : erro_max,
        "suavidade_rad"  : suavidade,
        "fitness_medio"  : fitness_medio,
        "fitness_max"    : fitness_max,
        "erros_pos"      : erros_pos,
        "historico_pos"  : historico_pos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPERIMENTO COMPLETO: N_RUNS execuções de cada algoritmo
# ─────────────────────────────────────────────────────────────────────────────
def rodar_experimento():
    """
    Executa N_RUNS execuções independentes de PSO (antigo), PSO (novo) e AG,
    coletando todas as métricas.
    """
    print("=" * 75)
    print(f"EXPERIMENTO: PSO (antigo) vs PSO (novo) vs AG – {N_RUNS} execuções cada")
    print(f"Trajetória: {N_PASSOS} pontos")
    print(f"  PSO Antigo: {PSO_POP}p, {PSO_ITER}it (w=0.8, c1=0.5, c2=0.5)")
    print(f"  PSO Novo:   {PSO_NOVO_POP}p, {PSO_NOVO_ITER}it (w=0.5, c1=1.5, c2=1.5)")
    print(f"  AG:         {AG_POP}ind, {AG_GENS}ger")
    print("=" * 75)
    
    resultados_pso_antigo = []
    resultados_pso_novo   = []
    resultados_ag         = []
    
    for run in range(N_RUNS):
        seed = SEED_BASE + run
        print(f"\n▶ Run {run+1}/{N_RUNS}  (seed={seed})")
        
        # --- PSO ANTIGO ---
        print("  [PSO-OLD]", end=" ", flush=True)
        hq_pso_old, he_pso_old, t_pso_old = rodar_pso(seed=seed, verbose=False)
        m_pso_old = calcular_metricas(hq_pso_old, he_pso_old)
        m_pso_old["tempo_s"]    = t_pso_old
        m_pso_old["historico_q"] = hq_pso_old
        m_pso_old["historico_erro"] = he_pso_old
        resultados_pso_antigo.append(m_pso_old)
        print(f"RMSE={m_pso_old['rmse_pos_mm']:.3f}mm  "
              f"MaxErr={m_pso_old['erro_max_mm']:.3f}mm  "
              f"t={t_pso_old:.1f}s")
        
        # --- PSO NOVO ---
        print("  [PSO-NEW]", end=" ", flush=True)
        hq_pso_new, he_pso_new, t_pso_new = rodar_pso_novo(seed=seed, verbose=False)
        m_pso_new = calcular_metricas(hq_pso_new, he_pso_new)
        m_pso_new["tempo_s"]    = t_pso_new
        m_pso_new["historico_q"] = hq_pso_new
        m_pso_new["historico_erro"] = he_pso_new
        resultados_pso_novo.append(m_pso_new)
        print(f"RMSE={m_pso_new['rmse_pos_mm']:.3f}mm  "
              f"MaxErr={m_pso_new['erro_max_mm']:.3f}mm  "
              f"t={t_pso_new:.1f}s")
        
        # --- AG ---
        print("  [AG]     ", end=" ", flush=True)
        hq_ag, he_ag, t_ag = rodar_ag(seed=seed, verbose=False)
        m_ag = calcular_metricas(hq_ag, he_ag)
        m_ag["tempo_s"]     = t_ag
        m_ag["historico_q"] = hq_ag
        m_ag["historico_erro"] = he_ag
        resultados_ag.append(m_ag)
        print(f"RMSE={m_ag['rmse_pos_mm']:.3f}mm  "
              f"MaxErr={m_ag['erro_max_mm']:.3f}mm  "
              f"t={t_ag:.1f}s")
    
    return resultados_pso_antigo, resultados_pso_novo, resultados_ag


# ─────────────────────────────────────────────────────────────────────────────
# 8. TABELA COMPARATIVA (saída para o artigo)
# ─────────────────────────────────────────────────────────────────────────────
def imprimir_tabela(resultados_pso_antigo, resultados_pso_novo, resultados_ag):
    """Imprime tabela com média ± desvio-padrão (estilo paper científico)."""
    
    metricas_labels = [
        ("rmse_pos_mm",   "RMSE Posição (mm)"),
        ("erro_medio_mm", "Erro Médio  (mm)"),
        ("erro_max_mm",   "Erro Máximo (mm)"),
        ("suavidade_rad", "Suavidade   (rad)"),
        ("fitness_medio", "Fitness Médio"),
        ("fitness_max",   "Fitness Máximo"),
        ("tempo_s",       "Tempo Exec. (s)"),
    ]
    
    print("\n" + "=" * 80)
    print(f"{'MÉTRICA':<25} {'PSO-OLD (μ ± σ)':>18}  {'PSO-NEW (μ ± σ)':>18}  {'AG  (μ ± σ)':>18}")
    print("=" * 80)
    
    for chave, nome in metricas_labels:
        vals_old = np.array([r[chave] for r in resultados_pso_antigo])
        vals_new = np.array([r[chave] for r in resultados_pso_novo])
        vals_ag  = np.array([r[chave] for r in resultados_ag])
        mu_old, sd_old = vals_old.mean(), vals_old.std()
        mu_new, sd_new = vals_new.mean(), vals_new.std()
        mu_ag,  sd_ag  = vals_ag.mean(),  vals_ag.std()
        print(f"{nome:<25} {mu_old:8.4f} ± {sd_old:.4f}  "
              f"{mu_new:8.4f} ± {sd_new:.4f}  "
              f"{mu_ag:8.4f} ± {sd_ag:.4f}")
    
    print("=" * 80)
    print(f"  N_RUNS={N_RUNS}  |  N_PASSOS={N_PASSOS}")
    print(f"  PSO-OLD: pop={PSO_POP}, iter={PSO_ITER}, w=0.8, c1=0.5, c2=0.5")
    print(f"  PSO-NEW: pop={PSO_NOVO_POP}, iter={PSO_NOVO_ITER}, w=0.5, c1=1.5, c2=1.5 (exploração/refinamento)")
    print(f"  AG: pop={AG_POP}, gen={AG_GENS}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. VISUALIZAÇÕES PARA O ARTIGO
# ─────────────────────────────────────────────────────────────────────────────
def gerar_graficos(resultados_pso_antigo, resultados_pso_novo, resultados_ag):
    """
    Gera figuras comparando os 3 algoritmos: PSO-OLD, PSO-NEW e AG
    """
    
    # Identifica melhor run de cada algoritmo (menor RMSE)
    idx_best_old = int(np.argmin([r["rmse_pos_mm"] for r in resultados_pso_antigo]))
    idx_best_new = int(np.argmin([r["rmse_pos_mm"] for r in resultados_pso_novo]))
    idx_best_ag  = int(np.argmin([r["rmse_pos_mm"] for r in resultados_ag]))
    
    best_old = resultados_pso_antigo[idx_best_old]
    best_new = resultados_pso_novo[idx_best_new]
    best_ag  = resultados_ag[idx_best_ag]
    
    cor_old = "#1f77b4"   # azul
    cor_new = "#2ca02c"   # verde
    cor_ag  = "#d62728"   # vermelho
    cor_traj = "#7f7f7f"  # cinza
    
    # ── FIGURA 1: Trajetória 3D (apenas melhor run de cada) ──────────────────
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    
    ax.plot(*TRAJETORIA.T, "--", color=cor_traj, lw=1.8, label="Trajetória desejada")
    ax.plot(*best_old["historico_pos"].T, "-", color=cor_old, lw=2.2,
            marker="o", ms=3, label=f"PSO-OLD (RMSE={best_old['rmse_pos_mm']:.2f} mm)")
    ax.plot(*best_new["historico_pos"].T, "-", color=cor_new, lw=2.2,
            marker="^", ms=3, label=f"PSO-NEW (RMSE={best_new['rmse_pos_mm']:.2f} mm)")
    ax.plot(*best_ag["historico_pos"].T, "-", color=cor_ag, lw=2.2,
            marker="s", ms=3, label=f"AG   (RMSE={best_ag['rmse_pos_mm']:.2f} mm)")
    
    ax.scatter(*TRAJETORIA[0], color="tab:orange", marker="^", s=70)
    ax.scatter(*TRAJETORIA[-1], color="tab:green",  marker="x", s=90)
    
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("Trajetória 3D: PSO-OLD vs PSO-NEW vs AG (melhor run)")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig1_trajetoria_3d_comparacao.png")
    plt.savefig(fig_path, dpi=300)
    print(f"  → {fig_path} salva")
    plt.close()
    
    # ── FIGURA 2: Erro de rastreamento por ponto ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    pontos = np.arange(1, N_PASSOS + 1)
    
    # Intervalo de confiança (média ± dp) ao longo dos pontos
    erros_old_mat = np.array([r["erros_pos"] for r in resultados_pso_antigo])
    erros_new_mat = np.array([r["erros_pos"] for r in resultados_pso_novo])
    erros_ag_mat  = np.array([r["erros_pos"] for r in resultados_ag])
    
    mu_old = erros_old_mat.mean(axis=0)
    sd_old = erros_old_mat.std(axis=0)
    mu_new = erros_new_mat.mean(axis=0)
    sd_new = erros_new_mat.std(axis=0)
    mu_ag  = erros_ag_mat.mean(axis=0)
    sd_ag  = erros_ag_mat.std(axis=0)
    
    ax.fill_between(pontos, mu_old - sd_old, mu_old + sd_old, alpha=0.2, color=cor_old)
    ax.plot(pontos, mu_old, "-o", ms=4, lw=2, color=cor_old,
            label=f"PSO-OLD μ={mu_old.mean():.2f} mm")
    ax.fill_between(pontos, mu_new - sd_new, mu_new + sd_new, alpha=0.2, color=cor_new)
    ax.plot(pontos, mu_new, "-^", ms=4, lw=2, color=cor_new,
            label=f"PSO-NEW μ={mu_new.mean():.2f} mm")
    ax.fill_between(pontos, mu_ag - sd_ag, mu_ag + sd_ag, alpha=0.2, color=cor_ag)
    ax.plot(pontos, mu_ag,  "-s", ms=4, lw=2, color=cor_ag,
            label=f"AG    μ={mu_ag.mean():.2f} mm")
    
    ax.set_title(f"Erro de Rastreamento por Ponto (N={N_RUNS} execuções, μ ± σ)")
    ax.set_xlabel("Ponto da Trajetória")
    ax.set_ylabel("Erro de Posição (mm)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig2_erro_rastreamento_comparacao.png")
    plt.savefig(fig_path, dpi=300)
    print(f"  → {fig_path} salva")
    plt.close()
    
    # ── FIGURA 3: Boxplot comparativo (RMSE, Suavidade, Tempo) ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    metricas_box = [
        ("rmse_pos_mm",   "RMSE Posição (mm)"),
        ("suavidade_rad", "Suavidade (rad)"),
        ("tempo_s",       "Tempo de Exec. (s)"),
    ]
    
    for ax, (chave, titulo) in zip(axes, metricas_box):
        vals_old = [r[chave] for r in resultados_pso_antigo]
        vals_new = [r[chave] for r in resultados_pso_novo]
        vals_ag  = [r[chave] for r in resultados_ag]
        bp = ax.boxplot([vals_old, vals_new, vals_ag],
                        patch_artist=True, widths=0.6,
                        labels=["PSO-OLD", "PSO-NEW", "AG"],
                        medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor(cor_old + "88")
        bp["boxes"][1].set_facecolor(cor_new + "88")
        bp["boxes"][2].set_facecolor(cor_ag  + "88")
        ax.set_title(titulo, fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    fig.suptitle(f"Comparativo de 3 Algoritmos (N={N_RUNS} execuções independentes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig3_boxplot_comparacao_3algoritmos.png")
    plt.savefig(fig_path, dpi=300)
    print(f"  → {fig_path} salva")
    plt.close()

    gerar_video_trajetoria(best_old, best_new, best_ag)
    gerar_video_movimento_bracos(best_old, best_new, best_ag)


def gerar_video_trajetoria(best_old, best_new, best_ag, arquivo_saida="video_trajetoria_3algoritmos.gif"):
    """
    Gera uma animação da trajetória completa dos 3 algoritmos (PSO-OLD, PSO-NEW, AG).
    """
    trajetoria_old = best_old["historico_pos"]
    trajetoria_new = best_new["historico_pos"]
    trajetoria_ag = best_ag["historico_pos"]
    n_frames = max(len(trajetoria_old), len(trajetoria_new), len(trajetoria_ag), len(TRAJETORIA))

    def _setup_axis(ax, titulo, cor_algo):
        ax.plot(*TRAJETORIA.T, "--", color="#7f7f7f", lw=1.2, alpha=0.7, label="Desejada")
        ax.set_title(titulo)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.scatter(*TRAJETORIA[0], color="tab:orange", marker="^", s=50)
        ax.scatter(*TRAJETORIA[-1], color="tab:green", marker="x", s=60)
        ax.legend(loc="best", fontsize=8)
        ax.set_box_aspect((1, 1, 0.7))
        ax.view_init(elev=24, azim=-55)
        return ax

    fig = plt.figure(figsize=(16, 5))
    ax_old = fig.add_subplot(131, projection="3d")
    ax_new = fig.add_subplot(132, projection="3d")
    ax_ag = fig.add_subplot(133, projection="3d")
    
    _setup_axis(ax_old, "PSO-OLD - evolução da trajetória", "#1f77b4")
    _setup_axis(ax_new, "PSO-NEW - evolução da trajetória", "#2ca02c")
    _setup_axis(ax_ag, "AG - evolução da trajetória", "#d62728")

    linha_old, = ax_old.plot([], [], [], "-o", color="#1f77b4", lw=2.2, ms=3,
                             label=f"PSO-OLD (RMSE={best_old['rmse_pos_mm']:.2f} mm)")
    linha_new, = ax_new.plot([], [], [], "-^", color="#2ca02c", lw=2.2, ms=3,
                             label=f"PSO-NEW (RMSE={best_new['rmse_pos_mm']:.2f} mm)")
    linha_ag, = ax_ag.plot([], [], [], "-s", color="#d62728", lw=2.2, ms=3,
                            label=f"AG (RMSE={best_ag['rmse_pos_mm']:.2f} mm)")

    texto_old = ax_old.text2D(0.03, 0.95, "", transform=ax_old.transAxes)
    texto_new = ax_new.text2D(0.03, 0.95, "", transform=ax_new.transAxes)
    texto_ag = ax_ag.text2D(0.03, 0.95, "", transform=ax_ag.transAxes)

    def _atualizar(frame):
        idx_old = min(frame + 1, len(trajetoria_old))
        idx_new = min(frame + 1, len(trajetoria_new))
        idx_ag = min(frame + 1, len(trajetoria_ag))

        dados_old = trajetoria_old[:idx_old]
        dados_new = trajetoria_new[:idx_new]
        dados_ag = trajetoria_ag[:idx_ag]

        linha_old.set_data(dados_old[:, 0], dados_old[:, 1])
        linha_old.set_3d_properties(dados_old[:, 2])
        linha_new.set_data(dados_new[:, 0], dados_new[:, 1])
        linha_new.set_3d_properties(dados_new[:, 2])
        linha_ag.set_data(dados_ag[:, 0], dados_ag[:, 1])
        linha_ag.set_3d_properties(dados_ag[:, 2])

        texto_old.set_text(f"Passo {idx_old}/{len(trajetoria_old)}")
        texto_new.set_text(f"Passo {idx_new}/{len(trajetoria_new)}")
        texto_ag.set_text(f"Passo {idx_ag}/{len(trajetoria_ag)}")
        return linha_old, linha_new, linha_ag, texto_old, texto_new, texto_ag

    anim = FuncAnimation(fig, _atualizar, frames=n_frames, interval=180, blit=False)

    try:
        arquivo_completo = os.path.join(OUTPUT_DIR, arquivo_saida)
        anim.save(arquivo_completo, writer=PillowWriter(fps=6))
        print(f"  → {arquivo_completo} salvo")
    except Exception as exc:
        print(f"  ! Não foi possível salvar GIF ({exc}).")
        print("  ! Se quiser MP4, instale o ffmpeg ou use o arquivo GIF gerado.")

    plt.close(fig)


def posicoes_dos_elos(q):
    """Retorna as posições de cada elo do robô para um vetor de ângulos q."""
    transformacao = sm.SE3()
    pontos = [transformacao.t]
    
    for link, angulo in zip(bimo.links, q):
        transformacao = transformacao * link.A(angulo)
        pontos.append(transformacao.t)
    
    return np.array(pontos)


def gerar_video_movimento_bracos(best_old, best_new, best_ag):
    """
    Gera uma animação mostrando o movimento dos elos do braço BIMO
    para PSO-OLD, PSO-NEW e AG, lado a lado.
    """
    historico_q_old = best_old["historico_q"]
    historico_q_new = best_new["historico_q"]
    historico_q_ag = best_ag["historico_q"]
    n_frames = max(len(historico_q_old), len(historico_q_new), len(historico_q_ag))
    
    # Pré-calcular posições dos elos para cada frame
    caminhos_old = [posicoes_dos_elos(q) for q in historico_q_old]
    caminhos_new = [posicoes_dos_elos(q) for q in historico_q_new]
    caminhos_ag = [posicoes_dos_elos(q) for q in historico_q_ag]
    
    # Obter pontas (última posição de cada frame)
    pontas_old = np.array([caminho[-1] for caminho in caminhos_old])
    pontas_new = np.array([caminho[-1] for caminho in caminhos_new])
    pontas_ag = np.array([caminho[-1] for caminho in caminhos_ag])
    
    # Calcular limites de visualização
    todos_pontos = np.vstack([np.concatenate(caminhos_old), np.concatenate(caminhos_new), np.concatenate(caminhos_ag)])
    limite_xy = np.max(np.abs(todos_pontos[:, :2])) + 100
    limite_z = max(300, np.max(todos_pontos[:, 2]) + 100)
    
    fig = plt.figure(figsize=(16, 5))
    ax_old = fig.add_subplot(131, projection="3d")
    ax_new = fig.add_subplot(132, projection="3d")
    ax_ag = fig.add_subplot(133, projection="3d")
    
    # Configurar eixos
    for ax, titulo, cor in [(ax_old, "PSO-OLD - Movimento dos Elos", "#1f77b4"),
                             (ax_new, "PSO-NEW - Movimento dos Elos", "#2ca02c"),
                             (ax_ag, "AG - Movimento dos Elos", "#d62728")]:
        ax.set_title(titulo)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_xlim(-limite_xy, limite_xy)
        ax.set_ylim(-limite_xy, limite_xy)
        ax.set_zlim(0, limite_z)
    
    # Linhas dos braços e rastros
    linha_old, = ax_old.plot([], [], [], "o-", lw=2.5, color="#1f77b4", markersize=6)
    rastro_old, = ax_old.plot([], [], [], "-", lw=2, color="#1f77b4", alpha=0.6)
    ponta_old, = ax_old.plot([], [], [], "o", markersize=10, color="#ff7f0e")
    
    linha_new, = ax_new.plot([], [], [], "^-", lw=2.5, color="#2ca02c", markersize=6)
    rastro_new, = ax_new.plot([], [], [], "-", lw=2, color="#2ca02c", alpha=0.6)
    ponta_new, = ax_new.plot([], [], [], "^", markersize=10, color="#ff7f0e")
    
    linha_ag, = ax_ag.plot([], [], [], "s-", lw=2.5, color="#d62728", markersize=6)
    rastro_ag, = ax_ag.plot([], [], [], "-", lw=2, color="#d62728", alpha=0.6)
    ponta_ag, = ax_ag.plot([], [], [], "s", markersize=10, color="#ff7f0e")
    
    # Texto de status
    texto_old = ax_old.text2D(0.02, 0.95, "", transform=ax_old.transAxes, fontsize=10)
    texto_new = ax_new.text2D(0.02, 0.95, "", transform=ax_new.transAxes, fontsize=10)
    texto_ag = ax_ag.text2D(0.02, 0.95, "", transform=ax_ag.transAxes, fontsize=10)
    
    def atualizar(frame):
        idx_old = min(frame, len(caminhos_old) - 1)
        idx_new = min(frame, len(caminhos_new) - 1)
        idx_ag = min(frame, len(caminhos_ag) - 1)
        
        # PSO-OLD
        pts_old = caminhos_old[idx_old]
        linha_old.set_data(pts_old[:, 0], pts_old[:, 1])
        linha_old.set_3d_properties(pts_old[:, 2])
        
        rastro_pts_old = pontas_old[:idx_old + 1]
        rastro_old.set_data(rastro_pts_old[:, 0], rastro_pts_old[:, 1])
        rastro_old.set_3d_properties(rastro_pts_old[:, 2])
        
        ponta_old.set_data([rastro_pts_old[-1, 0]], [rastro_pts_old[-1, 1]])
        ponta_old.set_3d_properties([rastro_pts_old[-1, 2]])
        
        texto_old.set_text(f"Frame {idx_old + 1}/{len(caminhos_old)}")
        
        # PSO-NEW
        pts_new = caminhos_new[idx_new]
        linha_new.set_data(pts_new[:, 0], pts_new[:, 1])
        linha_new.set_3d_properties(pts_new[:, 2])
        
        rastro_pts_new = pontas_new[:idx_new + 1]
        rastro_new.set_data(rastro_pts_new[:, 0], rastro_pts_new[:, 1])
        rastro_new.set_3d_properties(rastro_pts_new[:, 2])
        
        ponta_new.set_data([rastro_pts_new[-1, 0]], [rastro_pts_new[-1, 1]])
        ponta_new.set_3d_properties([rastro_pts_new[-1, 2]])
        
        texto_new.set_text(f"Frame {idx_new + 1}/{len(caminhos_new)}")
        
        # AG
        pts_ag = caminhos_ag[idx_ag]
        linha_ag.set_data(pts_ag[:, 0], pts_ag[:, 1])
        linha_ag.set_3d_properties(pts_ag[:, 2])
        
        rastro_pts_ag = pontas_ag[:idx_ag + 1]
        rastro_ag.set_data(rastro_pts_ag[:, 0], rastro_pts_ag[:, 1])
        rastro_ag.set_3d_properties(rastro_pts_ag[:, 2])
        
        ponta_ag.set_data([rastro_pts_ag[-1, 0]], [rastro_pts_ag[-1, 1]])
        ponta_ag.set_3d_properties([rastro_pts_ag[-1, 2]])
        
        texto_ag.set_text(f"Frame {idx_ag + 1}/{len(caminhos_ag)}")
        
        return linha_old, rastro_old, ponta_old, texto_old, \
               linha_new, rastro_new, ponta_new, texto_new, \
               linha_ag, rastro_ag, ponta_ag, texto_ag
    
    anim = FuncAnimation(fig, atualizar, frames=n_frames, interval=180, blit=False)
    
    try:
        arquivo_saida = os.path.join(OUTPUT_DIR, "video_movimento_bracos_3algoritmos.gif")
        anim.save(arquivo_saida, writer=PillowWriter(fps=6))
        print(f"  → {arquivo_saida} salvo")
    except Exception as exc:
        print(f"  ! Erro ao salvar video de movimento ({exc})")
    
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 10. PONTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # ── Modo rápido para testar o código (1 run, poucos pontos) ──────────────
    # Descomente o bloco abaixo para teste rápido antes do experimento completo
    # N_RUNS = 2; PSO_ITER = 100; AG_GENS = 100; N_PASSOS = 8
    # TRAJETORIA = TRAJETORIA[:N_PASSOS]
    
    print("\nINICIANDO EXPERIMENTO COMPLETO")
    print(f"Estimativa de tempo: ~{N_RUNS * (PSO_POP*PSO_ITER + AG_POP*AG_GENS) / 8000:.0f} min\n")
    
    resultados_pso_antigo, resultados_pso_novo, resultados_ag = rodar_experimento()
    
    print("\n\n─── TABELA COMPARATIVA PARA O ARTIGO ───")
    imprimir_tabela(resultados_pso_antigo, resultados_pso_novo, resultados_ag)
    
    print("\n\n─── GERANDO FIGURAS ───")
    gerar_graficos(resultados_pso_antigo, resultados_pso_novo, resultados_ag)
    
    print("\n✓ Experimento concluído. Figuras e métricas prontas para o artigo.")
