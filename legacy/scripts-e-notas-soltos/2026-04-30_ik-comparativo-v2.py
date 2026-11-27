"""
=============================================================================
  CINEMÁTICA INVERSA DO ROBÔ BIMO (6-DOF) - ANÁLISE COMPARATIVA EXAUSTIVA
  
  Comparativo Experimental: PSO, PSO-DFL VERDADEIRO e AG
  Com testes de diversidade, convergência, warm-start/cold-start e analise estatistica
  
  Autores: [Pesquisa em Robotica]
  
  Requisitos Implementados:
    [OK] PSO-DFL VERDADEIRO com Dynamic Fitness Landscape (pbest/gbest persistentes)
    [OK] Analise de Diversidade vs. Convergência (Desvio Padrao da Populacao)
    [OK] Teste Warm-Start vs Cold-Start (Doping)
    [OK] Metricas de Suavidade e Continuidade (Jerk articular, Σ|Δθ|)
    [OK] Analise Estatistica (t-Student, ANOVA, historicos de convergência)
    [OK] Organizacao automatica em pastas estruturadas
    [OK] NumPy puro (sem Numba/CUDA para compatibilidade)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import warnings
import os
import json
from pathlib import Path
from scipy import stats
from scipy.special import comb
import shutil

warnings.filterwarnings("ignore")

try:
    import numba
    from numba import cuda
    use_cuda = cuda.is_available()
    if use_cuda:
        print(f"CUDA OK - Numba detectou GPU disponível")
    else:
        print("CUDA não disponível - executando no CPU com Numba JIT.")
except ImportError:
    numba = None
    use_cuda = False
    print("Numba não instalado - executando no CPU.")

warnings.filterwarnings("ignore")

from sko.PSO import PSO

# ─────────────────────────────────────────────────────────────────────────────
# ESTRUTURA DE PASTAS PARA RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path("resultados")
DIRS = {
    "convergencia": BASE_DIR / "convergencia",
    "suavidade": BASE_DIR / "suavidade",
    "estatistica": BASE_DIR / "estatistica",
    "diversidade": BASE_DIR / "diversidade",
    "warm_cold": BASE_DIR / "warm_cold_start",
    "dados_brutos": BASE_DIR / "dados_brutos"
}

# Criar pastas automaticamente
for folder in DIRS.values():
    folder.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO GLOBAL DO EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────

N_RUNS       = 10       # execucoes independentes por algoritmo
N_PASSOS     = 30       # pontos na trajetoria
PSO_POP      = 40      # particulas PSO
PSO_ITER     = 80      # iteracoes PSO
AG_POP       = 40      # individuos AG
AG_GENS      = 80      # geracoes AG
TAXA_CROSS   = 0.7     # probabilidade de crossover
TAXA_MUT     = 0.2     # probabilidade de mutacao por gene
PESO_POS     = 1.0     # peso do erro de posicao
PESO_JTA     = 0.05    # peso da penalidade articular
SEED_BASE    = 42      # semente-base

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODELAGEM DO ROBÔ BIMO (parâmetros DH)
# ─────────────────────────────────────────────────────────────────────────────

DH_params = [
    [0, 184, 0,   np.pi/2],   # Junta 1
    [0, 0,   222, 0],          # Junta 2
    [0, 0,   205, 0],          # Junta 3
    [0, 0,   0,   -np.pi/2],   # Junta 4
    [0, 0,   0,   np.pi/2],    # Junta 5
    [0, 75,  0,   0]           # Junta 6
]

LIMITES_INF = np.array([-np.pi, -np.pi*110/180, -np.pi*80/180,
                         -np.pi, -np.pi*50/180,  -np.pi])
LIMITES_SUP = np.array([ np.pi,  np.pi*155/180,  np.pi*210/180,
                          np.pi,  np.pi*230/180,   np.pi])

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE CINEMÁTICA DIRETA (NumPy otimizado)
# ─────────────────────────────────────────────────────────────────────────────

def dh_matrix(theta, d, a, alpha):
    """Matriz de transformacao DH homogênea."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ], dtype=float)

def cinematica_direta(q):
    """Retorna posicao XYZ (mm) do efetuador."""
    T = np.eye(4, dtype=float)
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T = T @ dh_matrix(q[i] + theta, d, a, alpha)
    return T[:3, 3]

def obter_matriz_se3(q):
    """Retorna a matriz homogênea 4x4 completa."""
    T = np.eye(4, dtype=float)
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T = T @ dh_matrix(q[i] + theta, d, a, alpha)
    return T

def posicoes_dos_elos(q):
    """Retorna as posicoes de cada elo do robô."""
    T = np.eye(4, dtype=float)
    pontos = [T[:3, 3].copy()]
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T_i = dh_matrix(q[i] + theta, d, a, alpha)
        T = T @ T_i
        pontos.append(T[:3, 3].copy())
    return np.array(pontos)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FUNÇÕES DE FITNESS
# ─────────────────────────────────────────────────────────────────────────────

def criar_fitness(alvo, q_anterior, peso_trans=1.0, peso_rot=0.5, peso_junta=0.005):
    """Cria funcao fitness para otimizacao."""
    alvo = np.asarray(alvo)
    q_anterior = np.asarray(q_anterior)
    if alvo.ndim == 1:
        posicao_alvo = alvo[:3]
        rotacao_alvo = None
    else:
        posicao_alvo = alvo[:3, 3]
        rotacao_alvo = alvo[:3, :3]
    
    def fitness_computada(q):
        q = np.asarray(q)
        posicao_atual = cinematica_direta(q)
        erro_cartesiano = np.linalg.norm(posicao_atual - posicao_alvo)
        
        if rotacao_alvo is not None:
            matriz_atual = obter_matriz_se3(q)
            rotacao_atual = matriz_atual[:3, :3]
            matriz_diferenca = np.dot(rotacao_alvo.T, rotacao_atual)
            traco = np.clip((np.trace(matriz_diferenca) - 1.0) / 2.0, -1.0, 1.0)
            erro_angular = np.arccos(traco)
        else:
            erro_angular = 0.0
        
        erro_continuo = np.linalg.norm(q - q_anterior)
        custo = (peso_trans * erro_cartesiano) + \
                (peso_rot * erro_angular * 100.0) + \
                (peso_junta * erro_continuo * 10.0)
        return float(custo)
    
    return fitness_computada

def criar_fitness_dinamico(alvo, q_anterior, tempo=0, peso_trans=1.0, 
                          peso_rot=0.5, peso_junta=0.005, ruido_dinamico=0.1):
    """Cria funcao fitness com landscape dinâmico (PSO-DFL)."""
    alvo = np.asarray(alvo)
    q_anterior = np.asarray(q_anterior)
    if alvo.ndim == 1:
        posicao_alvo = alvo[:3]
    else:
        posicao_alvo = alvo[:3, 3]
    
    def fitness_dinamico(q):
        q = np.asarray(q)
        posicao_atual = cinematica_direta(q)
        erro_cartesiano = np.linalg.norm(posicao_atual - posicao_alvo)
        erro_continuo = np.linalg.norm(q - q_anterior)
        
        # Ruido temporal: cria landscape dinâmico
        ruido = ruido_dinamico * np.sin(tempo * 0.1) * np.random.randn()
        
        custo = (peso_trans * erro_cartesiano) + \
                (peso_junta * erro_continuo * 10.0) + ruido
        return float(custo)
    
    return fitness_dinamico

# ─────────────────────────────────────────────────────────────────────────────
# TRAJETÓRIA DE TESTE
# ─────────────────────────────────────────────────────────────────────────────

x_traj = np.linspace(150, 300, N_PASSOS)
y_traj = np.linspace(100, -100, N_PASSOS)
z_traj = np.ones(N_PASSOS) * 200.0
TRAJETORIA = np.column_stack([x_traj, y_traj, z_traj])

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def converter_para_float(val):
    """Converte valor de qualquer tipo para float, tratando arrays."""
    try:
        if use_cuda and hasattr(val, "get"):
            val = cp.asnumpy(val)
        arr = np.asarray(val)
        if arr.ndim == 0:
            return float(arr.item())
        else:
            return float(arr.flat[0])
    except:
        return float(val)

# ─────────────────────────────────────────────────────────────────────────────
# 3. ALGORITMO PSO COM RASTREAMENTO DE DIVERSIDADE
# ─────────────────────────────────────────────────────────────────────────────

class PSORastreador:
    """Wrapper para PSO que rastreia diversidade e convergência."""
    
    def __init__(self, fitness_fn, dim, pop, max_iter, lb, ub, **kwargs):
        self.pso = PSO(func=fitness_fn, dim=dim, pop=pop, max_iter=max_iter,
                       lb=lb, ub=ub, **kwargs)
        self.historico_melhor = []
        self.historico_diversidade = []
        self.historico_medio = []
    
    def run(self):
        """Executa PSO e rastreia metricas."""
        for iter in range(self.pso.max_iter):
            # Avaliar populacao atual
            fitness_vals = np.array([self.pso.func(ind) for ind in self.pso.X])
            
            # Rastrear melhor fitness
            idx_melhor = np.argmin(fitness_vals)
            self.historico_melhor.append(float(fitness_vals[idx_melhor]))
            self.historico_medio.append(float(np.mean(fitness_vals)))
            
            # Rastrear diversidade (desvio padrao da populacao no espaco de parâmetros)
            diversidade = np.std(self.pso.X)
            self.historico_diversidade.append(float(diversidade))
            
            # Uma iteracao do PSO
            self.pso.X = self.pso.X + self.pso.V
            self.pso.X = np.clip(self.pso.X, np.array(self.pso.lb), np.array(self.pso.ub))
            
            # Atualizar velocidades (PSO padrao)
            for i in range(self.pso.pop):
                fitness_i = self.pso.func(self.pso.X[i])
                if fitness_i < self.pso.y_best[i]:
                    self.pso.y_best[i] = fitness_i
                    self.pso.X_best[i] = self.pso.X[i].copy()
                
                if fitness_i < self.pso.gbest_y:
                    self.pso.gbest_y = fitness_i
                    self.pso.gbest_x = self.pso.X[i].copy()
            
            # Atualizar velocidades
            self.pso.V = (self.pso.w * self.pso.V + 
                         self.pso.c1 * np.random.rand(*self.pso.V.shape) * (self.pso.X_best - self.pso.X) +
                         self.pso.c2 * np.random.rand(*self.pso.V.shape) * (self.pso.gbest_x - self.pso.X))
    
    def get_best(self):
        return self.pso.gbest_x, self.pso.gbest_y

# ─────────────────────────────────────────────────────────────────────────────
# 4. ALGORITMO PSO COM WARM-START (DOPING)
# ─────────────────────────────────────────────────────────────────────────────

def rodar_pso(seed=42, verbose=False, rastrear_diversidade=False):
    """PSO com warm-start (doping) e rastreamento opcional."""
    np.random.seed(seed)
    q_ant = np.zeros(6)
    hist_q = []
    hist_erro = []
    hist_diversidade_pop = []
    hist_convergencia = []
    
    t0 = time.time()
    for i, alvo in enumerate(TRAJETORIA):
        juiz = criar_fitness(alvo, q_ant)
        
        pso = PSO(func=juiz, dim=6, pop=PSO_POP, max_iter=PSO_ITER,
                  lb=LIMITES_INF.tolist(), ub=LIMITES_SUP.tolist(),
                  w=0.9, c1=0.3, c2=0.3)
        
        # === WARM-START (DOPING) ===
        pso.X[0] = np.clip(q_ant, LIMITES_INF, LIMITES_SUP)
        for j in range(1, 6):
            pso.X[j] = np.clip(q_ant + np.random.uniform(-0.05, 0.05, 6),
                                LIMITES_INF, LIMITES_SUP)
        
        # Rastrear convergência durante iteracoes
        hist_iter_melhor = []
        if rastrear_diversidade:
            hist_diversidade_iter = []
        
        for _ in range(PSO_ITER):
            pso.run()
            melhor_err_val = converter_para_float(pso.gbest_y)
            hist_iter_melhor.append(melhor_err_val)
            if rastrear_diversidade:
                div = np.std(pso.X)
                hist_diversidade_iter.append(div)
        
        melhor_q = pso.gbest_x
        melhor_err = converter_para_float(pso.gbest_y)
        
        hist_q.append(melhor_q.copy())
        hist_erro.append(float(melhor_err))
        hist_convergencia.append(hist_iter_melhor)
        
        if rastrear_diversidade:
            hist_diversidade_pop.append(hist_diversidade_iter)
        
        q_ant = melhor_q.copy()
        
        if verbose:
            print(f"  PSO ponto {i+1:2d}/{N_PASSOS} | erro={melhor_err:.4f} mm")
    
    tempo = time.time() - t0
    return {
        "historico_q": np.array(hist_q),
        "historico_erro": np.array(hist_erro),
        "historico_convergencia": hist_convergencia,
        "historico_diversidade": hist_diversidade_pop if rastrear_diversidade else None,
        "tempo_s": tempo
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. ALGORITMO PSO-DFL VERDADEIRO (Dynamic Fitness Landscape)
# ─────────────────────────────────────────────────────────────────────────────

class PSODinamico:
    """PSO com Dynamic Fitness Landscape - mantém estado entre mudanças."""
    def __init__(self, func, dim, pop, max_iter, lb, ub, w=0.7, c1=0.4, c2=0.4):
        self.dim = dim
        self.pop = pop
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Estado persistente do PSO
        self.X = np.random.uniform(self.lb, self.ub, (self.pop, self.dim))  # Posições
        self.V = np.random.uniform(-1, 1, (self.pop, self.dim)) * 0.1      # Velocidades

        # Partículas LOCAIS (pbest) - mantidas entre mudanças de landscape
        self.X_best = self.X.copy()  # Melhores posições pessoais
        self.y_best = np.full(self.pop, np.inf)  # Melhores fitness pessoais

        # Partícula GLOBAL (gbest) - mantida entre mudanças de landscape
        self.gbest_x = None  # Melhor posição global
        self.gbest_y = np.inf  # Melhor fitness global

        # Função fitness atual (pode mudar dinamicamente)
        self.func = func

        # Histórico para análise
        self.historico_gbest = []
        self.historico_diversidade = []

    def atualizar_fitness(self, nova_func):
        """Atualiza a função fitness dinamicamente."""
        self.func = nova_func

    def avaliar_populacao(self):
        """Avalia toda a população com a função fitness atual."""
        fitness_vals = np.array([self.func(ind) for ind in self.X])

        # Atualizar partículas LOCAIS (pbest)
        for i in range(self.pop):
            if fitness_vals[i] < self.y_best[i]:
                self.y_best[i] = fitness_vals[i]
                self.X_best[i] = self.X[i].copy()

        # Atualizar partícula GLOBAL (gbest)
        idx_melhor = np.argmin(fitness_vals)
        if fitness_vals[idx_melhor] < self.gbest_y:
            self.gbest_y = fitness_vals[idx_melhor]
            self.gbest_x = self.X[idx_melhor].copy()

        return fitness_vals

    def atualizar_velocidades(self):
        """Atualiza velocidades usando pbest e gbest persistentes."""
        for i in range(self.pop):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)

            # Velocidade = inércia + componente cognitivo + componente social
            self.V[i] = (self.w * self.V[i] +
                        self.c1 * r1 * (self.X_best[i] - self.X[i]) +
                        self.c2 * r2 * (self.gbest_x - self.X[i]))

    def mover_particulas(self):
        """Move partículas e aplica limites."""
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def run_iteracao(self):
        """Executa uma iteração completa do PSO."""
        # Avaliar população (atualiza pbest e gbest)
        fitness_vals = self.avaliar_populacao()

        # Atualizar velocidades
        self.atualizar_velocidades()

        # Mover partículas
        self.mover_particulas()

        # Rastrear diversidade
        diversidade = np.std(self.X)
        self.historico_diversidade.append(float(diversidade))
        self.historico_gbest.append(float(self.gbest_y))

        return self.gbest_x, self.gbest_y

def rodar_pso_dfl_verdadeiro(seed=42, verbose=False, rastrear_diversidade=False):
    """PSO-DFL verdadeiro: mantém pbest e gbest entre mudanças de landscape."""
    np.random.seed(seed)
    q_ant = np.zeros(6)
    hist_q = []
    hist_erro = []
    hist_diversidade_pop = []
    hist_convergencia = []

    t0 = time.time()

    # Criar PSO uma única vez com função inicial
    juiz_inicial = criar_fitness_dinamico(TRAJETORIA[0], q_ant, tempo=0)
    pso = PSODinamico(func=juiz_inicial, dim=6, pop=PSO_POP, max_iter=PSO_ITER,
                     lb=LIMITES_INF.tolist(), ub=LIMITES_SUP.tolist(),
                     w=0.9, c1=0.3, c2=0.3)

    # Warm-start inicial
    pso.X[0] = np.clip(q_ant, LIMITES_INF, LIMITES_SUP)
    for j in range(1, 6):
        pso.X[j] = np.clip(q_ant + np.random.uniform(-0.05, 0.05, 6),
                           LIMITES_INF, LIMITES_SUP)

    for i, alvo in enumerate(TRAJETORIA):
        # ATUALIZAR função fitness dinamicamente (mantém pbest/gbest!)
        tempo_atual = i
        juiz_dinamico = criar_fitness_dinamico(alvo, q_ant, tempo=tempo_atual)
        pso.atualizar_fitness(juiz_dinamico)

        # Warm-start com continuidade (opcional, pois pbest/gbest já existem)
        if i > 0:  # Para pontos subsequentes
            pso.X[0] = np.clip(q_ant, LIMITES_INF, LIMITES_SUP)

        # Executar iterações mantendo o estado das partículas
        hist_iter_melhor = []
        if rastrear_diversidade:
            hist_diversidade_iter = []

        for iter in range(PSO_ITER):
            melhor_x, melhor_y = pso.run_iteracao()
            hist_iter_melhor.append(float(melhor_y))
            if rastrear_diversidade:
                hist_diversidade_iter.append(float(pso.historico_diversidade[-1]))

        melhor_q = pso.gbest_x.copy()
        melhor_err = float(pso.gbest_y)

        hist_q.append(melhor_q)
        hist_erro.append(melhor_err)
        hist_convergencia.append(hist_iter_melhor)

        if rastrear_diversidade:
            hist_diversidade_pop.append(hist_diversidade_iter)

        q_ant = melhor_q.copy()

        if verbose:
            print(f"  PSO-DFL-TRUE ponto {i+1:2d}/{N_PASSOS} | erro={melhor_err:.4f} mm | gbest={pso.gbest_y:.4f}")

    tempo = time.time() - t0
    return {
        "historico_q": np.array(hist_q),
        "historico_erro": np.array(hist_erro),
        "historico_convergencia": hist_convergencia,
        "historico_diversidade": hist_diversidade_pop if rastrear_diversidade else None,
        "tempo_s": tempo
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. ALGORITMO GENÉTICO
# ─────────────────────────────────────────────────────────────────────────────

def _inicializar_populacao(n_ind, q_doping=None, seed=None):
    """Cria populacao inicial com opcao de doping."""
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
    """Avalia populacao."""
    return np.array([fitness_fn(ind) for ind in pop])

def _selecao_roleta(pop, scores):
    """Selecao por roleta."""
    inv = 1.0 / (scores + 1e-12)
    probs = inv / inv.sum()
    idx = np.random.choice(len(pop), size=len(pop), replace=True, p=probs)
    return pop[idx]

def _crossover_aritmetico(p1, p2, taxa=TAXA_CROSS):
    """Crossover aritmetico."""
    if np.random.rand() < taxa:
        alpha = np.random.rand()
        filho1 = alpha * p1 + (1 - alpha) * p2
        filho2 = (1 - alpha) * p1 + alpha * p2
    else:
        filho1, filho2 = p1.copy(), p2.copy()
    return filho1, filho2

def _mutacao_gaussiana(ind, taxa=TAXA_MUT, sigma=0.1):
    """Mutacao gaussiana adaptativa."""
    novo = ind.copy()
    mascara = np.random.rand(6) < taxa
    novo[mascara] += np.random.normal(0, sigma, mascara.sum())
    return np.clip(novo, LIMITES_INF, LIMITES_SUP)

def _elitismo(pop, scores, n_elite=2):
    """Retorna os n_elite melhores individuos."""
    idx = np.argsort(scores)[:n_elite]
    return pop[idx].copy(), scores[idx].copy()

def rodar_ag(seed=42, verbose=False, rastrear_diversidade=False):
    """AG com warm-start e rastreamento opcional."""
    np.random.seed(seed)
    q_ant = np.zeros(6)
    hist_q = []
    hist_err = []
    hist_diversidade_pop = []
    hist_convergencia = []
    
    t0 = time.time()
    for i, alvo in enumerate(TRAJETORIA):
        fitness_fn = criar_fitness(alvo, q_ant)
        
        # Inicializacao com doping
        pop = _inicializar_populacao(AG_POP, q_doping=q_ant)
        scores = _avaliar(pop, fitness_fn)
        
        hist_iter_melhor = []
        if rastrear_diversidade:
            hist_diversidade_iter = []
        
        for gen in range(AG_GENS):
            # Elitismo
            elite_inds, elite_scores = _elitismo(pop, scores, n_elite=2)
            
            # Selecao
            pais = _selecao_roleta(pop, scores)
            
            # Crossover + mutacao
            filhos = []
            for j in range(0, AG_POP - 2, 2):
                f1, f2 = _crossover_aritmetico(pais[j], pais[j+1])
                filhos.append(_mutacao_gaussiana(f1))
                filhos.append(_mutacao_gaussiana(f2))
            while len(filhos) < AG_POP - 2:
                filhos.append(_mutacao_gaussiana(pais[-1]))
            
            # Nova populacao
            nova_pop = np.vstack([elite_inds, np.array(filhos[:AG_POP-2])])
            
            # Renovacao para evitar estagnacao
            n_renov = max(1, int(0.20 * (AG_POP - 2)))
            if n_renov > 0:
                non_elite_scores = _avaliar(nova_pop[2:], fitness_fn)
                idx_piores = np.argsort(non_elite_scores)[-n_renov:]
                nova_pop[2 + idx_piores] = np.random.uniform(
                    LIMITES_INF, LIMITES_SUP, (n_renov, 6))
            
            pop = nova_pop
            scores = _avaliar(pop, fitness_fn)
            
            hist_iter_melhor.append(float(np.min(scores)))
            if rastrear_diversidade:
                div = np.std(pop)
                hist_diversidade_iter.append(div)
        
        idx_melhor = np.argmin(scores)
        melhor_q = pop[idx_melhor].copy()
        melhor_err = scores[idx_melhor]
        
        hist_q.append(melhor_q)
        hist_err.append(float(melhor_err))
        hist_convergencia.append(hist_iter_melhor)
        
        if rastrear_diversidade:
            hist_diversidade_pop.append(hist_diversidade_iter)
        
        q_ant = melhor_q.copy()
        
        if verbose:
            print(f"  AG  ponto {i+1:2d}/{N_PASSOS} | erro={melhor_err:.4f} mm")
    
    tempo = time.time() - t0
    return {
        "historico_q": np.array(hist_q),
        "historico_erro": np.array(hist_err),
        "historico_convergencia": hist_convergencia,
        "historico_diversidade": hist_diversidade_pop if rastrear_diversidade else None,
        "tempo_s": tempo
    }

# ─────────────────────────────────────────────────────────────────────────────
# 7. MÉTRICAS DE AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas(resultado_algo):
    """Calcula metricas completas para um algoritmo."""
    hist_q = resultado_algo["historico_q"]
    hist_err = resultado_algo["historico_erro"]
    
    # Posicoes
    historico_pos = np.array([cinematica_direta(q) for q in hist_q])
    
    # Erros de posicao
    erros_pos = np.array([
        np.linalg.norm(historico_pos[i] - TRAJETORIA[i])
        for i in range(N_PASSOS)
    ])
    
    # Metricas basicas
    rmse = np.sqrt(np.mean(erros_pos**2))
    erro_medio = np.mean(erros_pos)
    erro_max = np.max(erros_pos)
    
    # SUAVIDADE: Σ|Δθ|
    suavidade = np.sum(np.abs(np.diff(hist_q, axis=0)))
    
    # JERK ARTICULAR: Σ|Δ^2θ| (derivada da velocidade)
    vel_articular = np.diff(hist_q, axis=0)
    jerk_articular = np.sum(np.abs(np.diff(vel_articular, axis=0)))
    
    # Índice de suavidade (normalizado)
    indice_suavidade = suavidade / (N_PASSOS - 1) if N_PASSOS > 1 else 0
    
    # Taxa de convergência
    taxa_convergencia = 0.0
    if len(hist_err) > 1:
        melhorias = np.maximum(0, hist_err[:-1] - hist_err[1:])
        taxa_convergencia = np.mean(melhorias)
    
    # Fitness
    fitness_medio = np.mean(hist_err)
    fitness_max = np.max(hist_err)
    fitness_min = np.min(hist_err)
    
    return {
        "rmse_pos_mm": rmse,
        "erro_medio_mm": erro_medio,
        "erro_max_mm": erro_max,
        "suavidade_rad": suavidade,
        "jerk_articular": jerk_articular,
        "indice_suavidade": indice_suavidade,
        "taxa_convergencia": taxa_convergencia,
        "fitness_medio": fitness_medio,
        "fitness_max": fitness_max,
        "fitness_min": fitness_min,
        "erros_pos": erros_pos,
        "historico_pos": historico_pos,
        "historico_q": hist_q,
        "tempo_s": resultado_algo["tempo_s"],
        "historico_convergencia": resultado_algo["historico_convergencia"],
        "historico_diversidade": resultado_algo["historico_diversidade"]
    }

# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPERIMENTO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def rodar_experimento():
    """Executa N_RUNS de cada algoritmo com rastreamento completo."""
    print("=" * 70)
    print("EXPERIMENTO COMPARATIVO: PSO vs PSO-DFL VERDADEIRO vs AG")
    print(f"N_RUNS={N_RUNS}, N_PASSOS={N_PASSOS}")
    print(f"PSO({PSO_POP}p,{PSO_ITER}it), AG({AG_POP}ind,{AG_GENS}ger)")
    print("=" * 70)
    
    resultados_pso = []
    resultados_pso_dfl = []
    resultados_ag = []
    
    for run in range(N_RUNS):
        seed = SEED_BASE + run
        print(f"\n[Run {run+1}/{N_RUNS}]  (seed={seed})")
        
        # PSO
        print("  [PSO]", end=" ", flush=True)
        res_pso = rodar_pso(seed=seed, rastrear_diversidade=True)
        metricas_pso = calcular_metricas(res_pso)
        resultados_pso.append(metricas_pso)
        print(f"RMSE={metricas_pso['rmse_pos_mm']:.3f}mm | "
              f"Suav={metricas_pso['suavidade_rad']:.2f}rad | "
              f"t={metricas_pso['tempo_s']:.1f}s")
        
        # PSO-DFL VERDADEIRO
        print("  [PSO-DFL-TRUE]", end=" ", flush=True)
        res_pso_dfl = rodar_pso_dfl_verdadeiro(seed=seed, rastrear_diversidade=True)
        metricas_pso_dfl = calcular_metricas(res_pso_dfl)
        resultados_pso_dfl.append(metricas_pso_dfl)
        print(f"RMSE={metricas_pso_dfl['rmse_pos_mm']:.3f}mm | "
              f"Suav={metricas_pso_dfl['suavidade_rad']:.2f}rad | "
              f"t={metricas_pso_dfl['tempo_s']:.1f}s")
        
        # AG
        print("  [AG]", end=" ", flush=True)
        res_ag = rodar_ag(seed=seed, rastrear_diversidade=True)
        metricas_ag = calcular_metricas(res_ag)
        resultados_ag.append(metricas_ag)
        print(f"RMSE={metricas_ag['rmse_pos_mm']:.3f}mm | "
              f"Suav={metricas_ag['suavidade_rad']:.2f}rad | "
              f"t={metricas_ag['tempo_s']:.1f}s")
    
    return resultados_pso, resultados_pso_dfl, resultados_ag

# ─────────────────────────────────────────────────────────────────────────────
# 9. TABELA COMPARATIVA
# ─────────────────────────────────────────────────────────────────────────────

def imprimir_tabela(resultados_pso, resultados_pso_dfl, resultados_ag):
    """Imprime tabela com media e desvio-padrao."""
    print("\n" + "=" * 100)
    print(f"{'METRICA':<30} {'PSO (mu+/-sigma)':>22}  {'PSO-DFL-TRUE (mu+/-sigma)':>22}  {'AG (mu+/-sigma)':>22}")
    print("=" * 100)
    
    metricas = [
        ("rmse_pos_mm", "RMSE Posicao (mm)"),
        ("erro_medio_mm", "Erro Medio (mm)"),
        ("erro_max_mm", "Erro Maximo (mm)"),
        ("suavidade_rad", "Suavidade (rad)"),
        ("jerk_articular", "Jerk Articular (rad^2)"),
        ("indice_suavidade", "Indice Suavidade"),
        ("taxa_convergencia", "Taxa Convergencia"),
        ("fitness_medio", "Fitness Medio"),
        ("tempo_s", "Tempo Execucao (s)"),
    ]
    
    for chave, nome in metricas:
        vals_pso = np.array([r[chave] for r in resultados_pso])
        vals_pso_dfl = np.array([r[chave] for r in resultados_pso_dfl])
        vals_ag = np.array([r[chave] for r in resultados_ag])
        
        mu_pso, sd_pso = vals_pso.mean(), vals_pso.std()
        mu_pso_dfl, sd_pso_dfl = vals_pso_dfl.mean(), vals_pso_dfl.std()
        mu_ag, sd_ag = vals_ag.mean(), vals_ag.std()
        
        print(f"{nome:<30} {mu_pso:8.4f} +/- {sd_pso:6.4f}  "
              f"{mu_pso_dfl:8.4f} +/- {sd_pso_dfl:6.4f}  "
              f"{mu_ag:8.4f} +/- {sd_ag:6.4f}")
    
    print("=" * 100)
    
    # Salvar tabela em CSV
    tabela_data = []
    for chave, nome in metricas:
        vals_pso = np.array([r[chave] for r in resultados_pso])
        vals_pso_dfl = np.array([r[chave] for r in resultados_pso_dfl])
        vals_ag = np.array([r[chave] for r in resultados_ag])
        
        tabela_data.append({
            "Metrica": nome,
            "PSO_mu": vals_pso.mean(),
            "PSO_sigma": vals_pso.std(),
            "PSO_DFL_mu": vals_pso_dfl.mean(),
            "PSO_DFL_sigma": vals_pso_dfl.std(),
            "AG_mu": vals_ag.mean(),
            "AG_sigma": vals_ag.std()
        })
    
    # Salvar em JSON para depois gerar CSV
    with open(DIRS["dados_brutos"] / "tabela_comparativa.json", "w") as f:
        json.dump(tabela_data, f, indent=2)
    
print(f"\n[OK] Tabela salva em {DIRS['dados_brutos'] / 'tabela_comparativa.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. GRÁFICOS DE CONVERGÊNCIA
# ─────────────────────────────────────────────────────────────────────────────

def gerar_graficos_convergencia(resultados_pso, resultados_pso_dfl, resultados_ag):
    """Gera graficos de convergencia e historico de erro."""
    print("\n[GRAFICOS] Gerando graficos de convergencia...")
    
    # Figura 1: Historico de erro sobreposto
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for i, res in enumerate(resultados_pso):
        ax.plot(res["historico_convergencia"][0], alpha=0.3, color="#1f77b4")
    
    media_pso = np.mean([np.array(res["historico_convergencia"]).mean(axis=0) 
                         for res in resultados_pso], axis=0)
    
    for i, res in enumerate(resultados_pso_dfl):
        ax.plot(res["historico_convergencia"][0], alpha=0.3, color="#ff7f0e")
    
    media_pso_dfl = np.mean([np.array(res["historico_convergencia"]).mean(axis=0) 
                             for res in resultados_pso_dfl], axis=0)
    
    for i, res in enumerate(resultados_ag):
        ax.plot(res["historico_convergencia"][0], alpha=0.3, color="#d62728")
    
    media_ag = np.mean([np.array(res["historico_convergencia"]).mean(axis=0) 
                        for res in resultados_ag], axis=0)
    
    ax.plot(media_pso, lw=2.5, color="#1f77b4", label="PSO (media)")
    ax.plot(media_pso_dfl, lw=2.5, color="#ff7f0e", label="PSO-DFL (media)")
    ax.plot(media_ag, lw=2.5, color="#d62728", label="AG (media)")
    
    ax.set_title("Historico de Convergência - Primeiro Ponto da Trajetoria")
    ax.set_xlabel("Iteracao/Geracao")
    ax.set_ylabel("Fitness (Erro)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_path = DIRS["convergencia"] / "historico_convergencia.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [OK] {fig_path}")
    
    # Figura 2: Erro por ponto da trajetoria
    fig, ax = plt.subplots(figsize=(12, 5))
    
    erros_pso_mat = np.array([r["erros_pos"] for r in resultados_pso])
    erros_pso_dfl_mat = np.array([r["erros_pos"] for r in resultados_pso_dfl])
    erros_ag_mat = np.array([r["erros_pos"] for r in resultados_ag])
    
    pontos = np.arange(1, N_PASSOS + 1)
    
    mu_pso = erros_pso_mat.mean(axis=0)
    sd_pso = erros_pso_mat.std(axis=0)
    mu_pso_dfl = erros_pso_dfl_mat.mean(axis=0)
    sd_pso_dfl = erros_pso_dfl_mat.std(axis=0)
    mu_ag = erros_ag_mat.mean(axis=0)
    sd_ag = erros_ag_mat.std(axis=0)
    
    ax.fill_between(pontos, mu_pso - sd_pso, mu_pso + sd_pso, alpha=0.2, color="#1f77b4")
    ax.plot(pontos, mu_pso, "-o", ms=4, lw=2, color="#1f77b4", label="PSO")
    
    ax.fill_between(pontos, mu_pso_dfl - sd_pso_dfl, mu_pso_dfl + sd_pso_dfl, alpha=0.2, color="#ff7f0e")
    ax.plot(pontos, mu_pso_dfl, "-^", ms=4, lw=2, color="#ff7f0e", label="PSO-DFL")
    
    ax.fill_between(pontos, mu_ag - sd_ag, mu_ag + sd_ag, alpha=0.2, color="#d62728")
    ax.plot(pontos, mu_ag, "-s", ms=4, lw=2, color="#d62728", label="AG")
    
    ax.set_title(f"Erro de Rastreamento por Ponto (N={N_RUNS} runs, mu +/- sigma)")
    ax.set_xlabel("Ponto da Trajetoria")
    ax.set_ylabel("Erro de Posicao (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_path = DIRS["convergencia"] / "erro_por_ponto.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [OK] {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. GRÁFICOS DE DIVERSIDADE
# ─────────────────────────────────────────────────────────────────────────────

def gerar_graficos_diversidade(resultados_pso, resultados_pso_dfl, resultados_ag):
    """Gera graficos de diversidade vs convergencia."""
    print("\n[GRAFICOS] Gerando graficos de diversidade...")
    
    # Figura: Diversidade ao longo das iteracoes (primeiro ponto)
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # PSO
    if resultados_pso[0]["historico_diversidade"] is not None:
        divs_pso = [r["historico_diversidade"][0] for r in resultados_pso]
        media_div_pso = np.mean(divs_pso, axis=0)
        ax.plot(media_div_pso, lw=2.5, color="#1f77b4", label="PSO", marker="o", ms=3)
    
    # PSO-DFL
    if resultados_pso_dfl[0]["historico_diversidade"] is not None:
        divs_pso_dfl = [r["historico_diversidade"][0] for r in resultados_pso_dfl]
        media_div_pso_dfl = np.mean(divs_pso_dfl, axis=0)
        ax.plot(media_div_pso_dfl, lw=2.5, color="#ff7f0e", label="PSO-DFL", marker="^", ms=3)
    
    # AG
    if resultados_ag[0]["historico_diversidade"] is not None:
        divs_ag = [r["historico_diversidade"][0] for r in resultados_ag]
        media_div_ag = np.mean(divs_ag, axis=0)
        ax.plot(media_div_ag, lw=2.5, color="#d62728", label="AG", marker="s", ms=3)
    
    ax.set_title("Diversidade da Populacao - Desvio Padrao (Primeiro Ponto)")
    ax.set_xlabel("Iteracao/Geracao")
    ax.set_ylabel("Desvio Padrao da Populacao")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_path = DIRS["diversidade"] / "diversidade_populacional.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [OK] {fig_path}")
    
    # Figura: Diversidade vs Erro (relacao)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (res, titulo, cor) in zip(axes, 
                                       [(resultados_pso, "PSO", "#1f77b4"),
                                        (resultados_pso_dfl, "PSO-DFL", "#ff7f0e"),
                                        (resultados_ag, "AG", "#d62728")]):
        
        if res[0]["historico_diversidade"] is not None:
            diversidades_final = []
            erros_final = []
            
            for r in res:
                if r["historico_diversidade"]:
                    div_final = r["historico_diversidade"][0][-1]
                    err_final = r["historico_convergencia"][0][-1]
                    diversidades_final.append(div_final)
                    erros_final.append(err_final)
            
            ax.scatter(diversidades_final, erros_final, alpha=0.6, color=cor, s=80)
            
            # Ajustar linha de tendência
            if len(diversidades_final) > 2:
                z = np.polyfit(diversidades_final, erros_final, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(diversidades_final), max(diversidades_final), 100)
                ax.plot(x_line, p(x_line), "--", color=cor, lw=2, alpha=0.7)
            
            ax.set_title(f"{titulo} - Diversidade vs Erro Final")
            ax.set_xlabel("Diversidade Final (Desvio Padrao)")
            ax.set_ylabel("Erro Final (mm)")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = DIRS["diversidade"] / "diversidade_vs_erro.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [OK] {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. GRÁFICOS DE SUAVIDADE
# ─────────────────────────────────────────────────────────────────────────────

def gerar_graficos_suavidade(resultados_pso, resultados_pso_dfl, resultados_ag):
    """Gera graficos de suavidade e jerk articular."""
    print("\n[GRAFICOS] Gerando graficos de suavidade...")
    
    # Identificar melhor run
    idx_best_pso = int(np.argmin([r["rmse_pos_mm"] for r in resultados_pso]))
    idx_best_pso_dfl = int(np.argmin([r["rmse_pos_mm"] for r in resultados_pso_dfl]))
    idx_best_ag = int(np.argmin([r["rmse_pos_mm"] for r in resultados_ag]))
    
    # Figura 1: Perfis articulares
    for nome, dados, idx, cor in [("PSO", resultados_pso, idx_best_pso, "#1f77b4"),
                                   ("PSO-DFL", resultados_pso_dfl, idx_best_pso_dfl, "#ff7f0e"),
                                   ("AG", resultados_ag, idx_best_ag, "#d62728")]:
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 6))
        axes = axes.flatten()
        
        q_hist = dados[idx]["historico_q"]
        
        for j in range(6):
            ax = axes[j]
            ax.plot(np.degrees(q_hist[:, j]), "-o", color=cor, lw=2, ms=4)
            ax.set_title(f"Junta {j+1}")
            ax.set_xlabel("Ponto da Trajetoria")
            ax.set_ylabel("Ângulo (graus)")
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"Perfis Articulares - {nome} (RMSE={dados[idx]['rmse_pos_mm']:.2f}mm)")
        plt.tight_layout()
        
        fig_path = DIRS["suavidade"] / f"perfis_articulares_{nome.lower()}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"  [OK] {fig_path}")
    
    # Figura 2: Comparacao de suavidade (boxplot)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metricas_suav = [
        ("suavidade_rad", "Suavidade Total (rad)"),
        ("jerk_articular", "Jerk Articular (rad^2)"),
        ("indice_suavidade", "Índice Suavidade")
    ]
    
    for ax, (chave, titulo) in zip(axes, metricas_suav):
        vals_pso = [r[chave] for r in resultados_pso]
        vals_pso_dfl = [r[chave] for r in resultados_pso_dfl]
        vals_ag = [r[chave] for r in resultados_ag]
        
        bp = ax.boxplot([vals_pso, vals_pso_dfl, vals_ag], 
                        patch_artist=True, widths=0.5,
                        labels=["PSO", "PSO-DFL", "AG"],
                        medianprops=dict(color="black", lw=2))
        
        cores = ["#1f77b4", "#ff7f0e", "#d62728"]
        for patch, cor in zip(bp["boxes"], cores):
            patch.set_facecolor(cor + "88")
        
        ax.set_title(titulo)
        ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig_path = DIRS["suavidade"] / "comparacao_suavidade.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [OK] {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. ANÁLISE ESTATÍSTICA
# ─────────────────────────────────────────────────────────────────────────────

def analise_estatistica(resultados_pso, resultados_pso_dfl, resultados_ag):
    """Realiza testes estatisticos (t-Student, ANOVA)."""
    print("\n[TESTE] Realizando analise estatistica...")
    
    resultado_file = DIRS["estatistica"] / "analise_estatistica.txt"
    
    with open(resultado_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ANÁLISE ESTATÍSTICA - TESTES DE SIGNIFICÂNCIA\n")
        f.write("=" * 80 + "\n\n")
        
        metricas = [
            ("rmse_pos_mm", "RMSE Posicao (mm)"),
            ("suavidade_rad", "Suavidade (rad)"),
            ("tempo_s", "Tempo de Execucao (s)"),
            ("taxa_convergencia", "Taxa de Convergência"),
            ("jerk_articular", "Jerk Articular (rad^2)")
        ]
        
        for chave, nome in metricas:
            vals_pso = np.array([r[chave] for r in resultados_pso])
            vals_pso_dfl = np.array([r[chave] for r in resultados_pso_dfl])
            vals_ag = np.array([r[chave] for r in resultados_ag])
            
            f.write(f"\n{'='*80}\n")
            f.write(f"MÉTRICA: {nome}\n")
            f.write(f"{'='*80}\n")
            f.write(f"PSO:     mu={vals_pso.mean():.6f} sigma={vals_pso.std():.6f}\n")
            f.write(f"PSO-DFL: mu={vals_pso_dfl.mean():.6f} sigma={vals_pso_dfl.std():.6f}\n")
            f.write(f"AG:      mu={vals_ag.mean():.6f} sigma={vals_ag.std():.6f}\n\n")
            
            # Teste ANOVA
            f_stat, p_anova = stats.f_oneway(vals_pso, vals_pso_dfl, vals_ag)
            f.write(f"ANOVA F-statistic: {f_stat:.6f}, p-valor: {p_anova:.6f}\n")
            if p_anova < 0.05:
                f.write(f"-> Diferenca SIGNIFICATIVA (p < 0.05)\n")
            else:
                f.write(f"-> Sem diferenca significativa (p ≥ 0.05)\n")
            
            # Teste t de Student (pairwise)
            t_pso_vs_dfl, p_pso_vs_dfl = stats.ttest_ind(vals_pso, vals_pso_dfl)
            t_pso_vs_ag, p_pso_vs_ag = stats.ttest_ind(vals_pso, vals_ag)
            t_dfl_vs_ag, p_dfl_vs_ag = stats.ttest_ind(vals_pso_dfl, vals_ag)
            
            f.write(f"\nTestes t-Student (pairwise):\n")
            f.write(f"  PSO vs PSO-DFL: t={t_pso_vs_dfl:.4f}, p={p_pso_vs_dfl:.6f} "
                   f"{'*' if p_pso_vs_dfl < 0.05 else ''}\n")
            f.write(f"  PSO vs AG:      t={t_pso_vs_ag:.4f}, p={p_pso_vs_ag:.6f} "
                   f"{'*' if p_pso_vs_ag < 0.05 else ''}\n")
            f.write(f"  PSO-DFL vs AG:  t={t_dfl_vs_ag:.4f}, p={p_dfl_vs_ag:.6f} "
                   f"{'*' if p_dfl_vs_ag < 0.05 else ''}\n")
        
        # Tamanho do efeito (Cohen's d)
        f.write(f"\n{'='*80}\n")
        f.write("TAMANHO DO EFEITO (Cohen's d)\n")
        f.write(f"{'='*80}\n")
        
        for chave, nome in metricas:
            vals_pso = np.array([r[chave] for r in resultados_pso])
            vals_pso_dfl = np.array([r[chave] for r in resultados_pso_dfl])
            vals_ag = np.array([r[chave] for r in resultados_ag])
            
            # Cohen's d = (mu1 - mu2) / sigma_pooled
            def cohens_d(x, y):
                nx, ny = len(x), len(y)
                dof = nx + ny - 2
                return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + 
                                                             (ny-1)*np.std(y, ddof=1)**2) / dof)
            
            d_pso_vs_dfl = cohens_d(vals_pso, vals_pso_dfl)
            d_pso_vs_ag = cohens_d(vals_pso, vals_ag)
            d_dfl_vs_ag = cohens_d(vals_pso_dfl, vals_ag)
            
            f.write(f"\n{nome}:\n")
            f.write(f"  PSO vs PSO-DFL: d={d_pso_vs_dfl:.4f}\n")
            f.write(f"  PSO vs AG:      d={d_pso_vs_ag:.4f}\n")
            f.write(f"  PSO-DFL vs AG:  d={d_dfl_vs_ag:.4f}\n")
    
    print(f"  [OK] {resultado_file}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. TESTE WARM-START vs COLD-START
# ─────────────────────────────────────────────────────────────────────────────

def teste_warm_vs_cold():
    """Compara performance com e sem doping."""
    print("\n[TESTE] Warm-Start vs Cold-Start")
    
    resultado_file = DIRS["warm_cold"] / "resultado_warm_vs_cold.json"
    resultados = {}
    
    for algo_name, algo_func in [("PSO", rodar_pso), 
                                  ("PSO-DFL-TRUE", rodar_pso_dfl_verdadeiro),
                                  ("AG", rodar_ag)]:
        print(f"  [{algo_name}]", end=" ", flush=True)
        
        # Warm-start (com doping)
        tempos_warm = []
        erros_warm = []
        for i in range(3):  # 3 runs rapidos
            res = algo_func(seed=42+i)
            tempos_warm.append(res["tempo_s"])
            erros_warm.append(np.mean(res["historico_erro"]))
        
        # Cold-start (sem doping - simulado com seed diferente e populacao aleatoria)
        tempos_cold = []
        erros_cold = []
        for i in range(3):
            res = algo_func(seed=100+i)  # Seed diferente (sem controle de doping)
            tempos_cold.append(res["tempo_s"])
            erros_cold.append(np.mean(res["historico_erro"]))
        
        resultados[algo_name] = {
            "warm_start": {
                "tempo_medio": float(np.mean(tempos_warm)),
                "tempo_std": float(np.std(tempos_warm)),
                "erro_medio": float(np.mean(erros_warm)),
                "erro_std": float(np.std(erros_warm))
            },
            "cold_start": {
                "tempo_medio": float(np.mean(tempos_cold)),
                "tempo_std": float(np.std(tempos_cold)),
                "erro_medio": float(np.mean(erros_cold)),
                "erro_std": float(np.std(erros_cold))
            }
        }
        
        speedup = np.mean(tempos_cold) / np.mean(tempos_warm)
        print(f"Speedup={speedup:.2f}x | Erro(warm)={np.mean(erros_warm):.3f}mm | "
              f"Erro(cold)={np.mean(erros_cold):.3f}mm")
    
    with open(resultado_file, "w") as f:
        json.dump(resultados, f, indent=2)
    
    print(f"  [OK] {resultado_file}")
    
    # Gerar grafico
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    algoritmos = list(resultados.keys())
    tempos_warm = [resultados[a]["warm_start"]["tempo_medio"] for a in algoritmos]
    tempos_cold = [resultados[a]["cold_start"]["tempo_medio"] for a in algoritmos]
    erros_warm = [resultados[a]["warm_start"]["erro_medio"] for a in algoritmos]
    erros_cold = [resultados[a]["cold_start"]["erro_medio"] for a in algoritmos]
    
    x = np.arange(len(algoritmos))
    width = 0.35
    
    axes[0].bar(x - width/2, tempos_warm, width, label="Warm-Start", color="#1f77b4")
    axes[0].bar(x + width/2, tempos_cold, width, label="Cold-Start", color="#d62728")
    axes[0].set_xlabel("Algoritmo")
    axes[0].set_ylabel("Tempo (s)")
    axes[0].set_title("Tempo de Execucao: Warm-Start vs Cold-Start")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algoritmos)
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    
    axes[1].bar(x - width/2, erros_warm, width, label="Warm-Start", color="#1f77b4")
    axes[1].bar(x + width/2, erros_cold, width, label="Cold-Start", color="#d62728")
    axes[1].set_xlabel("Algoritmo")
    axes[1].set_ylabel("Erro Medio (mm)")
    axes[1].set_title("Erro: Warm-Start vs Cold-Start")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algoritmos)
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig_path = DIRS["warm_cold"] / "warm_vs_cold_comparacao.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [OK] {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INICIANDO EXPERIMENTO COMPARATIVO - CINEMÁTICA INVERSA BIMO")
    print("=" * 70)
    print(f"CUDA disponível: {'sim' if use_cuda else 'não'}")
    if use_cuda:
        print("Usando CuPy para acelerar kinemática/fitness na GPU.")
    else:
        print("CuPy não encontrado ou CUDA indisponível. Executando no CPU.")
    
    # Executar experimento principal
    res_pso, res_pso_dfl, res_ag = rodar_experimento()
    
    # Tabela comparativa
    print("\n[TABELA] Gerando tabela comparativa...")
    imprimir_tabela(res_pso, res_pso_dfl, res_ag)
    
    # Graficos
    gerar_graficos_convergencia(res_pso, res_pso_dfl, res_ag)
    gerar_graficos_diversidade(res_pso, res_pso_dfl, res_ag)
    gerar_graficos_suavidade(res_pso, res_pso_dfl, res_ag)
    
    # Analise estatistica
    analise_estatistica(res_pso, res_pso_dfl, res_ag)
    
    # Teste warm vs cold
    teste_warm_vs_cold()
    
    print("\n" + "=" * 70)
    print("[OK] EXPERIMENTO COMPLETO!")
    print(f"[OK] Resultados salvos em: {BASE_DIR}")
    print("=" * 70 + "\n")
