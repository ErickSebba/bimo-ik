"""
=============================================================================
  CINEMÁTICA INVERSA DO ROBÔ BIMO (6-DOF) — v4.0
  Comparativo JUSTO: PSO | PSO-DFL | AG

  TODAS AS CORREÇÕES CRÍTICAS APLICADAS:
  ═══════════════════════════════════════════════════════════════════════

  [✓] CORREÇÃO 1 — Sem warm-start universal
      Todos os algoritmos iniciam com população 100% aleatória (cold-start).
      Elimina vantagem de doping que mascarava diferenças reais.

  [✓] CORREÇÃO 2 — Comparação estruturalmente justa
      PSO standard  : nova população aleatória a cada passo da trajetória.
      PSO-DFL       : MESMA implementação base, mas estado (X, V, pbest,
                      gbest) PRESERVADO entre passos + reavaliação de
                      pbest/gbest ao trocar landscape.
      AG            : nova população aleatória a cada passo.
      Diferença PSO vs PSO-DFL: APENAS preservação de estado + reavaliação.

  [✓] CORREÇÃO 3 — Fitness functions SIMÉTRICAS
      Ambas usam exatamente os mesmos termos:
        f(q) = peso_pos × ‖FK(q)−alvo‖ + peso_jta × ‖q−q_ant‖ × 10
      Sem termo de rotação (trajetória define apenas posição XYZ).
      DFL: pesos ADAPTATIVOS que variam por passo → landscape realmente dinâmico.
        • peso_pos ∈ [0.7×base … 1.3×base] — cresce → prioriza precisão no fim
        • peso_jta ∈ [1.5×base … 0.5×base] — cai   → relaxa continuidade no fim
      PSO/AG: pesos FIXOS — landscape estático por passo.

  [✓] CORREÇÃO 4 — Histórico de convergência REAL para TODOS
      PSO reimplementado manualmente (PSOBase) — sem sko.PSO.
      Rastreamento iteração-por-iteração para PSO, PSO-DFL e AG.
      Gráfico de convergência agora é comparável entre os três.

  [✓] CORREÇÃO 5 — FDR (Benjamini-Hochberg) nos testes estatísticos
      15 testes (5 métricas × 3 pares) → p_ajustado reportado.

  Autores: Erick Sebba [Pesquisa em Robótica] — v4.0
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import warnings
import json
import os
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ESTRUTURA DE PASTAS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("resultados")
DIRS = {
    "convergencia": BASE_DIR / "convergencia",
    "suavidade":    BASE_DIR / "suavidade",
    "estatistica":  BASE_DIR / "estatistica",
    "diversidade":  BASE_DIR / "diversidade",
    "dados_brutos": BASE_DIR / "dados_brutos",
    "artigo":       BASE_DIR / "artigo",
}
for folder in DIRS.values():
    folder.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
N_RUNS    = 10    # execuções independentes
N_PASSOS  = 30    # pontos da trajetória
PSO_POP   = 40    # partículas
PSO_ITER  = 80    # iterações PSO
AG_POP    = 40    # indivíduos
AG_GENS   = 80    # gerações
TAXA_CROSS = 0.7
TAXA_MUT   = 0.2
PESO_POS   = 1.0  # peso base do erro cartesiano
PESO_JTA   = 0.05 # peso base da continuidade articular
SEED_BASE  = 42
PSO_W, PSO_C1, PSO_C2 = 0.7, 1.5, 1.5

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODELAGEM DO ROBÔ BIMO — Parâmetros DH
# ─────────────────────────────────────────────────────────────────────────────
DH_params = [
    [0, 184, 0,   np.pi / 2],   # Junta 1
    [0, 0,   222, 0],            # Junta 2
    [0, 0,   205, 0],            # Junta 3
    [0, 0,   0,   -np.pi / 2],  # Junta 4
    [0, 0,   0,    np.pi / 2],  # Junta 5
    [0, 75,  0,   0],            # Junta 6
]

LIMITES_INF = np.array([-np.pi, -np.pi*110/180, -np.pi*80/180,
                         -np.pi, -np.pi*50/180,  -np.pi])
LIMITES_SUP = np.array([ np.pi,  np.pi*155/180,  np.pi*210/180,
                          np.pi,  np.pi*230/180,   np.pi])

# ─────────────────────────────────────────────────────────────────────────────
# 2. CINEMÁTICA DIRETA
# ─────────────────────────────────────────────────────────────────────────────
def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ], dtype=float)

def cinematica_direta(q):
    T = np.eye(4, dtype=float)
    for i in range(6):
        th, d, a, al = DH_params[i]
        T = T @ dh_matrix(q[i] + th, d, a, al)
    return T[:3, 3]

def posicoes_dos_elos(q):
    T = np.eye(4, dtype=float)
    pts = [T[:3, 3].copy()]
    for i in range(6):
        th, d, a, al = DH_params[i]
        T = T @ dh_matrix(q[i] + th, d, a, al)
        pts.append(T[:3, 3].copy())
    return np.array(pts)

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAJETÓRIA DE TESTE
# ─────────────────────────────────────────────────────────────────────────────
x_traj = np.linspace(150, 300, N_PASSOS)
y_traj = np.linspace(100, -100, N_PASSOS)
z_traj = np.ones(N_PASSOS) * 200.0
TRAJETORIA = np.column_stack([x_traj, y_traj, z_traj])

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNÇÕES DE FITNESS — SIMÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
#
#  Ambas as funções usam EXATAMENTE os mesmos termos físicos:
#
#    f(q) = peso_pos × ‖FK(q) − alvo_xyz‖   ← erro cartesiano (mm)
#         + peso_jta × ‖q − q_ant‖ × 10     ← penalidade de continuidade (rad)
#
#  A diferença está nos PESOS:
#    - Fitness estática (PSO/AG): pesos FIXOS em todos os passos.
#    - Fitness DFL:               pesos ADAPTATIVOS que variam suavemente
#                                  ao longo da trajetória.
#
#  Modulação DFL (t ∈ [0, 1], t = passo / (N_PASSOS - 1)):
#    peso_pos(t) = PESO_POS × (0.7 + 0.6 × t)   → [0.7×base … 1.3×base]
#    peso_jta(t) = PESO_JTA × (1.5 − t)          → [1.5×base … 0.5×base]
#
#  Interpretação física:
#    No início da trajetória o DFL valoriza mais a continuidade articular
#    (suavidade no arranque). No final, valoriza mais a precisão posicional.
#    Isso muda a FORMA do vale de custo a cada passo, não apenas seu centro —
#    é o landscape que é dinâmico, não só o alvo.
# ─────────────────────────────────────────────────────────────────────────────

def criar_fitness_estatica(alvo, q_anterior,
                           peso_pos=PESO_POS, peso_jta=PESO_JTA):
    """Fitness estática (PSO padrão e AG). Pesos fixos."""
    alvo  = np.asarray(alvo, dtype=float)
    q_ant = np.asarray(q_anterior, dtype=float)

    def _f(q):
        q = np.asarray(q, dtype=float)
        return float(peso_pos * np.linalg.norm(cinematica_direta(q) - alvo)
                     + peso_jta * np.linalg.norm(q - q_ant) * 10.0)
    return _f


def criar_fitness_dfl(alvo, q_anterior, passo,
                      peso_pos_base=PESO_POS, peso_jta_base=PESO_JTA,
                      n_passos=N_PASSOS):
    """Fitness adaptativa (PSO-DFL). Pesos modulados pelo progresso.

    O landscape muda de forma (não só de centro) a cada passo:
      - peso_pos cresce: pressão por precisão aumenta ao longo da trajetória.
      - peso_jta decresce: penalidade por descontinuidade relaxa no final.

    Isso implementa o conceito de DFL genuíno: o ambiente de otimização
    tem estrutura temporal, não é apenas uma sequência de problemas estáticos.
    """
    alvo  = np.asarray(alvo, dtype=float)
    q_ant = np.asarray(q_anterior, dtype=float)

    t = passo / max(1, n_passos - 1)             # ∈ [0.0, 1.0]
    peso_pos = peso_pos_base * (0.7 + 0.6 * t)   # 0.7×base → 1.3×base
    peso_jta = peso_jta_base * (1.5 - t)          # 1.5×base → 0.5×base

    def _f(q):
        q = np.asarray(q, dtype=float)
        return float(peso_pos * np.linalg.norm(cinematica_direta(q) - alvo)
                     + peso_jta * np.linalg.norm(q - q_ant) * 10.0)
    return _f

# ─────────────────────────────────────────────────────────────────────────────
# 5. PSO BASE — implementação manual para rastreamento REAL
# ─────────────────────────────────────────────────────────────────────────────
class PSOBase:
    """PSO implementado do zero. Rastreia gbest e diversidade iteração-a-iteração.

    Parâmetros clássicos:
      w   = inércia          (0.7 → exploração balanceada)
      c1  = componente cognitivo
      c2  = componente social
    Inicialização: cold-start uniforme em [lb, ub].
    """

    def __init__(self, func, dim, pop, lb, ub, w=0.7, c1=1.5, c2=1.5):
        self.func = func
        self.dim  = dim
        self.pop  = pop
        self.lb   = np.array(lb, dtype=float)
        self.ub   = np.array(ub, dtype=float)
        self.w    = w
        self.c1   = c1
        self.c2   = c2
        # Velocidade máxima = 20% do range em cada dimensão
        self.v_max = 0.2 * (self.ub - self.lb)
        self._inicializar()

    def _inicializar(self):
        """Cold-start: população e velocidades aleatórias."""
        self.X      = np.random.uniform(self.lb, self.ub, (self.pop, self.dim))
        self.V      = np.random.uniform(-0.5, 0.5, (self.pop, self.dim)) \
                      * (self.ub - self.lb) * 0.1
        self.X_best = self.X.copy()
        f0          = np.array([self.func(x) for x in self.X])
        self.y_best = f0.copy()
        idx         = np.argmin(f0)
        self.gbest_x = self.X[idx].copy()
        self.gbest_y = float(f0[idx])
        self.hist_gbest = []
        self.hist_div   = []

    def step(self):
        """Uma iteração PSO completa."""
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = (self.w * self.V
                  + self.c1 * r1 * (self.X_best - self.X)
                  + self.c2 * r2 * (self.gbest_x - self.X))
        # Clampar velocidade
        self.V = np.clip(self.V, -self.v_max, self.v_max)
        self.X = np.clip(self.X + self.V, self.lb, self.ub)

        for i in range(self.pop):
            fi = self.func(self.X[i])
            if fi < self.y_best[i]:
                self.y_best[i]  = fi
                self.X_best[i]  = self.X[i].copy()
            if fi < self.gbest_y:
                self.gbest_y = fi
                self.gbest_x = self.X[i].copy()

        self.hist_gbest.append(float(self.gbest_y))
        self.hist_div.append(float(np.std(self.X)))

    def run(self, n_iter):
        for _ in range(n_iter):
            self.step()


# ─────────────────────────────────────────────────────────────────────────────
# 6. PSO-DFL — herda PSOBase, adiciona preservação de estado e reavaliação
# ─────────────────────────────────────────────────────────────────────────────
class PSODinamico(PSOBase):
    """PSO com Dynamic Fitness Landscape.

    Diferença em relação ao PSOBase:
      • Uma única instância é criada para toda a trajetória.
      • A cada novo passo, `atualizar_landscape()` substitui a função
        objetivo SEM reinicializar X, V, X_best, y_best nem gbest.
      • Após a troca, pbest e gbest são reavaliados no novo landscape —
        isso evita que referências obsoletas distorçam as velocidades.

    Essa reavaliação é o núcleo do conceito DFL: as partículas "sabem"
    que o ambiente mudou e recalibram suas memórias antes de continuar.
    """

    def atualizar_landscape(self, nova_func):
        """Troca landscape e reavalia memórias. Preserva posições e velocidades."""
        self.func    = nova_func
        # Reavaliar gbest no novo landscape
        self.gbest_y = float(self.func(self.gbest_x))
        # Reavaliar todos os pbest
        for i in range(self.pop):
            self.y_best[i] = float(self.func(self.X_best[i]))
        # Atualizar gbest se algum pbest for melhor
        idx = np.argmin(self.y_best)
        if self.y_best[idx] < self.gbest_y:
            self.gbest_y = self.y_best[idx]
            self.gbest_x = self.X_best[idx].copy()
        # Limpar histórico do passo anterior
        self.hist_gbest = []
        self.hist_div   = []

# ─────────────────────────────────────────────────────────────────────────────
# 7. FUNÇÕES DE EXECUÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def rodar_pso(seed=42, verbose=False):
    """PSO padrão: nova população aleatória a cada passo. Fitness estática.

    Cada ponto da trajetória é tratado como um problema IK independente.
    Não há memória entre passos — é o comportamento clássico do PSO em IK.
    """
    np.random.seed(seed)
    q_ant = np.zeros(6)
    hist_q, hist_err, hist_conv, hist_div = [], [], [], []

    t0 = time.time()
    for i, alvo in enumerate(TRAJETORIA):
        juiz = criar_fitness_estatica(alvo, q_ant)
        pso  = PSOBase(func=juiz, dim=6, pop=PSO_POP,
                       lb=LIMITES_INF, ub=LIMITES_SUP,
                       w=PSO_W, c1=PSO_C1, c2=PSO_C2)
        pso.run(PSO_ITER)

        hist_q.append(pso.gbest_x.copy())
        hist_err.append(pso.gbest_y)
        hist_conv.append(pso.hist_gbest)
        hist_div.append(pso.hist_div)
        q_ant = pso.gbest_x.copy()

        if verbose:
            print(f"  PSO ponto {i+1:2d}/{N_PASSOS} | err={pso.gbest_y:.3f}")

    return {
        "historico_q":            np.array(hist_q),
        "historico_erro":         np.array(hist_err),
        "historico_convergencia": hist_conv,
        "historico_diversidade":  hist_div,
        "tempo_s":                time.time() - t0,
    }


def rodar_pso_dfl(seed=42, verbose=False):
    """PSO-DFL: estado preservado entre passos + reavaliação + fitness adaptativa.

    Uma única instância PSODinamico percorre toda a trajetória.
    A função `atualizar_landscape()` é chamada a cada passo, mudando
    tanto o alvo quanto os pesos (landscape adaptativamente dinâmico).
    """
    np.random.seed(seed)
    q_ant = np.zeros(6)
    hist_q, hist_err, hist_conv, hist_div = [], [], [], []

    # Instância única — persiste por toda a trajetória
    juiz_0 = criar_fitness_dfl(TRAJETORIA[0], q_ant, passo=0)
    pso    = PSODinamico(func=juiz_0, dim=6, pop=PSO_POP,
                         lb=LIMITES_INF, ub=LIMITES_SUP,
                         w=PSO_W, c1=PSO_C1, c2=PSO_C2)

    t0 = time.time()
    for i, alvo in enumerate(TRAJETORIA):
        juiz = criar_fitness_dfl(alvo, q_ant, passo=i)
        pso.atualizar_landscape(juiz)   # preserva X, V, pbest, gbest
        pso.run(PSO_ITER)

        hist_q.append(pso.gbest_x.copy())
        hist_err.append(pso.gbest_y)
        hist_conv.append(pso.hist_gbest.copy())
        hist_div.append(pso.hist_div.copy())
        q_ant = pso.gbest_x.copy()

        if verbose:
            print(f"  PSO-DFL ponto {i+1:2d}/{N_PASSOS} | err={pso.gbest_y:.3f}")

    return {
        "historico_q":            np.array(hist_q),
        "historico_erro":         np.array(hist_err),
        "historico_convergencia": hist_conv,
        "historico_diversidade":  hist_div,
        "tempo_s":                time.time() - t0,
    }


def rodar_ag(seed=42, verbose=False):
    """Algoritmo Genético: nova população aleatória a cada passo. Fitness estática.

    Operadores: elitismo (top-2) + seleção por roleta +
                crossover aritmético + mutação gaussiana + renovação de 20%.
    """
    np.random.seed(seed)
    q_ant = np.zeros(6)
    hist_q, hist_err, hist_conv, hist_div = [], [], [], []

    t0 = time.time()
    for i, alvo in enumerate(TRAJETORIA):
        fn  = criar_fitness_estatica(alvo, q_ant)
        pop = np.random.uniform(LIMITES_INF, LIMITES_SUP, (AG_POP, 6))
        sc  = np.array([fn(ind) for ind in pop])

        conv_iter, div_iter = [], []

        for gen in range(AG_GENS):
            # Elitismo
            idx_e = np.argsort(sc)[:2]
            elite = pop[idx_e].copy()

            # Seleção por roleta (proporcional ao inverso do fitness)
            inv  = 1.0 / (sc + 1e-12)
            prob = inv / inv.sum()
            pais = pop[np.random.choice(AG_POP, size=AG_POP, replace=True, p=prob)]

            # Crossover aritmético + mutação gaussiana
            filhos = []
            for j in range(0, AG_POP - 2, 2):
                p1, p2 = pais[j], pais[j+1]
                if np.random.rand() < TAXA_CROSS:
                    a = np.random.rand()
                    f1 = a * p1 + (1 - a) * p2
                    f2 = (1 - a) * p1 + a * p2
                else:
                    f1, f2 = p1.copy(), p2.copy()
                # Mutação gaussiana
                for f in [f1, f2]:
                    mask = np.random.rand(6) < TAXA_MUT
                    f[mask] += np.random.normal(0, 0.1, mask.sum())
                filhos += [np.clip(f1, LIMITES_INF, LIMITES_SUP),
                           np.clip(f2, LIMITES_INF, LIMITES_SUP)]

            while len(filhos) < AG_POP - 2:
                fext = pais[-1].copy()
                mask = np.random.rand(6) < TAXA_MUT
                fext[mask] += np.random.normal(0, 0.1, mask.sum())
                filhos.append(np.clip(fext, LIMITES_INF, LIMITES_SUP))

            nova_pop = np.vstack([elite, np.array(filhos[:AG_POP - 2])])

            # Renovação de 20% dos piores não-elite
            n_renov = max(1, int(0.20 * (AG_POP - 2)))
            sc_ne   = np.array([fn(ind) for ind in nova_pop[2:]])
            idx_p   = np.argsort(sc_ne)[-n_renov:]
            nova_pop[2 + idx_p] = np.random.uniform(LIMITES_INF, LIMITES_SUP,
                                                      (n_renov, 6))
            pop = nova_pop
            sc  = np.array([fn(ind) for ind in pop])

            conv_iter.append(float(np.min(sc)))
            div_iter.append(float(np.std(pop)))

        idx_b = np.argmin(sc)
        hist_q.append(pop[idx_b].copy())
        hist_err.append(float(sc[idx_b]))
        hist_conv.append(conv_iter)
        hist_div.append(div_iter)
        q_ant = pop[idx_b].copy()

        if verbose:
            print(f"  AG  ponto {i+1:2d}/{N_PASSOS} | err={sc[idx_b]:.3f}")

    return {
        "historico_q":            np.array(hist_q),
        "historico_erro":         np.array(hist_err),
        "historico_convergencia": hist_conv,
        "historico_diversidade":  hist_div,
        "tempo_s":                time.time() - t0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 8. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
def calcular_metricas(res):
    hist_q   = res["historico_q"]
    hist_err = res["historico_erro"]

    pos_real = np.array([cinematica_direta(q) for q in hist_q])
    erros    = np.array([np.linalg.norm(pos_real[k] - TRAJETORIA[k])
                         for k in range(N_PASSOS)])

    rmse       = np.sqrt(np.mean(erros ** 2))
    suavidade  = np.sum(np.abs(np.diff(hist_q, axis=0)))
    vel_art    = np.diff(hist_q, axis=0)
    jerk       = np.sum(np.abs(np.diff(vel_art, axis=0)))
    taxa_conv  = (np.mean(np.maximum(0, hist_err[:-1] - hist_err[1:]))
                  if len(hist_err) > 1 else 0.0)

    return {
        "rmse_pos_mm":            rmse,
        "erro_medio_mm":          float(np.mean(erros)),
        "erro_max_mm":            float(np.max(erros)),
        "suavidade_rad":          suavidade,
        "jerk_articular":         jerk,
        "indice_suavidade":       suavidade / (N_PASSOS - 1),
        "taxa_convergencia":      taxa_conv,
        "fitness_medio":          float(np.mean(hist_err)),
        "tempo_s":                res["tempo_s"],
        "erros_pos":              erros,
        "historico_pos":          pos_real,
        "historico_q":            hist_q,
        "historico_convergencia": res["historico_convergencia"],
        "historico_diversidade":  res["historico_diversidade"],
    }

# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPERIMENTO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def rodar_experimento():
    print("=" * 70)
    print("EXPERIMENTO v4.0 — PSO | PSO-DFL | AG — cold-start universal")
    print(f"N_RUNS={N_RUNS} | N_PASSOS={N_PASSOS}")
    print(f"PSO/DFL: {PSO_POP}p × {PSO_ITER}it | AG: {AG_POP}ind × {AG_GENS}ger")
    print("=" * 70)

    res_pso, res_dfl, res_ag = [], [], []

    for run in range(N_RUNS):
        seed = SEED_BASE + run
        print(f"\n[Run {run+1}/{N_RUNS}] seed={seed}")

        print("  [PSO]    ", end="", flush=True)
        r = rodar_pso(seed=seed)
        m = calcular_metricas(r)
        res_pso.append(m)
        print(f"RMSE={m['rmse_pos_mm']:.3f} mm | "
              f"Suav={m['suavidade_rad']:.2f} rad | t={m['tempo_s']:.1f}s")

        print("  [PSO-DFL]", end="", flush=True)
        r = rodar_pso_dfl(seed=seed)
        m = calcular_metricas(r)
        res_dfl.append(m)
        print(f"RMSE={m['rmse_pos_mm']:.3f} mm | "
              f"Suav={m['suavidade_rad']:.2f} rad | t={m['tempo_s']:.1f}s")

        print("  [AG]     ", end="", flush=True)
        r = rodar_ag(seed=seed)
        m = calcular_metricas(r)
        res_ag.append(m)
        print(f"RMSE={m['rmse_pos_mm']:.3f} mm | "
              f"Suav={m['suavidade_rad']:.2f} rad | t={m['tempo_s']:.1f}s")

    return res_pso, res_dfl, res_ag

# ─────────────────────────────────────────────────────────────────────────────
# 10. TABELA COMPARATIVA
# ─────────────────────────────────────────────────────────────────────────────
METRICAS_TABELA = [
    ("rmse_pos_mm",      "RMSE Posição (mm)"),
    ("erro_medio_mm",    "Erro Médio (mm)"),
    ("erro_max_mm",      "Erro Máximo (mm)"),
    ("suavidade_rad",    "Suavidade (rad)"),
    ("jerk_articular",   "Jerk Articular (rad²)"),
    ("indice_suavidade", "Índice Suavidade"),
    ("taxa_convergencia","Taxa Convergência"),
    ("fitness_medio",    "Fitness Médio"),
    ("tempo_s",          "Tempo Execução (s)"),
]

def imprimir_tabela(res_pso, res_dfl, res_ag):
    print("\n" + "=" * 104)
    print(f"{'MÉTRICA':<30} {'PSO (μ±σ)':>24}  {'PSO-DFL (μ±σ)':>24}  {'AG (μ±σ)':>24}")
    print("=" * 104)

    tabela_json = []
    for chave, nome in METRICAS_TABELA:
        v_pso = np.array([r[chave] for r in res_pso])
        v_dfl = np.array([r[chave] for r in res_dfl])
        v_ag  = np.array([r[chave] for r in res_ag])
        print(f"{nome:<30} "
              f"{v_pso.mean():10.4f}±{v_pso.std():8.4f}  "
              f"{v_dfl.mean():10.4f}±{v_dfl.std():8.4f}  "
              f"{v_ag.mean():10.4f}±{v_ag.std():8.4f}")
        tabela_json.append({"Metrica": nome,
                             "PSO_mu": v_pso.mean(), "PSO_sigma": v_pso.std(),
                             "PSO_DFL_mu": v_dfl.mean(), "PSO_DFL_sigma": v_dfl.std(),
                             "AG_mu": v_ag.mean(),  "AG_sigma": v_ag.std()})
    print("=" * 104)

    p = DIRS["dados_brutos"] / "tabela_comparativa.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(tabela_json, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Tabela salva em {p}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. GRÁFICOS DE CONVERGÊNCIA
# ─────────────────────────────────────────────────────────────────────────────
CORES = {"PSO": "#1f77b4", "PSO-DFL": "#ff7f0e", "AG": "#d62728"}

def gerar_graficos_convergencia(res_pso, res_dfl, res_ag):
    print("\n[GRAFICOS] Convergência...")

    # --- Fig 1: Histórico iteração-por-iteração do ponto 1 (REAL) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Histórico de Convergência — Primeiro Ponto da Trajetória\n"
                 "(curvas individuais + média; dados REAIS, iteração-por-iteração)")
    ax.set_xlabel("Iteração / Geração")
    ax.set_ylabel("Fitness (custo composto)")

    for label, resultados, cor in [
        ("PSO",     res_pso, CORES["PSO"]),
        ("PSO-DFL", res_dfl, CORES["PSO-DFL"]),
        ("AG",      res_ag,  CORES["AG"]),
    ]:
        curvas = [np.array(r["historico_convergencia"][0]) for r in resultados
                  if r["historico_convergencia"] and len(r["historico_convergencia"][0]) > 0]
        if not curvas:
            continue
        for c in curvas:
            ax.plot(c, alpha=0.2, color=cor, lw=0.8)
        media = np.mean([c[:min(len(c), PSO_ITER)] for c in curvas], axis=0)
        ax.plot(media, lw=2.5, color=cor, label=f"{label} (média)")

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = DIRS["convergencia"] / "historico_convergencia.png"
    plt.savefig(p, dpi=300); plt.close()
    print(f"  [OK] {p}")

    # --- Fig 2: Erro por ponto da trajetória ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"Erro de Rastreamento por Ponto (N={N_RUNS} runs, μ ± σ)")
    ax.set_xlabel("Ponto da Trajetória")
    ax.set_ylabel("Erro de Posição (mm)")
    pontos = np.arange(1, N_PASSOS + 1)

    for label, resultados, cor, mk in [
        ("PSO",     res_pso, CORES["PSO"],     "o"),
        ("PSO-DFL", res_dfl, CORES["PSO-DFL"], "^"),
        ("AG",      res_ag,  CORES["AG"],       "s"),
    ]:
        mat = np.array([r["erros_pos"] for r in resultados])
        mu  = mat.mean(axis=0)
        sd  = mat.std(axis=0)
        ax.fill_between(pontos, mu - sd, mu + sd, alpha=0.18, color=cor)
        ax.plot(pontos, mu, f"-{mk}", ms=4, lw=2, color=cor, label=label)

    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = DIRS["convergencia"] / "erro_por_ponto.png"
    plt.savefig(p, dpi=300); plt.close()
    print(f"  [OK] {p}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. GRÁFICOS DE DIVERSIDADE
# ─────────────────────────────────────────────────────────────────────────────
def gerar_graficos_diversidade(res_pso, res_dfl, res_ag):
    print("\n[GRAFICOS] Diversidade...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Diversidade Populacional — Primeiro Ponto da Trajetória")

    # Painel A: Diversidade ao longo das iterações
    ax = axes[0]
    ax.set_title("Diversidade (desvio-padrão da população)")
    ax.set_xlabel("Iteração / Geração")
    ax.set_ylabel("Desvio-Padrão das Posições")

    for label, resultados, cor, mk in [
        ("PSO",     res_pso, CORES["PSO"],     "o"),
        ("PSO-DFL", res_dfl, CORES["PSO-DFL"], "^"),
        ("AG",      res_ag,  CORES["AG"],       "s"),
    ]:
        curvas = [np.array(r["historico_diversidade"][0]) for r in resultados
                  if r["historico_diversidade"] and len(r["historico_diversidade"][0]) > 0]
        if not curvas:
            continue
        n = min(len(c) for c in curvas)
        media = np.mean([c[:n] for c in curvas], axis=0)
        ax.plot(media, lw=2.5, color=cor, label=label, marker=mk,
                markevery=max(1, n // 10), ms=4)

    ax.legend(); ax.grid(True, alpha=0.3)

    # Painel B: Diversidade final vs Erro final (scatter)
    ax2 = axes[1]
    ax2.set_title("Diversidade Final vs Erro Final")
    ax2.set_xlabel("Diversidade Final (Desvio-Padrão)")
    ax2.set_ylabel("Erro Final de Posição (mm)")

    for label, resultados, cor in [
        ("PSO",     res_pso, CORES["PSO"]),
        ("PSO-DFL", res_dfl, CORES["PSO-DFL"]),
        ("AG",      res_ag,  CORES["AG"]),
    ]:
        divs, errs = [], []
        for r in resultados:
            if r["historico_diversidade"] and r["historico_diversidade"][0]:
                divs.append(r["historico_diversidade"][0][-1])
                errs.append(r["erros_pos"][0])
        if len(divs) > 2:
            ax2.scatter(divs, errs, color=cor, alpha=0.7, s=80, label=label)
            z = np.polyfit(divs, errs, 1)
            xr = np.linspace(min(divs), max(divs), 50)
            ax2.plot(xr, np.poly1d(z)(xr), "--", color=cor, lw=1.5, alpha=0.7)

    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    p = DIRS["diversidade"] / "diversidade_populacional.png"
    plt.savefig(p, dpi=300); plt.close()
    print(f"  [OK] {p}")

    # Figura extra: diversidade vs erro scatter individual
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Diversidade vs Erro Final — por Algoritmo")
    for ax, (label, resultados, cor) in zip(axes, [
        ("PSO",     res_pso, CORES["PSO"]),
        ("PSO-DFL", res_dfl, CORES["PSO-DFL"]),
        ("AG",      res_ag,  CORES["AG"]),
    ]):
        divs, errs = [], []
        for r in resultados:
            if r["historico_diversidade"] and r["historico_diversidade"][0]:
                divs.append(r["historico_diversidade"][0][-1])
                errs.append(r["erros_pos"][0])
        if len(divs) > 2:
            ax.scatter(divs, errs, color=cor, alpha=0.7, s=80)
            z  = np.polyfit(divs, errs, 1)
            xr = np.linspace(min(divs), max(divs), 50)
            ax.plot(xr, np.poly1d(z)(xr), "--", color=cor, lw=2, alpha=0.7)
        ax.set_title(f"{label}")
        ax.set_xlabel("Diversidade Final"); ax.set_ylabel("Erro Final (mm)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = DIRS["diversidade"] / "diversidade_vs_erro.png"
    plt.savefig(p, dpi=300); plt.close()
    print(f"  [OK] {p}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. GRÁFICOS DE SUAVIDADE
# ─────────────────────────────────────────────────────────────────────────────
def gerar_graficos_suavidade(res_pso, res_dfl, res_ag):
    print("\n[GRAFICOS] Suavidade...")

    # Perfis articulares do melhor run de cada algoritmo
    for label, resultados, cor in [
        ("PSO",     res_pso, CORES["PSO"]),
        ("PSO-DFL", res_dfl, CORES["PSO-DFL"]),
        ("AG",      res_ag,  CORES["AG"]),
    ]:
        idx_b = int(np.argmin([r["rmse_pos_mm"] for r in resultados]))
        q_hist = resultados[idx_b]["historico_q"]
        rmse_b = resultados[idx_b]["rmse_pos_mm"]

        fig, axes = plt.subplots(2, 3, figsize=(14, 6))
        fig.suptitle(f"Perfis Articulares — {label} (RMSE={rmse_b:.2f} mm, cold-start)")
        for j, ax in enumerate(axes.flat):
            ax.plot(np.degrees(q_hist[:, j]), "-o", color=cor, lw=2, ms=4)
            ax.set_title(f"Junta {j+1}")
            ax.set_xlabel("Ponto da Trajetória")
            ax.set_ylabel("Ângulo (°)")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = DIRS["suavidade"] / f"perfis_articulares_{label.lower().replace('-','_')}.png"
        plt.savefig(p, dpi=300); plt.close()
        print(f"  [OK] {p}")

    # Boxplots comparativos
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Comparação de Suavidade — cold-start universal")
    for ax, (chave, titulo) in zip(axes, [
        ("suavidade_rad",   "Suavidade Total (rad)"),
        ("jerk_articular",  "Jerk Articular (rad²)"),
        ("indice_suavidade","Índice Suavidade"),
    ]):
        vals_pso = [r[chave] for r in res_pso]
        vals_dfl = [r[chave] for r in res_dfl]
        vals_ag  = [r[chave] for r in res_ag]
        bp = ax.boxplot([vals_pso, vals_dfl, vals_ag],
                        patch_artist=True, widths=0.5,
                        labels=["PSO", "PSO-DFL", "AG"],
                        medianprops=dict(color="black", lw=2))
        for patch, cor in zip(bp["boxes"],
                              [CORES["PSO"], CORES["PSO-DFL"], CORES["AG"]]):
            patch.set_facecolor(cor + "88")
        ax.set_title(titulo)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = DIRS["suavidade"] / "comparacao_suavidade.png"
    plt.savefig(p, dpi=300); plt.close()
    print(f"  [OK] {p}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. ANÁLISE ESTATÍSTICA COM FDR
# ─────────────────────────────────────────────────────────────────────────────
def analise_estatistica(res_pso, res_dfl, res_ag):
    print("\n[STAT] Análise estatística com FDR (Benjamini-Hochberg)...")

    metricas = [
        ("rmse_pos_mm",     "RMSE Posição (mm)"),
        ("suavidade_rad",   "Suavidade (rad)"),
        ("jerk_articular",  "Jerk Articular (rad²)"),
        ("tempo_s",         "Tempo de Execução (s)"),
        ("taxa_convergencia","Taxa de Convergência"),
    ]

    p = DIRS["estatistica"] / "analise_estatistica.txt"
    with open(p, "w", encoding="utf-8") as f:

        def w(s=""): f.write(s + "\n")

        w("=" * 80)
        w("ANÁLISE ESTATÍSTICA v4.0 — COLD-START UNIVERSAL")
        w("Correção para múltiplos testes: FDR (Benjamini-Hochberg)")
        w("=" * 80)

        # Coletar todos os p-valores brutos para correção global
        todos_pares = []
        todos_p_brutos = []
        todos_dados = {}

        for chave, nome in metricas:
            v_pso = np.array([r[chave] for r in res_pso])
            v_dfl = np.array([r[chave] for r in res_dfl])
            v_ag  = np.array([r[chave] for r in res_ag])
            todos_dados[chave] = (v_pso, v_dfl, v_ag, nome)

            for (g1, g2, label) in [
                (v_pso, v_dfl, "PSO vs PSO-DFL"),
                (v_pso, v_ag,  "PSO vs AG"),
                (v_dfl, v_ag,  "PSO-DFL vs AG"),
            ]:
                t_stat, p_br = stats.ttest_ind(g1, g2)
                todos_pares.append((chave, label, t_stat))
                todos_p_brutos.append(p_br)

        # Correção FDR global
        _, p_ajustados, _, _ = multipletests(todos_p_brutos, alpha=0.05, method="fdr_bh")
        pair_idx = 0

        for chave, nome in metricas:
            v_pso, v_dfl, v_ag, _ = todos_dados[chave]
            w()
            w("=" * 80)
            w(f"MÉTRICA: {nome}")
            w("=" * 80)
            w(f"PSO:     μ={v_pso.mean():.6f}  σ={v_pso.std():.6f}")
            w(f"PSO-DFL: μ={v_dfl.mean():.6f}  σ={v_dfl.std():.6f}")
            w(f"AG:      μ={v_ag.mean():.6f}  σ={v_ag.std():.6f}")
            w()

            # ANOVA
            F, p_anova = stats.f_oneway(v_pso, v_dfl, v_ag)
            w(f"ANOVA  F={F:.4f}  p={p_anova:.6f}  "
              f"{'→ SIGNIFICATIVO (p<0.05)' if p_anova < 0.05 else '→ não significativo'}")

            # t-Student pairwise com FDR
            w("\nTeste t-Student pairwise (p_ajustado FDR):")
            for (c2, label, t_stat) in todos_pares[pair_idx:pair_idx + 3]:
                p_br  = todos_p_brutos[pair_idx]
                p_adj = p_ajustados[pair_idx]
                sig   = " *" if p_adj < 0.05 else ""
                w(f"  {label:<22}: t={t_stat:7.4f}  p_bruto={p_br:.6f}  "
                  f"p_adj={p_adj:.6f}{sig}")
                pair_idx += 1

        # Cohen's d
        w()
        w("=" * 80)
        w("TAMANHO DO EFEITO (Cohen's d)")
        w("=" * 80)

        def cohens_d(x, y):
            nx, ny = len(x), len(y)
            sp = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2)
                         / (nx + ny - 2))
            return (np.mean(x) - np.mean(y)) / (sp + 1e-12)

        for chave, nome in metricas:
            v_pso, v_dfl, v_ag, _ = todos_dados[chave]
            w(f"\n{nome}:")
            w(f"  PSO vs PSO-DFL: d={cohens_d(v_pso, v_dfl):7.4f}")
            w(f"  PSO vs AG:      d={cohens_d(v_pso, v_ag):7.4f}")
            w(f"  PSO-DFL vs AG:  d={cohens_d(v_dfl, v_ag):7.4f}")

        w()
        w("=" * 80)
        w(f"Total de testes: {len(todos_p_brutos)}  |  Método: FDR Benjamini-Hochberg")
        w("Interpretação de d: |d|<0.2 trivial | 0.2–0.5 pequeno | "
          "0.5–0.8 médio | >0.8 grande")
        w("=" * 80)

    print(f"  [OK] {p}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. VISUALIZAÇÃO DA FITNESS ADAPTATIVA (diagnóstico)
# ─────────────────────────────────────────────────────────────────────────────
def visualizar_fitness_adaptativa():
    """Plota como os pesos da fitness DFL variam ao longo da trajetória."""
    print("\n[GRAFICOS] Fitness adaptativa DFL...")
    passos = np.arange(N_PASSOS)
    t      = passos / max(1, N_PASSOS - 1)

    peso_pos_dfl = PESO_POS * (0.7 + 0.6 * t)
    peso_jta_dfl = PESO_JTA * (1.5 - t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Modulação de Pesos — Fitness Adaptativa DFL vs Estática")

    ax = axes[0]
    ax.plot(passos, [PESO_POS] * N_PASSOS, "--",
            color=CORES["PSO"], lw=2, label="PSO/AG (fixo)")
    ax.plot(passos, peso_pos_dfl, "-o",
            color=CORES["PSO-DFL"], lw=2, ms=4, label="PSO-DFL (adaptativo)")
    ax.set_title("Peso Posicional (peso_pos)")
    ax.set_xlabel("Passo da Trajetória")
    ax.set_ylabel("Valor do Peso")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(passos, [PESO_JTA] * N_PASSOS, "--",
             color=CORES["PSO"], lw=2, label="PSO/AG (fixo)")
    ax2.plot(passos, peso_jta_dfl, "-^",
             color=CORES["PSO-DFL"], lw=2, ms=4, label="PSO-DFL (adaptativo)")
    ax2.set_title("Peso de Continuidade (peso_jta)")
    ax2.set_xlabel("Passo da Trajetória")
    ax2.set_ylabel("Valor do Peso")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = DIRS["dados_brutos"] / "fitness_adaptativa_dfl.png"
    plt.savefig(p, dpi=300); plt.close()
    print(f"  [OK] {p}")

# ─────────────────────────────────────────────────────────────────────────────
# 16. GRÁFICO 3D DA TRAJETÓRIA
# ─────────────────────────────────────────────────────────────────────────────
def gerar_grafico_trajetoria_3d_melhorado(resultados_pso, resultados_pso_dfl, resultados_ag):
    """Gera o gráfico 3D da trajetória com zoom automático, projeções e linhas espessas."""
    print("\n[GRAFICOS] Gerando gráfico 3D da trajetória (Versão Centralizada)...")
    
    fig = plt.figure(figsize=(12, 9)) # Aumentado levemente o canvas
    ax = fig.add_subplot(111, projection='3d')
    
    # Identificar a melhor run de cada algoritmo com base no menor RMSE
    melhor_pso = min(resultados_pso, key=lambda x: x['rmse_pos_mm'])
    melhor_pso_dfl = min(resultados_pso_dfl, key=lambda x: x['rmse_pos_mm'])
    melhor_ag = min(resultados_ag, key=lambda x: x['rmse_pos_mm'])
    
    # ==========================================
    # CÁLCULO DE LIMITES (ZOOM AUTOMÁTICO)
    # ==========================================
    # Juntar todos os pontos para achar os limites exatos da caixa
    todos_x = np.concatenate([TRAJETORIA[:,0], melhor_pso['historico_pos'][:,0], melhor_pso_dfl['historico_pos'][:,0], melhor_ag['historico_pos'][:,0]])
    todos_y = np.concatenate([TRAJETORIA[:,1], melhor_pso['historico_pos'][:,1], melhor_pso_dfl['historico_pos'][:,1], melhor_ag['historico_pos'][:,1]])
    todos_z = np.concatenate([TRAJETORIA[:,2], melhor_pso['historico_pos'][:,2], melhor_pso_dfl['historico_pos'][:,2], melhor_ag['historico_pos'][:,2]])
    
    x_min, x_max = todos_x.min(), todos_x.max()
    y_min, y_max = todos_y.min(), todos_y.max()
    z_min_data, z_max = todos_z.min(), todos_z.max()
    
    # Criar uma margem de 5% ao redor dos dados
    dx = (x_max - x_min) * 0.05
    dy = (y_max - y_min) * 0.05
    dz = (z_max - z_min_data) * 0.05 if (z_max - z_min_data) > 0 else 5.0 # Proteção caso o Z seja perfeitamente constante
    
    # Definir o chão da sombra logo abaixo dos dados reais
    z_sombra = z_min_data - (dz * 1.5)
    
    # ==========================================
    # PLOTAGEM DAS TRAJETÓRIAS (Linhas e Marcadores Maiores)
    # ==========================================
    
    # 1. Plotar a Trajetória Desejada
    ax.plot(TRAJETORIA[:, 0], TRAJETORIA[:, 1], TRAJETORIA[:, 2],
            '--', color='black', linewidth=4.0, label='Trajetória desejada', zorder=4)
            
    # Projeção da desejada no plano XY (Sombra)
    ax.plot(TRAJETORIA[:, 0], TRAJETORIA[:, 1], np.full_like(TRAJETORIA[:, 2], z_sombra),
            '--', color='gray', linewidth=2.0, alpha=0.5, zorder=1)

    # 2. Plotar Trajetórias Obtidas
    # PSO
    ax.plot(melhor_pso['historico_pos'][:, 0], melhor_pso['historico_pos'][:, 1], melhor_pso['historico_pos'][:, 2],
            '-o', markersize=6, linewidth=2.5, alpha=0.85, color='#1f77b4', 
            label=f"PSO  (RMSE={melhor_pso['rmse_pos_mm']:.2f} mm)")
    
    # PSO-DFL
    ax.plot(melhor_pso_dfl['historico_pos'][:, 0], melhor_pso_dfl['historico_pos'][:, 1], melhor_pso_dfl['historico_pos'][:, 2],
            '-^', markersize=8, linewidth=3.5, alpha=0.95, color='#ff7f0e', markeredgecolor='black', markeredgewidth=0.9,
            label=f"PSO-DFL  (RMSE={melhor_pso_dfl['rmse_pos_mm']:.2f} mm)", zorder=3)
            
    # AG 
    ax.plot(melhor_ag['historico_pos'][:, 0], melhor_ag['historico_pos'][:, 1], melhor_ag['historico_pos'][:, 2],
            '-s', markersize=6, linewidth=3.0, alpha=1.0, color='#d62728', 
            label=f"AG  (RMSE={melhor_ag['rmse_pos_mm']:.2f} mm)")

    # 3. Marcações de início e fim
    ax.scatter(TRAJETORIA[0, 0], TRAJETORIA[0, 1], TRAJETORIA[0, 2], 
               marker='^', s=200, color='#2ca02c', edgecolor='black', label='Ponto inicial', zorder=5)
    ax.scatter(TRAJETORIA[-1, 0], TRAJETORIA[-1, 1], TRAJETORIA[-1, 2], 
               marker='X', s=200, color='#9467bd', edgecolor='black', label='Ponto final', zorder=5)
               
    # ==========================================
    # FORMATAÇÃO DA CAIXA 3D E CÂMERA
    # ==========================================
    
    # Aplicar o zoom exato calculado
    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)
    ax.set_zlim(z_sombra, z_max + dz)
    ax.set_box_aspect([x_max - x_min, y_max - y_min, (z_max + dz) - z_sombra])
    
    # Ajuste de câmera para melhor visualização
    ax.view_init(elev=28, azim=-30)
    ax.grid(True, linestyle=':', alpha=0.35)
    
    ax.set_title("Rastreamento do Efetuador: Desejada vs Obtida", fontsize=16, pad=20)
    ax.set_xlabel("X (mm)", labelpad=12, fontsize=11)
    ax.set_ylabel("Y (mm)", labelpad=12, fontsize=11)
    ax.set_zlabel("Z (mm)", labelpad=12, fontsize=11)
    
    # Ajuste da legenda 
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=11)
    
    plt.tight_layout()
    fig_path = DIRS["convergencia"] / "trajetoria_efetuador_3d_melhorada.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 17. GRÁFICOS 3D INDIVIDUAIS POR ALGORITMO
# ─────────────────────────────────────────────────────────────────────────────
def gerar_graficos_3d_individuais(resultados_pso, resultados_pso_dfl, resultados_ag):
    """Gera gráficos 3D individuais para cada algoritmo."""
    print("\n[GRAFICOS] Gerando gráficos 3D individuais por algoritmo...")
    
    algoritmos = [
        ("PSO", resultados_pso, CORES["PSO"]),
        ("PSO-DFL", resultados_pso_dfl, CORES["PSO-DFL"]),
        ("AG", resultados_ag, CORES["AG"]),
    ]
    
    for nome_algo, resultados, cor in algoritmos:
        # Selecionar a melhor run com base no menor RMSE
        melhor = min(resultados, key=lambda x: x['rmse_pos_mm'])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # ==========================================
        # CÁLCULO DE LIMITES
        # ==========================================
        todos_x = np.concatenate([TRAJETORIA[:, 0], melhor['historico_pos'][:, 0]])
        todos_y = np.concatenate([TRAJETORIA[:, 1], melhor['historico_pos'][:, 1]])
        todos_z = np.concatenate([TRAJETORIA[:, 2], melhor['historico_pos'][:, 2]])
        
        x_min, x_max = todos_x.min(), todos_x.max()
        y_min, y_max = todos_y.min(), todos_y.max()
        z_min_data, z_max = todos_z.min(), todos_z.max()
        
        # Margem de 5%
        dx = (x_max - x_min) * 0.05
        dy = (y_max - y_min) * 0.05
        dz = (z_max - z_min_data) * 0.05 if (z_max - z_min_data) > 0 else 5.0
        
        z_sombra = z_min_data - (dz * 1.5)
        
        # ==========================================
        # PLOTAGEM
        # ==========================================
        
        # Trajetória desejada
        ax.plot(TRAJETORIA[:, 0], TRAJETORIA[:, 1], TRAJETORIA[:, 2],
                '--', color='black', linewidth=4.0, label='Desejada', zorder=4)
        
        # Projeção da desejada
        ax.plot(TRAJETORIA[:, 0], TRAJETORIA[:, 1], np.full_like(TRAJETORIA[:, 2], z_sombra),
                '--', color='gray', linewidth=2.0, alpha=0.5, zorder=1)
        
        # Trajetória obtida
        ax.plot(melhor['historico_pos'][:, 0], melhor['historico_pos'][:, 1], melhor['historico_pos'][:, 2],
                '-o', markersize=7, linewidth=3.0, alpha=0.9, color=cor,
                label=f"{nome_algo} (RMSE={melhor['rmse_pos_mm']:.2f} mm)", zorder=3)
        
        # Pontos de início e fim
        ax.scatter(TRAJETORIA[0, 0], TRAJETORIA[0, 1], TRAJETORIA[0, 2],
                   marker='^', s=250, color='#2ca02c', edgecolor='black', linewidth=1.5,
                   label='Início', zorder=5)
        ax.scatter(TRAJETORIA[-1, 0], TRAJETORIA[-1, 1], TRAJETORIA[-1, 2],
                   marker='X', s=250, color='#9467bd', edgecolor='black', linewidth=1.5,
                   label='Fim', zorder=5)
        
        # ==========================================
        # FORMATAÇÃO
        # ==========================================
        
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)
        ax.set_zlim(z_sombra, z_max + dz)
        ax.set_box_aspect([x_max - x_min, y_max - y_min, (z_max + dz) - z_sombra])
        
        ax.view_init(elev=28, azim=-30)
        ax.grid(True, linestyle=':', alpha=0.35)
        
        ax.set_title(f"Rastreamento {nome_algo} — Trajetória 3D", fontsize=14, pad=15, weight='bold')
        ax.set_xlabel("X (mm)", labelpad=10, fontsize=11)
        ax.set_ylabel("Y (mm)", labelpad=10, fontsize=11)
        ax.set_zlabel("Z (mm)", labelpad=10, fontsize=11)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        fig_path = DIRS["convergencia"] / f"trajetoria_3d_{nome_algo.lower().replace('-', '_')}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 18. VISUALIZAÇÕES DE QUALIDADE PARA O ARTIGO
# ─────────────────────────────────────────────────────────────────────────────
def gerar_graficos_artigo(res_pso, res_dfl, res_ag):
    """
    Gera 5 figuras de alta qualidade para o artigo:
      Fig.1 – Trajetória 3D desejada vs PSO vs PSO-DFL vs AG (melhor run)
      Fig.2 – Erro de rastreamento por ponto (PSO vs PSO-DFL vs AG sobrepostos)
      Fig.3 – Ângulos articulares – PSO (melhor run)
      Fig.4 – Ângulos articulares – PSO-DFL (melhor run)
      Fig.5 – Ângulos articulares – AG (melhor run)
      Fig.6 – Boxplot comparativo das métricas principais
    """
    print("\n[ARTIGO] Gerando visualizações de qualidade para o artigo...")
    
    # Identifica melhor run de cada algoritmo (menor RMSE)
    idx_best_pso = int(np.argmin([r["rmse_pos_mm"] for r in res_pso]))
    idx_best_dfl = int(np.argmin([r["rmse_pos_mm"] for r in res_dfl]))
    idx_best_ag  = int(np.argmin([r["rmse_pos_mm"] for r in res_ag]))
    
    best_pso = res_pso[idx_best_pso]
    best_dfl = res_dfl[idx_best_dfl]
    best_ag  = res_ag[idx_best_ag]
    
    cor_pso = "#1f77b4"   # azul
    cor_dfl = "#ff7f0e"   # laranja
    cor_ag  = "#d62728"   # vermelho
    cor_traj = "#2ca02c"  # verde
    
    # ── FIGURA 1: Trajetória 3D ───────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")
    
    # Trajetória desejada
    ax.plot(*TRAJETORIA.T, "--", color="black", lw=2.5, label="Trajetória desejada")
    
    # Trajetórias obtidas
    ax.plot(*best_pso["historico_pos"].T, "-", color=cor_pso, lw=2.2,
            marker="o", ms=4, alpha=0.8, label=f"PSO (RMSE={best_pso['rmse_pos_mm']:.2f} mm)")
    ax.plot(*best_dfl["historico_pos"].T, "-", color=cor_dfl, lw=2.2,
            marker="^", ms=5, alpha=0.85, markeredgecolor="black", markeredgewidth=0.8,
            label=f"PSO-DFL (RMSE={best_dfl['rmse_pos_mm']:.2f} mm)")
    ax.plot(*best_ag["historico_pos"].T, "-", color=cor_ag, lw=2.2,
            marker="s", ms=4, alpha=0.8, label=f"AG (RMSE={best_ag['rmse_pos_mm']:.2f} mm)")
    
    # Marcadores de início e fim
    ax.scatter(*TRAJETORIA[0], color="#2ca02c", marker="^", s=200, 
               edgecolor="black", linewidth=1.5, label="Ponto inicial", zorder=5)
    ax.scatter(*TRAJETORIA[-1], color="#9467bd",  marker="X", s=200, 
               edgecolor="black", linewidth=1.5, label="Ponto final", zorder=5)
    
    ax.set_xlabel("X (mm)", fontsize=12, labelpad=10)
    ax.set_ylabel("Y (mm)", fontsize=12, labelpad=10)
    ax.set_zlabel("Z (mm)", fontsize=12, labelpad=10)
    ax.set_title("Comparativo 3D: Trajetória Desejada vs Obtida (melhor run)", 
                 fontsize=13, fontweight="bold", pad=20)
    ax.view_init(elev=28, azim=-30)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = DIRS["artigo"] / "fig1_trajetoria_3d_comparativo.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {fig_path}")
    
    # ── FIGURA 2: Erro de rastreamento por ponto ──────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    pontos = np.arange(1, N_PASSOS + 1)
    
    # Calcular médias e desvios padrão para cada algoritmo
    erros_pso_mat = np.array([r["erros_pos"] for r in res_pso])
    erros_dfl_mat = np.array([r["erros_pos"] for r in res_dfl])
    erros_ag_mat  = np.array([r["erros_pos"] for r in res_ag])
    
    mu_pso = erros_pso_mat.mean(axis=0)
    sd_pso = erros_pso_mat.std(axis=0)
    mu_dfl = erros_dfl_mat.mean(axis=0)
    sd_dfl = erros_dfl_mat.std(axis=0)
    mu_ag  = erros_ag_mat.mean(axis=0)
    sd_ag  = erros_ag_mat.std(axis=0)
    
    # Plotar com intervalos de confiança
    ax.fill_between(pontos, mu_pso - sd_pso, mu_pso + sd_pso, alpha=0.18, color=cor_pso)
    ax.plot(pontos, mu_pso, "-o", ms=5, lw=2.5, color=cor_pso,
            label=f"PSO (μ={mu_pso.mean():.2f} mm)")
    
    ax.fill_between(pontos, mu_dfl - sd_dfl, mu_dfl + sd_dfl, alpha=0.18, color=cor_dfl)
    ax.plot(pontos, mu_dfl, "-^", ms=6, lw=2.5, color=cor_dfl,
            label=f"PSO-DFL (μ={mu_dfl.mean():.2f} mm)")
    
    ax.fill_between(pontos, mu_ag - sd_ag, mu_ag + sd_ag, alpha=0.18, color=cor_ag)
    ax.plot(pontos, mu_ag,  "-s", ms=5, lw=2.5, color=cor_ag,
            label=f"AG (μ={mu_ag.mean():.2f} mm)")
    
    ax.set_title(f"Erro de Rastreamento por Ponto da Trajetória (N={N_RUNS} execuções, μ ± σ)", 
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Ponto da Trajetória", fontsize=11)
    ax.set_ylabel("Erro de Posição (mm)", fontsize=11)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    fig_path = DIRS["artigo"] / "fig2_erro_rastreamento.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {fig_path}")
    
    # ── FIGURAS 3, 4, 5: Ângulos articulares (individual para cada algoritmo) ──
    fig_num = 3
    for nome, dados, cor in [
        ("PSO", best_pso, cor_pso), 
        ("PSO-DFL", best_dfl, cor_dfl),
        ("AG", best_ag, cor_ag)
    ]:
        fig, ax = plt.subplots(figsize=(13, 5))
        
        cores_juntas = plt.cm.Set3(np.linspace(0, 1, 6))
        
        for j in range(6):
            ax.plot(np.degrees(dados["historico_q"][:, j]),
                    marker="o", ms=3.5, lw=2.2, label=f"Junta {j+1}",
                    color=cores_juntas[j])
        
        ax.set_title(f"Perfil Articular – {nome} (melhor run, RMSE={dados['rmse_pos_mm']:.2f} mm)", 
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Ponto da Trajetória", fontsize=11)
        ax.set_ylabel("Ângulo (graus)", fontsize=11)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, ncol=1)
        ax.grid(True, linestyle="--", alpha=0.4)
        
        plt.tight_layout()
        fname = f"fig{fig_num}_angulos_{nome.lower().replace('-', '_')}.png"
        fig_path = DIRS["artigo"] / fname
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [OK] {fig_path}")
        fig_num += 1
    
    # ── FIGURA 6: Boxplot comparativo ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    metricas_box = [
        ("rmse_pos_mm",   "RMSE Posição (mm)"),
        ("suavidade_rad", "Suavidade (rad)"),
        ("tempo_s",       "Tempo de Execução (s)"),
    ]
    
    for ax, (chave, titulo) in zip(axes, metricas_box):
        vals_pso = [r[chave] for r in res_pso]
        vals_dfl = [r[chave] for r in res_dfl]
        vals_ag  = [r[chave] for r in res_ag]
        
        bp = ax.boxplot([vals_pso, vals_dfl, vals_ag],
                        patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", lw=2.5),
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        bp["boxes"][0].set_facecolor(cor_pso + "cc")
        bp["boxes"][1].set_facecolor(cor_dfl + "cc")
        bp["boxes"][2].set_facecolor(cor_ag + "cc")
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["PSO", "PSO-DFL", "AG"], fontsize=11)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    
    fig.suptitle(f"Comparativo PSO vs PSO-DFL vs AG (N={N_RUNS} execuções independentes)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = DIRS["artigo"] / "fig6_boxplot_comparativo.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {fig_path}")
    
    print(f"\n[ARTIGO] Todas as visualizações foram salvas em: {DIRS['artigo']}")

# ─────────────────────────────────────────────────────────────────────────────
# 19. GRID 3D COMPARATIVO DE PONTOS DA TRAJETÓRIA
# ─────────────────────────────────────────────────────────────────────────────
def gerar_grid_3d_pontos_trajetoria(res_pso, res_dfl, res_ag):
    """
    Gera um grid de subplots 3D mostrando o braço robótico posicionado
    em pontos específicos da trajetória para cada algoritmo.
    
    Layout: 3 linhas (PSO, PSO-DFL, AG) x 5 colunas (pontos selecionados)
    Mostra: estrutura do braço (elos) e ponto final do efetuador
    """
    print("\n[ARTIGO] Gerando grid 3D do posicionamento do braço robótico...")
    
    # Identifica melhor run de cada algoritmo
    idx_best_pso = int(np.argmin([r["rmse_pos_mm"] for r in res_pso]))
    idx_best_dfl = int(np.argmin([r["rmse_pos_mm"] for r in res_dfl]))
    idx_best_ag  = int(np.argmin([r["rmse_pos_mm"] for r in res_ag]))
    
    best_pso = res_pso[idx_best_pso]
    best_dfl = res_dfl[idx_best_dfl]
    best_ag  = res_ag[idx_best_ag]
    
    # Selecionar 5 pontos distribuídos ao longo da trajetória
    indices_pontos = [0, 7, 14, 21, 29]  # Pontos: 1, 8, 15, 22, 30
    
    cor_pso = "#1f77b4"
    cor_dfl = "#ff7f0e"
    cor_ag  = "#d62728"
    
    # Criar figura com grid 3x5
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Comparativo 3D: Posicionamento do Braço Robótico — PSO vs PSO-DFL vs AG", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    algoritmos = [
        ("PSO", best_pso, cor_pso),
        ("PSO-DFL", best_dfl, cor_dfl),
        ("AG", best_ag, cor_ag),
    ]
    
    for row, (nome_algo, dados_algo, cor_algo) in enumerate(algoritmos):
        for col, idx_ponto in enumerate(indices_pontos):
            # Criar subplot
            ax = fig.add_subplot(3, 5, row * 5 + col + 1, projection="3d")
            
            # Obter ângulos das juntas para o ponto
            q_obtido = dados_algo["historico_q"][idx_ponto]
            
            # Calcular posições dos elos do braço
            elos_posicoes = posicoes_dos_elos(q_obtido)
            
            pos_desejada = TRAJETORIA[idx_ponto]
            pos_obtida = dados_algo["historico_pos"][idx_ponto]
            erro = np.linalg.norm(pos_desejada - pos_obtida)
            
            # Plotar base do robô (origem)
            ax.scatter(0, 0, 0, marker="X", s=150, color="black", 
                      edgecolor="white", linewidth=1.5, label="Base", zorder=6)
            
            # Plotar estrutura do braço (elos conectados)
            ax.plot(elos_posicoes[:, 0], elos_posicoes[:, 1], elos_posicoes[:, 2],
                   "-", color=cor_algo, lw=3.5, alpha=0.9, label="Braço obtido", zorder=4)
            
            # Plotar juntas (pequenos círculos ao longo dos elos)
            ax.scatter(elos_posicoes[:-1, 0], elos_posicoes[:-1, 1], elos_posicoes[:-1, 2],
                      marker="o", s=60, color=cor_algo, alpha=0.7, edgecolor="black", 
                      linewidth=0.8, zorder=5)
            
            # Plotar ponto final (efetuador) obtido - MAIOR
            ax.scatter(*pos_obtida, marker="o", s=200, color=cor_algo, 
                      alpha=0.95, edgecolor="black", linewidth=2, 
                      label="Efetuador obtido", zorder=7)
            
            # Plotar ponto final desejado (referência)
            ax.scatter(*pos_desejada, marker="*", s=300, color="green", 
                      alpha=0.8, edgecolor="darkgreen", linewidth=1.5,
                      label="Desejado", zorder=6)
            
            # Linha de erro (se significativo)
            if erro > 0.5:
                ax.plot([pos_desejada[0], pos_obtida[0]],
                       [pos_desejada[1], pos_obtida[1]],
                       [pos_desejada[2], pos_obtida[2]],
                       "--", color="red", lw=2, alpha=0.7, zorder=3)
            
            # Configurações de visualização
            ax.set_xlabel("X (mm)", fontsize=8)
            ax.set_ylabel("Y (mm)", fontsize=8)
            ax.set_zlabel("Z (mm)", fontsize=8)
            ax.tick_params(labelsize=7)
            
            # Título com informações
            titulo = f"{nome_algo} - Ponto {idx_ponto + 1}"
            ax.set_title(titulo, fontsize=10, fontweight="bold", pad=10)
            
            # Grid
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=20, azim=-45)
            
            # Limites de visualização 3D
            ax.set_xlim(-150, 450)
            ax.set_ylim(-300, 300)
            ax.set_zlim(0, 500)
            
            # Adicionar texto com erro
            erro_texto = f"Erro: {erro:.2f} mm"
            cor_fundo = "lightcoral" if erro > 3 else ("lightyellow" if erro > 1 else "lightgreen")
            ax.text2D(0.5, 0.05, erro_texto, transform=ax.transAxes,
                     fontsize=8, ha="center", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor=cor_fundo, alpha=0.85, 
                              edgecolor="black", linewidth=0.8))
    
    plt.tight_layout()
    fig_path = DIRS["artigo"] / "fig7_grid_3d_posicionamento_braco.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 20. COMPARAÇÃO 3D - PONTOS CHAVE (INICIAL, MEIO, FINAL)
# ─────────────────────────────────────────────────────────────────────────────
def gerar_comparacao_pontos_chave_3d(res_pso, res_dfl, res_ag):
    """
    Compara PSO, PSO-DFL e AG em pontos-chave (inicial, meio, final).
    Grid 3x3 mostrando a estrutura do braço em cada posição.
    """
    print("\n[ARTIGO] Gerando comparação de pontos-chave 3D...")
    
    # Identifica melhor run de cada algoritmo
    idx_best_pso = int(np.argmin([r["rmse_pos_mm"] for r in res_pso]))
    idx_best_dfl = int(np.argmin([r["rmse_pos_mm"] for r in res_dfl]))
    idx_best_ag  = int(np.argmin([r["rmse_pos_mm"] for r in res_ag]))
    
    best_pso = res_pso[idx_best_pso]
    best_dfl = res_dfl[idx_best_dfl]
    best_ag  = res_ag[idx_best_ag]
    
    # Pontos-chave: inicial, meio, final
    pontos_indice = [0, N_PASSOS // 2, N_PASSOS - 1]
    pontos_nome = ["Inicial", "Meio", "Final"]
    
    cor_pso = "#1f77b4"
    cor_dfl = "#ff7f0e"
    cor_ag  = "#d62728"
    
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("Comparação 3D: Braço Robótico em Pontos-Chave — PSO vs PSO-DFL vs AG", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    for idx_col, (ponto_idx, ponto_nome) in enumerate(zip(pontos_indice, pontos_nome)):
        algoritmos = [
            ("PSO", best_pso, cor_pso, 1),
            ("PSO-DFL", best_dfl, cor_dfl, 2),
            ("AG", best_ag, cor_ag, 3),
        ]
        
        for nome_algo, dados_algo, cor_algo, idx_row in algoritmos:
            # Criar subplot (3 linhas x 3 colunas)
            ax = fig.add_subplot(3, 3, (idx_row - 1) * 3 + idx_col + 1, projection="3d")
            
            # Obter ângulos e posições
            q_obtido = dados_algo["historico_q"][ponto_idx]
            elos_posicoes = posicoes_dos_elos(q_obtido)
            
            pos_desejada = TRAJETORIA[ponto_idx]
            pos_obtida = dados_algo["historico_pos"][ponto_idx]
            erro = np.linalg.norm(pos_desejada - pos_obtida)
            
            # Plotar base
            ax.scatter(0, 0, 0, marker="X", s=200, color="black", 
                      edgecolor="white", linewidth=2, zorder=6)
            
            # Plotar braço (elos)
            ax.plot(elos_posicoes[:, 0], elos_posicoes[:, 1], elos_posicoes[:, 2],
                   "-", color=cor_algo, lw=4, alpha=0.9, zorder=4)
            
            # Plotar juntas
            ax.scatter(elos_posicoes[:-1, 0], elos_posicoes[:-1, 1], elos_posicoes[:-1, 2],
                      marker="o", s=80, color=cor_algo, alpha=0.7, edgecolor="black", 
                      linewidth=1, zorder=5)
            
            # Efetuador obtido
            ax.scatter(*pos_obtida, marker="o", s=250, color=cor_algo, 
                      alpha=1, edgecolor="black", linewidth=2.5, zorder=7)
            
            # Efetuador desejado
            ax.scatter(*pos_desejada, marker="*", s=400, color="green", 
                      alpha=0.85, edgecolor="darkgreen", linewidth=2, zorder=6)
            
            # Linha de erro
            if erro > 0.5:
                ax.plot([pos_desejada[0], pos_obtida[0]],
                       [pos_desejada[1], pos_obtida[1]],
                       [pos_desejada[2], pos_obtida[2]],
                       "--", color="red", lw=2.5, alpha=0.8, zorder=3)
            
            # Labels e configurações
            ax.set_xlabel("X (mm)", fontsize=9)
            ax.set_ylabel("Y (mm)", fontsize=9)
            ax.set_zlabel("Z (mm)", fontsize=9)
            ax.tick_params(labelsize=8)
            
            # Título
            titulo = f"{nome_algo} - {ponto_nome}"
            ax.set_title(titulo, fontsize=11, fontweight="bold", pad=12)
            
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=22, azim=-40)
            
            # Limites uniformes
            ax.set_xlim(-150, 450)
            ax.set_ylim(-300, 300)
            ax.set_zlim(0, 500)
            
            # Caixa de erro
            erro_texto = f"Erro: {erro:.2f} mm"
            cor_fundo = "lightcoral" if erro > 3 else ("lightyellow" if erro > 1 else "lightgreen")
            ax.text2D(0.5, 0.08, erro_texto, transform=ax.transAxes,
                     fontsize=9, ha="center", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=cor_fundo, 
                              alpha=0.9, edgecolor="black", linewidth=1))
    
    plt.tight_layout()
    fig_path = DIRS["artigo"] / "fig8_comparacao_pontos_chave_3d.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CINEMÁTICA INVERSA BIMO 6-DOF — v4.0")
    print("Comparativo justo: PSO | PSO-DFL | AG | cold-start universal")
    print("=" * 70 + "\n")

    # Diagrama da fitness adaptativa (não depende do experimento)
    visualizar_fitness_adaptativa()

    # Experimento principal
    res_pso, res_dfl, res_ag = rodar_experimento()

    # Tabela
    print("\n[TABELA]")
    imprimir_tabela(res_pso, res_dfl, res_ag)

    # Gráficos
    gerar_graficos_convergencia(res_pso, res_dfl, res_ag)
    gerar_graficos_diversidade(res_pso, res_dfl, res_ag)
    gerar_graficos_suavidade(res_pso, res_dfl, res_ag)
    gerar_grafico_trajetoria_3d_melhorado(res_pso, res_dfl, res_ag)
    gerar_graficos_3d_individuais(res_pso, res_dfl, res_ag)
    
    # Visualizações para o artigo (qualidade alta)
    gerar_graficos_artigo(res_pso, res_dfl, res_ag)
    gerar_grid_3d_pontos_trajetoria(res_pso, res_dfl, res_ag)
    gerar_comparacao_pontos_chave_3d(res_pso, res_dfl, res_ag)

    # Estatística
    analise_estatistica(res_pso, res_dfl, res_ag)

    print("\n" + "=" * 70)
    print("[CONCLUÍDO] Todos os resultados salvos em:", BASE_DIR)
    print("=" * 70 + "\n")