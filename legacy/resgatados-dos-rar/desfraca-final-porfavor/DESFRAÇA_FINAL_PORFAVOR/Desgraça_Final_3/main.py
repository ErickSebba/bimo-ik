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
import time
import warnings
import json
import plot as plot_lib
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
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CINEMÁTICA INVERSA BIMO 6-DOF — v4.0")
    print("Comparativo justo: PSO | PSO-DFL | AG | cold-start universal")
    print("=" * 70 + "\n")

    # Experimento principal
    res_pso, res_dfl, res_ag = rodar_experimento()

    # Tabela
    print("\n[TABELA]")
    imprimir_tabela(res_pso, res_dfl, res_ag)

    # Plotagens via modulo plot.py (somente visualizacao)
    plot_lib.gerar_grafico_trajetoria(
        res_pso,
        res_dfl,
        res_ag,
        TRAJETORIA,
        cinematica_direta,
        str(DIRS["convergencia"]),
    )
    plot_lib.gerar_grafico_3d_estatico_comparativo(
        res_pso,
        res_dfl,
        res_ag,
        TRAJETORIA,
        posicoes_dos_elos,
        cinematica_direta,
        str(DIRS["convergencia"]),
    )
    plot_lib.gerar_animacao_3d_comparativa(
        res_pso,
        res_dfl,
        res_ag,
        TRAJETORIA,
        posicoes_dos_elos,
        cinematica_direta,
        str(DIRS["convergencia"]),
    )

    # Estatística
    analise_estatistica(res_pso, res_dfl, res_ag)

    print("\n" + "=" * 70)
    print("[CONCLUÍDO] Todos os resultados salvos em:", BASE_DIR)
    print("=" * 70 + "\n")
