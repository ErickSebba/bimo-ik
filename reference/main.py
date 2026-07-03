"""
=============================================================================
  CINEMATICA INVERSA DO ROBO BIMO (6-DOF) - v4.0
  Comparativo JUSTO: PSO | PSO-DFL | AG

  [FROZEN REFERENCE COPY - see reference/README.md. Logic unchanged;
   comments/strings re-encoded from a mojibake'd attachment.]

  CORRECOES CRITICAS APLICADAS (v4.0):
  - CORRECAO 1: Sem warm-start universal (cold-start em todos).
  - CORRECAO 2: PSO-DFL = mesma implementacao base do PSO, mas estado
    (X, V, pbest, gbest) preservado entre passos + reavaliacao ao trocar
    landscape.
  - CORRECAO 3: Fitness simetricas; DFL usa pesos adaptativos por passo.
  - CORRECAO 4: Historico de convergencia real (PSO manual, sem sko).
  - CORRECAO 5: FDR (Benjamini-Hochberg) nos testes estatisticos.

  Autores: Erick Sebba [Pesquisa em Robotica] - v4.0
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

# -----------------------------------------------------------------------------
# ESTRUTURA DE PASTAS
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# CONFIGURACAO GLOBAL
# -----------------------------------------------------------------------------
N_RUNS    = 10    # execucoes independentes
N_PASSOS  = 30    # pontos da trajetoria
PSO_POP   = 40    # particulas
PSO_ITER  = 80    # iteracoes PSO
AG_POP    = 40    # individuos
AG_GENS   = 80    # geracoes
TAXA_CROSS = 0.7
TAXA_MUT   = 0.2
PESO_POS   = 1.0  # peso base do erro cartesiano
PESO_JTA   = 0.05 # peso base da continuidade articular
SEED_BASE  = 42
PSO_W, PSO_C1, PSO_C2 = 0.7, 1.5, 1.5

# -----------------------------------------------------------------------------
# 1. MODELAGEM DO ROBO BIMO - Parametros DH
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 2. CINEMATICA DIRETA
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. TRAJETORIA DE TESTE
# -----------------------------------------------------------------------------
x_traj = np.linspace(150, 300, N_PASSOS)
y_traj = np.linspace(100, -100, N_PASSOS)
z_traj = np.ones(N_PASSOS) * 200.0
TRAJETORIA = np.column_stack([x_traj, y_traj, z_traj])

# -----------------------------------------------------------------------------
# 4. FUNCOES DE FITNESS - SIMETRICAS
# -----------------------------------------------------------------------------
#  f(q) = peso_pos * ||FK(q) - alvo||  +  peso_jta * ||q - q_ant|| * 10
#  Fitness estatica (PSO/AG): pesos fixos.
#  Fitness DFL: pesos adaptativos com t = passo / (N_PASSOS - 1):
#    peso_pos(t) = PESO_POS * (0.7 + 0.6 t)   -> [0.7x .. 1.3x]
#    peso_jta(t) = PESO_JTA * (1.5 - t)        -> [1.5x .. 0.5x]
# -----------------------------------------------------------------------------

def criar_fitness_estatica(alvo, q_anterior,
                           peso_pos=PESO_POS, peso_jta=PESO_JTA):
    """Fitness estatica (PSO padrao e AG). Pesos fixos."""
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
    """Fitness adaptativa (PSO-DFL). Pesos modulados pelo progresso."""
    alvo  = np.asarray(alvo, dtype=float)
    q_ant = np.asarray(q_anterior, dtype=float)

    t = passo / max(1, n_passos - 1)             # in [0.0, 1.0]
    peso_pos = peso_pos_base * (0.7 + 0.6 * t)   # 0.7x -> 1.3x
    peso_jta = peso_jta_base * (1.5 - t)          # 1.5x -> 0.5x

    def _f(q):
        q = np.asarray(q, dtype=float)
        return float(peso_pos * np.linalg.norm(cinematica_direta(q) - alvo)
                     + peso_jta * np.linalg.norm(q - q_ant) * 10.0)
    return _f

# -----------------------------------------------------------------------------
# 5. PSO BASE - implementacao manual para rastreamento REAL
# -----------------------------------------------------------------------------
class PSOBase:
    """PSO implementado do zero. Rastreia gbest e diversidade por iteracao."""

    def __init__(self, func, dim, pop, lb, ub, w=0.7, c1=1.5, c2=1.5):
        self.func = func
        self.dim  = dim
        self.pop  = pop
        self.lb   = np.array(lb, dtype=float)
        self.ub   = np.array(ub, dtype=float)
        self.w    = w
        self.c1   = c1
        self.c2   = c2
        # Velocidade maxima = 20% do range em cada dimensao
        self.v_max = 0.2 * (self.ub - self.lb)
        self._inicializar()

    def _inicializar(self):
        """Cold-start: populacao e velocidades aleatorias."""
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
        """Uma iteracao PSO completa."""
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


# -----------------------------------------------------------------------------
# 6. PSO-DFL - herda PSOBase, adiciona preservacao de estado e reavaliacao
# -----------------------------------------------------------------------------
class PSODinamico(PSOBase):
    """PSO com Dynamic Fitness Landscape.

    - Uma unica instancia e criada para toda a trajetoria.
    - A cada novo passo, `atualizar_landscape()` troca a funcao objetivo
      SEM reinicializar X, V, X_best, y_best nem gbest.
    - Apos a troca, pbest e gbest sao reavaliados no novo landscape.
    """

    def atualizar_landscape(self, nova_func):
        """Troca landscape e reavalia memorias. Preserva posicoes e velocidades."""
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
        # Limpar historico do passo anterior
        self.hist_gbest = []
        self.hist_div   = []

# -----------------------------------------------------------------------------
# 7. FUNCOES DE EXECUCAO
# -----------------------------------------------------------------------------

def rodar_pso(seed=42, verbose=False):
    """PSO padrao: nova populacao aleatoria a cada passo. Fitness estatica."""
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
    """PSO-DFL: estado preservado entre passos + reavaliacao + fitness adaptativa."""
    np.random.seed(seed)
    q_ant = np.zeros(6)
    hist_q, hist_err, hist_conv, hist_div = [], [], [], []

    # Instancia unica - persiste por toda a trajetoria
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
    """Algoritmo Genetico: nova populacao aleatoria a cada passo. Fitness estatica.

    Operadores: elitismo (top-2) + selecao por roleta +
                crossover aritmetico + mutacao gaussiana + renovacao de 20%.
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

            # Selecao por roleta (proporcional ao inverso do fitness)
            inv  = 1.0 / (sc + 1e-12)
            prob = inv / inv.sum()
            pais = pop[np.random.choice(AG_POP, size=AG_POP, replace=True, p=prob)]

            # Crossover aritmetico + mutacao gaussiana
            filhos = []
            for j in range(0, AG_POP - 2, 2):
                p1, p2 = pais[j], pais[j+1]
                if np.random.rand() < TAXA_CROSS:
                    a = np.random.rand()
                    f1 = a * p1 + (1 - a) * p2
                    f2 = (1 - a) * p1 + a * p2
                else:
                    f1, f2 = p1.copy(), p2.copy()
                # Mutacao gaussiana
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

            # Renovacao de 20% dos piores nao-elite
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

# -----------------------------------------------------------------------------
# 8. METRICAS
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 9. EXPERIMENTO PRINCIPAL
# -----------------------------------------------------------------------------
def rodar_experimento():
    print("=" * 70)
    print("EXPERIMENTO v4.0 - PSO | PSO-DFL | AG - cold-start universal")
    print(f"N_RUNS={N_RUNS} | N_PASSOS={N_PASSOS}")
    print(f"PSO/DFL: {PSO_POP}p x {PSO_ITER}it | AG: {AG_POP}ind x {AG_GENS}ger")
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

# -----------------------------------------------------------------------------
# 10. TABELA COMPARATIVA
# -----------------------------------------------------------------------------
METRICAS_TABELA = [
    ("rmse_pos_mm",      "RMSE Posicao (mm)"),
    ("erro_medio_mm",    "Erro Medio (mm)"),
    ("erro_max_mm",      "Erro Maximo (mm)"),
    ("suavidade_rad",    "Suavidade (rad)"),
    ("jerk_articular",   "Jerk Articular (rad2)"),
    ("indice_suavidade", "Indice Suavidade"),
    ("taxa_convergencia","Taxa Convergencia"),
    ("fitness_medio",    "Fitness Medio"),
    ("tempo_s",          "Tempo Execucao (s)"),
]

def imprimir_tabela(res_pso, res_dfl, res_ag):
    print("\n" + "=" * 104)
    print(f"{'METRICA':<30} {'PSO (mu+-sd)':>24}  {'PSO-DFL (mu+-sd)':>24}  {'AG (mu+-sd)':>24}")
    print("=" * 104)

    tabela_json = []
    for chave, nome in METRICAS_TABELA:
        v_pso = np.array([r[chave] for r in res_pso])
        v_dfl = np.array([r[chave] for r in res_dfl])
        v_ag  = np.array([r[chave] for r in res_ag])
        print(f"{nome:<30} "
              f"{v_pso.mean():10.4f}+-{v_pso.std():8.4f}  "
              f"{v_dfl.mean():10.4f}+-{v_dfl.std():8.4f}  "
              f"{v_ag.mean():10.4f}+-{v_ag.std():8.4f}")
        tabela_json.append({"Metrica": nome,
                            "PSO_mu": v_pso.mean(), "PSO_sigma": v_pso.std(),
                            "PSO_DFL_mu": v_dfl.mean(), "PSO_DFL_sigma": v_dfl.std(),
                            "AG_mu": v_ag.mean(),  "AG_sigma": v_ag.std()})
    print("=" * 104)

    p = DIRS["dados_brutos"] / "tabela_comparativa.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(tabela_json, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Tabela salva em {p}")

# -----------------------------------------------------------------------------
# 14. ANALISE ESTATISTICA COM FDR
# -----------------------------------------------------------------------------
def analise_estatistica(res_pso, res_dfl, res_ag):
    print("\n[STAT] Analise estatistica com FDR (Benjamini-Hochberg)...")

    metricas = [
        ("rmse_pos_mm",     "RMSE Posicao (mm)"),
        ("suavidade_rad",   "Suavidade (rad)"),
        ("jerk_articular",  "Jerk Articular (rad2)"),
        ("tempo_s",         "Tempo de Execucao (s)"),
        ("taxa_convergencia","Taxa de Convergencia"),
    ]

    p = DIRS["estatistica"] / "analise_estatistica.txt"
    with open(p, "w", encoding="utf-8") as f:

        def w(s=""): f.write(s + "\n")

        w("=" * 80)
        w("ANALISE ESTATISTICA v4.0 - COLD-START UNIVERSAL")
        w("Correcao para multiplos testes: FDR (Benjamini-Hochberg)")
        w("=" * 80)

        # Coletar todos os p-valores brutos para correcao global
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

        # Correcao FDR global
        _, p_ajustados, _, _ = multipletests(todos_p_brutos, alpha=0.05, method="fdr_bh")
        pair_idx = 0

        for chave, nome in metricas:
            v_pso, v_dfl, v_ag, _ = todos_dados[chave]
            w()
            w("=" * 80)
            w(f"METRICA: {nome}")
            w("=" * 80)
            w(f"PSO:     mu={v_pso.mean():.6f}  sd={v_pso.std():.6f}")
            w(f"PSO-DFL: mu={v_dfl.mean():.6f}  sd={v_dfl.std():.6f}")
            w(f"AG:      mu={v_ag.mean():.6f}  sd={v_ag.std():.6f}")
            w()

            # ANOVA
            F, p_anova = stats.f_oneway(v_pso, v_dfl, v_ag)
            w(f"ANOVA  F={F:.4f}  p={p_anova:.6f}  "
              f"{'SIGNIFICATIVO (p<0.05)' if p_anova < 0.05 else 'nao significativo'}")

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
        w(f"Total de testes: {len(todos_p_brutos)}  |  Metodo: FDR Benjamini-Hochberg")
        w("Interpretacao de d: |d|<0.2 trivial | 0.2-0.5 pequeno | "
          "0.5-0.8 medio | >0.8 grande")
        w("=" * 80)

    print(f"  [OK] {p}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CINEMATICA INVERSA BIMO 6-DOF - v4.0")
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

    # Estatistica
    analise_estatistica(res_pso, res_dfl, res_ag)

    print("\n" + "=" * 70)
    print("[CONCLUIDO] Todos os resultados salvos em:", BASE_DIR)
    print("=" * 70 + "\n")
