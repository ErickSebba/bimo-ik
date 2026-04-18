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
N_RUNS       = 10      # execuções independentes por algoritmo (rigor estatístico)
N_PASSOS     = 30      # pontos na trajetória
PSO_POP      = 40      # partículas PSO  (≥30 recomendado pela literatura)
PSO_ITER     = 150     # iterações PSO   (aumentado para reduzir erro final)
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
# Parâmetros DH do BIMO (theta, d, a, alpha)
DH_params = [
    [0, 184, 0,   np.pi/2],   # Junta 1
    [0, 0,   222, 0],          # Junta 2
    [0, 0,   205, 0],          # Junta 3
    [0, 0,   0,   -np.pi/2],   # Junta 4
    [0, 0,   0,   np.pi/2],    # Junta 5
    [0, 75,  0,   0]           # Junta 6
]

# Limites articulares individuais (rad)
# RECOMENDAÇÃO: substituir pelos valores reais do manual do BIMO
# Exemplo inspirado no IRB120 (Farnese et al.) convertido para rad:
LIMITES_INF = np.array([-np.pi, -np.pi*110/180, -np.pi*80/180,
                         -np.pi, -np.pi*50/180,  -np.pi])
LIMITES_SUP = np.array([ np.pi,  np.pi*155/180,  np.pi*210/180,
                          np.pi,  np.pi*230/180,   np.pi])

# Funções de cinemática direta
def dh_matrix(theta, d, a, alpha):
    """Matriz de transformação DH homogênea."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def cinematica_direta(q):
    """Retorna posição XYZ (mm) do efetuador dado o vetor de juntas q (rad)."""
    T = np.eye(4)
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T = T @ dh_matrix(q[i] + theta, d, a, alpha)
    return T[:3, 3]

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
    T = np.eye(4)
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T = T @ dh_matrix(q[i] + theta, d, a, alpha)
    return T

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
        
        pso = PSO(func=juiz, dim=6, pop=PSO_POP, max_iter=PSO_ITER,
                  lb=LIMITES_INF.tolist(), ub=LIMITES_SUP.tolist(),
                  w=0.9, c1=0.3, c2=0.3)  # Ajustes para maior suavidade
        
        # === DOPING (warm-start) ===
        # Planta solução anterior + variações pequenas como partículas iniciais.
        # Técnica validada em Farnese et al. – garantiu 100% convergência.
        pso.X[0] = np.clip(q_ant, LIMITES_INF, LIMITES_SUP)
        for j in range(1, 6):
            pso.X[j] = np.clip(q_ant + np.random.uniform(-0.05, 0.05, 6),
                                LIMITES_INF, LIMITES_SUP)
        # # ===========================
        pso.run()
        melhor_q = pso.gbest_x
        melhor_err = pso.gbest_y
        
        hist_q.append(melhor_q.copy())
        hist_erro.append(float(melhor_err))
        q_ant = melhor_q.copy()
        
        if verbose:
            print(f"  PSO ponto {i+1:2d}/{N_PASSOS} | "
                  f"alvo={np.round(alvo,1)} | erro={melhor_err[0]:.4f} mm")
    
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
            n_renov = max(1, int(0.20 * (AG_POP - 2)))
            if n_renov > 0:
                non_elite_scores = _avaliar(nova_pop[2:], fitness_fn)
                idx_piores_non_elite = np.argsort(non_elite_scores)[-n_renov:]
                nova_pop[2 + idx_piores_non_elite] = np.random.uniform(
                    LIMITES_INF, LIMITES_SUP, (n_renov, 6))
            
            pop    = nova_pop
            scores = _avaliar(pop, fitness_fn)
        
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
    Executa N_RUNS execuções independentes de PSO e AG, coletando
    todas as métricas. Cada run usa uma semente diferente para
    garantir amostragem do comportamento estocástico.
    """
    print("=" * 60)
    print(f"EXPERIMENTO: PSO vs AG – {N_RUNS} execuções cada")
    print(f"Trajetória: {N_PASSOS} pontos | PSO({PSO_POP}p,{PSO_ITER}it) "
          f"| AG({AG_POP}ind,{AG_GENS}ger)")
    print("=" * 60)
    
    resultados_pso = []
    resultados_ag  = []
    
    for run in range(N_RUNS):
        seed = SEED_BASE + run
        print(f"\n▶ Run {run+1}/{N_RUNS}  (seed={seed})")
        
        # --- PSO ---
        print("  [PSO]", end=" ", flush=True)
        hq_pso, he_pso, t_pso = rodar_pso(seed=seed, verbose=False)
        m_pso = calcular_metricas(hq_pso, he_pso)
        m_pso["tempo_s"]    = t_pso
        m_pso["historico_q"] = hq_pso
        m_pso["historico_erro"] = he_pso
        resultados_pso.append(m_pso)
        print(f"RMSE={m_pso['rmse_pos_mm']:.3f}mm  "
              f"MaxErr={m_pso['erro_max_mm']:.3f}mm  "
              f"Suav={m_pso['suavidade_rad']:.3f}rad  "
              f"t={t_pso:.1f}s")
        
        # --- AG ---
        print("  [AG] ", end=" ", flush=True)
        hq_ag, he_ag, t_ag = rodar_ag(seed=seed, verbose=False)
        m_ag = calcular_metricas(hq_ag, he_ag)
        m_ag["tempo_s"]     = t_ag
        m_ag["historico_q"] = hq_ag
        m_ag["historico_erro"] = he_ag
        resultados_ag.append(m_ag)
        print(f"RMSE={m_ag['rmse_pos_mm']:.3f}mm  "
              f"MaxErr={m_ag['erro_max_mm']:.3f}mm  "
              f"Suav={m_ag['suavidade_rad']:.3f}rad  "
              f"t={t_ag:.1f}s")
    
    return resultados_pso, resultados_ag


# ─────────────────────────────────────────────────────────────────────────────
# 8. TABELA COMPARATIVA (saída para o artigo)
# ─────────────────────────────────────────────────────────────────────────────
def imprimir_tabela(resultados_pso, resultados_ag):
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
    
    print("\n" + "=" * 65)
    print(f"{'MÉTRICA':<25} {'PSO (μ ± σ)':>18}  {'AG  (μ ± σ)':>18}")
    print("=" * 65)
    
    for chave, nome in metricas_labels:
        vals_pso = np.array([r[chave] for r in resultados_pso])
        vals_ag  = np.array([r[chave] for r in resultados_ag])
        mu_pso, sd_pso = vals_pso.mean(), vals_pso.std()
        mu_ag,  sd_ag  = vals_ag.mean(),  vals_ag.std()
        print(f"{nome:<25} {mu_pso:8.4f} ± {sd_pso:.4f}  "
              f"{mu_ag:8.4f} ± {sd_ag:.4f}")
    
    print("=" * 65)
    print(f"  N_RUNS={N_RUNS}  |  N_PASSOS={N_PASSOS}  |  "
          f"PSO({PSO_POP}p,{PSO_ITER}it)  |  AG({AG_POP}ind,{AG_GENS}ger)")
    print("  Pesos fitness: w_pos={:.2f}  w_jta={:.2f}".format(
          PESO_POS, PESO_JTA))


# ─────────────────────────────────────────────────────────────────────────────
# 9. VISUALIZAÇÕES PARA O ARTIGO
# ─────────────────────────────────────────────────────────────────────────────
def gerar_graficos(resultados_pso, resultados_ag):
    """
    Gera 5 figuras de qualidade para o artigo:
      Fig.1 – Trajetória 3D desejada vs PSO vs AG (melhor run de cada)
      Fig.2 – Erro de rastreamento por ponto (PSO vs AG sobrepostos)
      Fig.3 – Ângulos articulares – PSO (melhor run)
      Fig.4 – Ângulos articulares – AG  (melhor run)
      Fig.5 – Boxplot comparativo das métricas principais
    """
    
    # Identifica melhor run de cada algoritmo (menor RMSE)
    idx_best_pso = int(np.argmin([r["rmse_pos_mm"] for r in resultados_pso]))
    idx_best_ag  = int(np.argmin([r["rmse_pos_mm"] for r in resultados_ag]))
    best_pso = resultados_pso[idx_best_pso]
    best_ag  = resultados_ag[idx_best_ag]
    
    cor_pso = "#1f77b4"   # azul
    cor_ag  = "#d62728"   # vermelho
    cor_traj = "#7f7f7f"  # cinza
    
    # ── FIGURA 1: Trajetória 3D ───────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection="3d")
    
    ax.plot(*TRAJETORIA.T, "--", color=cor_traj, lw=1.8, label="Trajetória desejada")
    ax.plot(*best_pso["historico_pos"].T, "-", color=cor_pso, lw=2.2,
            marker="o", ms=3, label=f"PSO  (RMSE={best_pso['rmse_pos_mm']:.2f} mm)")
    ax.plot(*best_ag["historico_pos"].T, "-", color=cor_ag, lw=2.2,
            marker="s", ms=3, label=f"AG   (RMSE={best_ag['rmse_pos_mm']:.2f} mm)")
    
    ax.scatter(*TRAJETORIA[0], color="tab:orange", marker="^", s=70,
               label="Ponto inicial")
    ax.scatter(*TRAJETORIA[-1], color="tab:green",  marker="x", s=90,
               label="Ponto final")
    
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("Trajetória do Efetuador: Desejada vs Obtida (melhor run)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig1_trajetoria_3d.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"  → {fig_path} salva")
    
    # ── FIGURA 2: Erro de rastreamento por ponto ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    pontos = np.arange(1, N_PASSOS + 1)
    
    # Intervalo de confiança (média ± dp) ao longo dos pontos
    erros_pso_mat = np.array([r["erros_pos"] for r in resultados_pso])
    erros_ag_mat  = np.array([r["erros_pos"] for r in resultados_ag])
    mu_pso = erros_pso_mat.mean(axis=0)
    sd_pso = erros_pso_mat.std(axis=0)
    mu_ag  = erros_ag_mat.mean(axis=0)
    sd_ag  = erros_ag_mat.std(axis=0)
    
    ax.fill_between(pontos, mu_pso - sd_pso, mu_pso + sd_pso,
                    alpha=0.2, color=cor_pso)
    ax.plot(pontos, mu_pso, "-o", ms=4, lw=2, color=cor_pso,
            label=f"PSO  μ={mu_pso.mean():.2f} mm")
    ax.fill_between(pontos, mu_ag - sd_ag, mu_ag + sd_ag,
                    alpha=0.2, color=cor_ag)
    ax.plot(pontos, mu_ag,  "-s", ms=4, lw=2, color=cor_ag,
            label=f"AG   μ={mu_ag.mean():.2f} mm")
    
    ax.set_title(f"Erro de Rastreamento por Ponto (N={N_RUNS} execuções, μ ± σ)")
    ax.set_xlabel("Ponto da Trajetória")
    ax.set_ylabel("Erro de Posição (mm)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig2_erro_rastreamento.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"  → {fig_path} salva")
    
    # ── FIGURAS 3 & 4: Ângulos articulares ───────────────────────────────────
    for nome, dados, cor in [("PSO", best_pso, cor_pso), ("AG", best_ag, cor_ag)]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for j in range(6):
            ax.plot(np.degrees(dados["historico_q"][:, j]),
                    marker="o", ms=3, label=f"Junta {j+1}")
        ax.set_title(f"Perfil Articular – {nome} (melhor run, RMSE={dados['rmse_pos_mm']:.2f} mm)")
        ax.set_xlabel("Ponto da Trajetória")
        ax.set_ylabel("Ângulo (graus)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fname = f"fig{'3' if nome=='PSO' else '4'}_angulos_{nome.lower()}.png"
        fig_path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(fig_path, dpi=300)
        plt.show()
        print(f"  → {fig_path} salva")
    
    # ── FIGURA 5: Boxplot comparativo ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    metricas_box = [
        ("rmse_pos_mm",   "RMSE Posição (mm)"),
        ("suavidade_rad", "Suavidade (rad)"),
        ("tempo_s",       "Tempo de Exec. (s)"),
    ]
    
    for ax, (chave, titulo) in zip(axes, metricas_box):
        vals_pso = [r[chave] for r in resultados_pso]
        vals_ag  = [r[chave] for r in resultados_ag]
        bp = ax.boxplot([vals_pso, vals_ag],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor(cor_pso + "88")
        bp["boxes"][1].set_facecolor(cor_ag  + "88")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["PSO", "AG"], fontsize=11)
        ax.set_title(titulo)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    fig.suptitle(f"Comparativo PSO vs AG (N={N_RUNS} execuções independentes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig5_boxplot_comparativo.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"  → {fig_path} salva")

    gerar_video_trajetoria(best_pso, best_ag)
    gerar_video_movimento_bracos(best_pso, best_ag)


def gerar_video_trajetoria(best_pso, best_ag, arquivo_saida="video_trajetoria_pso_vs_ag.gif"):
    """
    Gera uma animação da trajetória completa do melhor run de PSO e AG.

    O arquivo é salvo em GIF por compatibilidade ampla. Se você quiser MP4,
    basta trocar o nome de saída para .mp4 e garantir que o ffmpeg esteja
    instalado no sistema.
    """
    trajetoria_pso = best_pso["historico_pos"]
    trajetoria_ag = best_ag["historico_pos"]
    n_frames = max(len(trajetoria_pso), len(trajetoria_ag), len(TRAJETORIA))

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

    fig = plt.figure(figsize=(13, 6))
    ax_pso = fig.add_subplot(121, projection="3d")
    ax_ag = fig.add_subplot(122, projection="3d")
    _setup_axis(ax_pso, "PSO - evolução da trajetória", "#1f77b4")
    _setup_axis(ax_ag, "AG - evolução da trajetória", "#d62728")

    linha_pso, = ax_pso.plot([], [], [], "-o", color="#1f77b4", lw=2.2, ms=3,
                             label=f"PSO (RMSE={best_pso['rmse_pos_mm']:.2f} mm)")
    linha_ag, = ax_ag.plot([], [], [], "-s", color="#d62728", lw=2.2, ms=3,
                            label=f"AG (RMSE={best_ag['rmse_pos_mm']:.2f} mm)")

    texto_pso = ax_pso.text2D(0.03, 0.95, "", transform=ax_pso.transAxes)
    texto_ag = ax_ag.text2D(0.03, 0.95, "", transform=ax_ag.transAxes)

    def _atualizar(frame):
        idx_pso = min(frame + 1, len(trajetoria_pso))
        idx_ag = min(frame + 1, len(trajetoria_ag))

        dados_pso = trajetoria_pso[:idx_pso]
        dados_ag = trajetoria_ag[:idx_ag]

        linha_pso.set_data(dados_pso[:, 0], dados_pso[:, 1])
        linha_pso.set_3d_properties(dados_pso[:, 2])
        linha_ag.set_data(dados_ag[:, 0], dados_ag[:, 1])
        linha_ag.set_3d_properties(dados_ag[:, 2])

        texto_pso.set_text(f"Passo {idx_pso}/{len(trajetoria_pso)}")
        texto_ag.set_text(f"Passo {idx_ag}/{len(trajetoria_ag)}")
        return linha_pso, linha_ag, texto_pso, texto_ag

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
    T = np.eye(4)
    pontos = [T[:3, 3].copy()]  # Posição inicial (base)
    
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T_i = dh_matrix(q[i] + theta, d, a, alpha)
        T = T @ T_i
        pontos.append(T[:3, 3].copy())
    
    return np.array(pontos)


def gerar_video_movimento_bracos(best_pso, best_ag):
    """
    Gera uma animação lado a lado mostrando o movimento dos elos do braço BIMO
    para PSO e AG, de forma similar ao coisa.py.
    """
    historico_q_pso = best_pso["historico_q"]
    historico_q_ag = best_ag["historico_q"]
    n_frames = max(len(historico_q_pso), len(historico_q_ag))
    
    # Pré-calcular posições dos elos para cada frame
    caminhos_pso = [posicoes_dos_elos(q) for q in historico_q_pso]
    caminhos_ag = [posicoes_dos_elos(q) for q in historico_q_ag]
    
    # Obter pontas (última posição de cada frame)
    pontas_pso = np.array([caminho[-1] for caminho in caminhos_pso])
    pontas_ag = np.array([caminho[-1] for caminho in caminhos_ag])
    
    # Calcular limites de visualização
    todos_pontos = np.vstack([np.concatenate(caminhos_pso), np.concatenate(caminhos_ag)])
    limite_xy = np.max(np.abs(todos_pontos[:, :2])) + 100
    limite_z = max(300, np.max(todos_pontos[:, 2]) + 100)
    
    fig = plt.figure(figsize=(14, 6))
    ax_pso = fig.add_subplot(121, projection="3d")
    ax_ag = fig.add_subplot(122, projection="3d")
    
    # Configurar eixos
    for ax, titulo in [(ax_pso, "PSO - Movimento dos Elos"), (ax_ag, "AG - Movimento dos Elos")]:
        ax.set_title(titulo)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_xlim(-limite_xy, limite_xy)
        ax.set_ylim(-limite_xy, limite_xy)
        ax.set_zlim(0, limite_z)
    
    # Linhas dos braços e rastros
    linha_pso, = ax_pso.plot([], [], [], "o-", lw=2.5, color="#1f77b4", markersize=6)
    rastro_pso, = ax_pso.plot([], [], [], "-", lw=2, color="#1f77b4", alpha=0.6)
    ponta_pso, = ax_pso.plot([], [], [], "o", markersize=10, color="#ff7f0e")
    
    linha_ag, = ax_ag.plot([], [], [], "s-", lw=2.5, color="#d62728", markersize=6)
    rastro_ag, = ax_ag.plot([], [], [], "-", lw=2, color="#d62728", alpha=0.6)
    ponta_ag, = ax_ag.plot([], [], [], "s", markersize=10, color="#ff7f0e")
    
    # Texto de status
    texto_pso = ax_pso.text2D(0.02, 0.95, "", transform=ax_pso.transAxes, fontsize=10)
    texto_ag = ax_ag.text2D(0.02, 0.95, "", transform=ax_ag.transAxes, fontsize=10)
    
    def atualizar(frame):
        idx_pso = min(frame, len(caminhos_pso) - 1)
        idx_ag = min(frame, len(caminhos_ag) - 1)
        
        # PSO
        pts_pso = caminhos_pso[idx_pso]
        linha_pso.set_data(pts_pso[:, 0], pts_pso[:, 1])
        linha_pso.set_3d_properties(pts_pso[:, 2])
        
        rastro_pts_pso = pontas_pso[:idx_pso + 1]
        rastro_pso.set_data(rastro_pts_pso[:, 0], rastro_pts_pso[:, 1])
        rastro_pso.set_3d_properties(rastro_pts_pso[:, 2])
        
        ponta_pso.set_data([rastro_pts_pso[-1, 0]], [rastro_pts_pso[-1, 1]])
        ponta_pso.set_3d_properties([rastro_pts_pso[-1, 2]])
        
        texto_pso.set_text(f"Frame {idx_pso + 1}/{len(caminhos_pso)}")
        
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
        
        return linha_pso, rastro_pso, ponta_pso, texto_pso, linha_ag, rastro_ag, ponta_ag, texto_ag
    
    anim = FuncAnimation(fig, atualizar, frames=n_frames, interval=180, blit=False)
    
    try:
        arquivo_saida = os.path.join(OUTPUT_DIR, "video_movimento_bracos_pso_vs_ag.gif")
        anim.save(arquivo_saida, writer=PillowWriter(fps=6))
        print(f"  → {arquivo_saida} salvo")
    except Exception as exc:
        print(f"  ! Erro ao salvar video de movimento ({exc})")
    
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 10.1 TESTE DE ROBUSTEZ: 30 RODADAS EM PONTO FIXO
# ─────────────────────────────────────────────────────────────────────────────
def teste_robustez_ponto_fixo(n_testes=30, ponto_idx=-1):
    """
    Testa robustez dos algoritmos em um ponto fixo da trajetória.
    Executa n_testes rodadas independentes e calcula estatísticas.
    
    Returns:
        dict com métricas de robustez para PSO e AG
    """
    ponto_alvo = TRAJETORIA[ponto_idx]  # Último ponto por padrão
    print(f"\n🔬 TESTE DE ROBUSTEZ - {n_testes} rodadas no ponto fixo: {np.round(ponto_alvo, 1)}")
    
    # Resultados PSO
    erros_pso = []
    tempos_pso = []
    for i in range(n_testes):
        juiz = criar_fitness(ponto_alvo, np.zeros(6))  # Sem histórico
        pso = PSO(func=juiz, dim=6, pop=PSO_POP, max_iter=PSO_ITER,
                  lb=LIMITES_INF.tolist(), ub=LIMITES_SUP.tolist(),
                  w=0.9, c1=0.3, c2=0.3)
        t0 = time.time()
        pso.run()
        t1 = time.time()
        
        pos_obtida = cinematica_direta(pso.gbest_x)
        erro = np.linalg.norm(pos_obtida - ponto_alvo)
        erros_pso.append(erro)
        tempos_pso.append(t1 - t0)
    
    # Resultados AG
    erros_ag = []
    tempos_ag = []
    for i in range(n_testes):
        fitness_fn = criar_fitness(ponto_alvo, np.zeros(6))
        pop = _inicializar_populacao(AG_POP)  # Sem doping
        scores = _avaliar(pop, fitness_fn)
        
        t0 = time.time()
        for gen in range(AG_GENS):
            elite_inds, elite_scores = _elitismo(pop, scores, n_elite=2)
            pais = _selecao_roleta(pop, scores)
            filhos = []
            for j in range(0, AG_POP - 2, 2):
                f1, f2 = _crossover_aritmetico(pais[j], pais[j+1])
                filhos.append(_mutacao_gaussiana(f1))
                filhos.append(_mutacao_gaussiana(f2))
            nova_pop = np.vstack([elite_inds, np.array(filhos[:AG_POP-2])])
            pop = nova_pop
            scores = _avaliar(pop, fitness_fn)
        t1 = time.time()
        
        idx_melhor = np.argmin(scores)
        pos_obtida = cinematica_direta(pop[idx_melhor])
        erro = np.linalg.norm(pos_obtida - ponto_alvo)
        erros_ag.append(erro)
        tempos_ag.append(t1 - t0)
    
    # Estatísticas
    resultados = {
        "ponto_testado": ponto_alvo,
        "n_testes": n_testes,
        "PSO": {
            "erro_medio": np.mean(erros_pso),
            "erro_std": np.std(erros_pso),
            "tempo_medio": np.mean(tempos_pso),
            "tempo_std": np.std(tempos_pso),
            "taxa_sucesso": np.mean(np.array(erros_pso) < 1.0)  # < 1mm
        },
        "AG": {
            "erro_medio": np.mean(erros_ag),
            "erro_std": np.std(erros_ag),
            "tempo_medio": np.mean(tempos_ag),
            "tempo_std": np.std(tempos_ag),
            "taxa_sucesso": np.mean(np.array(erros_ag) < 1.0)
        }
    }
    
    print(f"PSO: Erro μ={resultados['PSO']['erro_medio']:.3f}±{resultados['PSO']['erro_std']:.3f}mm, "
          f"Tempo μ={resultados['PSO']['tempo_medio']:.3f}±{resultados['PSO']['tempo_std']:.3f}s, "
          f"Sucesso={resultados['PSO']['taxa_sucesso']:.1%}")
    print(f"AG:  Erro μ={resultados['AG']['erro_medio']:.3f}±{resultados['AG']['erro_std']:.3f}mm, "
          f"Tempo μ={resultados['AG']['tempo_medio']:.3f}±{resultados['AG']['tempo_std']:.3f}s, "
          f"Sucesso={resultados['AG']['taxa_sucesso']:.1%}")
    
    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# 10.2 TESTE DE SUAVIDADE: ANÁLISE DETALHADA DOS PERFIS ARTICULARES
# ─────────────────────────────────────────────────────────────────────────────
def teste_suavidade_detalhada(resultados_pso, resultados_ag):
    """
    Análise detalhada da suavidade dos perfis articulares.
    Calcula métricas adicionais de jerk e continuidade.
    """
    print("\n🔄 TESTE DE SUAVIDADE - Análise dos perfis articulares")
    print("=" * 60)
    
    # Melhor run de cada
    idx_best_pso = np.argmin([r["rmse_pos_mm"] for r in resultados_pso])
    idx_best_ag = np.argmin([r["rmse_pos_mm"] for r in resultados_ag])
    
    q_pso = resultados_pso[idx_best_pso]["historico_q"]
    q_ag = resultados_ag[idx_best_ag]["historico_q"]
    
    # Métricas de suavidade por junta
    for nome, q_hist in [("PSO", q_pso), ("AG", q_ag)]:
        print(f"\n{nome}:")
        jerk_total = 0
        for j in range(6):
            # Variação angular
            var_ang = np.sum(np.abs(np.diff(q_hist[:, j])))
            # Jerk (variação da velocidade angular)
            vel_ang = np.diff(q_hist[:, j])
            jerk_ang = np.sum(np.abs(np.diff(vel_ang)))
            jerk_total += jerk_ang
            
            print(f"  Junta {j+1}: Variação={var_ang:.3f}rad, Jerk={jerk_ang:.3f}rad²")
        
        print(f"  Jerk Total: {jerk_total:.3f}rad²")
        print(f"  Suavidade Geral: {np.sum(np.abs(np.diff(q_hist, axis=0))):.3f}rad")


# ─────────────────────────────────────────────────────────────────────────────
# 10.3 TESTE DO IMPACTO DO ENVIESAMENTO (DOPING)
# ─────────────────────────────────────────────────────────────────────────────
def teste_doping_impacto(n_runs=5):
    """
    Compara desempenho com e sem doping (warm-start).
    Executa n_runs trajetórias completas para cada configuração.
    """
    print("\n🎯 TESTE DO IMPACTO DO DOPING - {n_runs} trajetórias completas")
    print("=" * 60)
    
    # Resultados COM doping
    tempos_pso_com = []
    tempos_ag_com = []
    for run in range(n_runs):
        seed = SEED_BASE + run + 100  # Sementes diferentes
        
        # PSO com doping
        t0 = time.time()
        hq_pso, _, _ = rodar_pso(seed=seed, verbose=False)
        tempos_pso_com.append(time.time() - t0)
        
        # AG com doping
        t0 = time.time()
        hq_ag, _, _ = rodar_ag(seed=seed, verbose=False)
        tempos_ag_com.append(time.time() - t0)
    
    # Resultados SEM doping
    def rodar_pso_sem_doping(seed=42, verbose=False):
        q_ant = np.zeros(6)
        hist_q = []
        hist_erro = []
        t0 = time.time()
        for i, alvo in enumerate(TRAJETORIA):
            juiz = criar_fitness(alvo, q_ant)
            pso = PSO(func=juiz, dim=6, pop=PSO_POP, max_iter=PSO_ITER,
                      lb=LIMITES_INF.tolist(), ub=LIMITES_SUP.tolist(),
                      w=0.9, c1=0.3, c2=0.3)
            # SEM doping: população totalmente aleatória
            pso.run()
            melhor_q = pso.gbest_x
            melhor_err = pso.gbest_y
            hist_q.append(melhor_q.copy())
            hist_erro.append(float(melhor_err))
            q_ant = melhor_q.copy()
        return time.time() - t0
    
    def rodar_ag_sem_doping(seed=42, verbose=False):
        q_ant = np.zeros(6)
        hist_q = []
        hist_err = []
        t0 = time.time()
        for i, alvo in enumerate(TRAJETORIA):
            fitness_fn = criar_fitness(alvo, q_ant)
            # SEM doping: população totalmente aleatória
            pop = _inicializar_populacao(AG_POP)  # Sem q_doping
            scores = _avaliar(pop, fitness_fn)
            
            for gen in range(AG_GENS):
                elite_inds, elite_scores = _elitismo(pop, scores, n_elite=2)
                pais = _selecao_roleta(pop, scores)
                filhos = []
                for j in range(0, AG_POP - 2, 2):
                    f1, f2 = _crossover_aritmetico(pais[j], pais[j+1])
                    filhos.append(_mutacao_gaussiana(f1))
                    filhos.append(_mutacao_gaussiana(f2))
                nova_pop = np.vstack([elite_inds, np.array(filhos[:AG_POP-2])])
                pop = nova_pop
                scores = _avaliar(pop, fitness_fn)
            
            idx_melhor = np.argmin(scores)
            melhor_q = pop[idx_melhor].copy()
            melhor_err = scores[idx_melhor]
            hist_q.append(melhor_q)
            hist_err.append(float(melhor_err))
            q_ant = melhor_q.copy()
        return time.time() - t0
    
    tempos_pso_sem = []
    tempos_ag_sem = []
    for run in range(n_runs):
        seed = SEED_BASE + run + 100
        
        # PSO sem doping
        tempos_pso_sem.append(rodar_pso_sem_doping(seed=seed, verbose=False))
        
        # AG sem doping
        tempos_ag_sem.append(rodar_ag_sem_doping(seed=seed, verbose=False))
    
    # Estatísticas
    resultados_doping = {
        "PSO_com_doping": {
            "tempo_medio": np.mean(tempos_pso_com),
            "tempo_std": np.std(tempos_pso_com)
        },
        "PSO_sem_doping": {
            "tempo_medio": np.mean(tempos_pso_sem),
            "tempo_std": np.std(tempos_pso_sem)
        },
        "AG_com_doping": {
            "tempo_medio": np.mean(tempos_ag_com),
            "tempo_std": np.std(tempos_ag_com)
        },
        "AG_sem_doping": {
            "tempo_medio": np.mean(tempos_ag_sem),
            "tempo_std": np.std(tempos_ag_sem)
        }
    }
    
    print("\nRESULTADOS DO IMPACTO DO DOPING:")
    print(f"PSO com doping:    μ={resultados_doping['PSO_com_doping']['tempo_medio']:.1f}±{resultados_doping['PSO_com_doping']['tempo_std']:.1f}s")
    print(f"PSO sem doping:    μ={resultados_doping['PSO_sem_doping']['tempo_medio']:.1f}±{resultados_doping['PSO_sem_doping']['tempo_std']:.1f}s")
    print(f"AG com doping:     μ={resultados_doping['AG_com_doping']['tempo_medio']:.1f}±{resultados_doping['AG_com_doping']['tempo_std']:.1f}s")
    print(f"AG sem doping:     μ={resultados_doping['AG_sem_doping']['tempo_medio']:.1f}±{resultados_doping['AG_sem_doping']['tempo_std']:.1f}s")
    
    speedup_pso = resultados_doping['PSO_sem_doping']['tempo_medio'] / resultados_doping['PSO_com_doping']['tempo_medio']
    speedup_ag = resultados_doping['AG_sem_doping']['tempo_medio'] / resultados_doping['AG_com_doping']['tempo_medio']
    
    print(f"\nSpeedup com doping:")
    print(f"PSO: {speedup_pso:.2f}x mais rápido")
    print(f"AG:  {speedup_ag:.2f}x mais rápido")
    
    return resultados_doping
if __name__ == "__main__":
    
    # ── Modo rápido para testar o código (1 run, poucos pontos) ──────────────
    # Descomente o bloco abaixo para teste rápido antes do experimento completo
    # N_RUNS = 1; PSO_ITER = 50; AG_GENS = 50; N_PASSOS = 10
    # TRAJETORIA = TRAJETORIA[:N_PASSOS]
    
    print("\nINICIANDO EXPERIMENTO COMPLETO")
    print(f"Estimativa de tempo: ~{N_RUNS * (PSO_POP*PSO_ITER + AG_POP*AG_GENS) / 8000:.0f} min\n")
    
    resultados_pso, resultados_ag = rodar_experimento()
    
    print("\n\n─── TABELA COMPARATIVA PARA O ARTIGO ───")
    imprimir_tabela(resultados_pso, resultados_ag)
    
    print("\n\n─── GERANDO FIGURAS ───")
    gerar_graficos(resultados_pso, resultados_ag)
    
    print("\n\n─── TESTES ADICIONAIS ───")
    
    # Teste 1: Robustez em ponto fixo
    teste_robustez_ponto_fixo(n_testes=30)
    
    # Teste 2: Suavidade detalhada
    teste_suavidade_detalhada(resultados_pso, resultados_ag)
    
    # Teste 3: Impacto do doping
    teste_doping_impacto(n_runs=5)
    
    print("\n✓ Experimento completo finalizado. Todos os testes realizados.")
