import numpy as np
import time

# Importando as variáveis globais de configuração do experimento
from bimo.config import (
    TRAJETORIA, AG_POP, AG_GENS, 
    TAXA_CROSS, TAXA_MUT,
    LIMITES_INF, LIMITES_SUP, N_PASSOS
)

# Importando a função auxiliar para criar o juiz (fitness)
from bimo.kinematics import criar_fitness

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