import numpy as np
import time
from sko.PSO import PSO

# Importando as variáveis globais de configuração do experimento
from bimo.config import (
    TRAJETORIA, PSO_POP, PSO_ITER, 
    LIMITES_INF, LIMITES_SUP, N_PASSOS
)

# Importando as funções auxiliares (Ajuste o caminho dependendo de onde você as colocou)
from bimo.kinematics import criar_fitness
from bimo.utils.metrics import converter_para_float

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