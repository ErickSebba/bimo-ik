"""
=============================================================================
Artigo: Análise Comparativa Simulada entre Algoritmo Genético e PSO 
        na Cinemática Inversa de um Braço Robótico Educacional (BIMO 6-DOF)
-----------------------------------------------------------------------------
Metodologia: Mapeamento Geométrico SE(3) Completo e Otimização Multiobjetivo
Dependências: numpy, scipy, matplotlib, roboticstoolbox-python, scikit-opt
=============================================================================
"""

import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
from spatialmath import SE3, SO3

# =====================================================================
# TRATAMENTO DE COMPATIBILIDADE SCIPY/ROBOTICSTOOLBOX
# =====================================================================
if not hasattr(scipy, "randn"):
    scipy.randn = np.random.randn

import roboticstoolbox as rtb
from sko.PSO import PSO
from sko.GA import GA

# =====================================================================
# 1. MODELAGEM DO ROBÔ BIMO COM LIMITES ARTICULARES REALISTAS
# =====================================================================
# Robôs reais, diferentemente de modelos teóricos abstratos, possuem
# batentes mecânicos rígidos. Fornecer uma liberdade infinita (-pi a pi)
# a todas as juntas causa configurações inatingíveis na física real,
# além de intensificar a complexidade topológica em algoritmos bio-inspirados.
limites_articulacao = [
    (-np.pi, np.pi),
    (-np.pi / 2, np.pi / 2),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
]

limite_inferior = [lim[0] for lim in limites_articulacao]
limite_superior = [lim[1] for lim in limites_articulacao]

# Instanciação paramétrica baseada na convenção Denavit-Hartenberg
bimo = rtb.DHRobot([
    rtb.RevoluteDH(d=184, a=0,   alpha=np.pi/2,  qlim=limites_articulacao[0]),
    rtb.RevoluteDH(d=0,   a=222, alpha=0,        qlim=limites_articulacao[1]),
    rtb.RevoluteDH(d=0,   a=205, alpha=0,        qlim=limites_articulacao[2]),
    rtb.RevoluteDH(d=0,   a=0,   alpha=-np.pi/2, qlim=limites_articulacao[3]),
    rtb.RevoluteDH(d=0,   a=0,   alpha=np.pi/2,  qlim=limites_articulacao[4]),
    rtb.RevoluteDH(d=75,  a=0,   alpha=0,        qlim=limites_articulacao[5])
], name="BIMO_6DOF")

def obter_matriz_se3(q):
    """Retorna a matriz homogênea 4x4 completa de translação e rotação."""
    return bimo.fkine(q).A

# =====================================================================
# 2. DEFINIÇÃO DA FUNÇÃO FITNESS MULTIOBJETIVO CIENTÍFICA
# =====================================================================
def construir_avaliador_fitness(matriz_alvo, q_anterior, 
                                peso_trans=1.0, peso_rot=0.5, peso_junta=0.005):
    """
    Constrói a função de aptidão em tempo de execução incorporando:
    1. Distância Euclidiana (mm) para controle translacional.
    2. Distância de Traço Matricial (radianos) para controle orientacional.
    3. Fator de Regularização para minimizar a descontinuidade articular.
    
    NOTA: O peso_junta é significativamente reduzido para promover
          o dimensionamento escalar correto frente às grandezas em mm.
    """
    posicao_alvo = matriz_alvo[:3, 3]
    rotacao_alvo = matriz_alvo[:3, :3]
    
    def fitness_computada(q):
        matriz_atual = obter_matriz_se3(q)
        posicao_atual = matriz_atual[:3, 3]
        rotacao_atual = matriz_atual[:3, :3]
        
        # Erro Translacional Euclidiano (L2 Norm)
        erro_cartesiano = np.linalg.norm(posicao_atual - posicao_alvo)
        
        # Erro de Orientação Computada via Traço (Trace)
        # Extrai o ângulo real de diferença entre duas matrizes SO(3).
        # A limitação do clip previne NaNs ocasionados por falhas de precisão do float64.
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

# =====================================================================
# 3. CONSTRUÇÃO DO PERFIL INTERPOLADO DA TRAJETÓRIA EM SE(3)
# =====================================================================
passos_discretos = 30
coord_x = np.linspace(150, 300, passos_discretos) 
coord_y = np.linspace(100, -100, passos_discretos)
coord_z = np.linspace(200, 200, passos_discretos)

# Fixação da premissa orientativa: A ferramenta robótica (ex: laser, pinça) 
# deve rigorosamente apontar 180 graus invertida no eixo Y para o piso.
rotacao_estrita = SO3.Ry(np.pi).A 

trajetoria_completa = []
for idx in range(passos_discretos):
    matriz_transformacao = np.eye(4)
    matriz_transformacao[:3, :3] = rotacao_estrita
    matriz_transformacao[0, 3] = coord_x[idx]
    matriz_transformacao[1, 3] = coord_y[idx]
    matriz_transformacao[2, 3] = coord_z[idx]
    trajetoria_completa.append(matriz_transformacao)

# =====================================================================
# 4. PLATAFORMA DE CONTROLE E AVALIAÇÃO ESTATÍSTICA (PSO & GA)
# =====================================================================
def executar_rotina_bioinspirada(modelo_metaheuristica="PSO", debug_logs=False):
    """
    Executa a resolução cinemática ao longo da trajetória completa,
    computando os custos transientes e injetando estratégias genéticas/inerciais.
    """
    q_registro_passado = np.zeros(6) 
    matriz_vetores_q = []
    lista_erros_rmse = []
    lista_custos_fitness = []
    
    # Parâmetros de Avaliação Experimental
    agentes_populacionais = 50
    iteracoes_geracionais = 250
    fracao_doping = 0.10 # Limite de 10% para inserção de Warm-Start
    qtde_doping = int(agentes_populacionais * fracao_doping)
    
    clock_inicio = time.time()
    
    for pt_idx, frame_alvo in enumerate(trajetoria_completa):
        avaliador = construir_avaliador_fitness(frame_alvo, q_registro_passado)
        
        if modelo_metaheuristica == "PSO":
            # Parâmetros Sociais e Cognitivos balanceados para exploração fina
            otimizador = PSO(func=avaliador, n_dim=6, pop=agentes_populacionais, 
                             max_iter=iteracoes_geracionais, 
                             lb=limite_inferior, ub=limite_superior, 
                             w=0.6, c1=1.5, c2=1.5)
            
            # Controle de Doping: Preenche-se apenas as primeiras partículas para 
            # não comprometer as reservas de entropia necessárias ao salto topológico.
            otimizador.X[0] = q_registro_passado
            for w in range(1, qtde_doping):
                otimizador.X[w] = q_registro_passado + np.random.uniform(-0.05, 0.05, 6)
                
            melhor_vetor_q, melhor_custo_j = otimizador.run()
            melhor_vetor_q = np.asarray(melhor_vetor_q, dtype=float).reshape(-1)
            melhor_custo_j = float(np.asarray(melhor_custo_j).squeeze())
            
        elif modelo_metaheuristica == "GA":
            # Algoritmo Genético com taxas Mendelianas equilibradas
            otimizador = GA(func=avaliador, n_dim=6, size_pop=agentes_populacionais, 
                            max_iter=iteracoes_geracionais, 
                            prob_mut=0.1, lb=limite_inferior, ub=limite_superior)
            
            melhor_vetor_q, melhor_custo_j = otimizador.run()
            melhor_vetor_q = np.asarray(melhor_vetor_q, dtype=float).reshape(-1)
            melhor_custo_j = float(np.asarray(melhor_custo_j).squeeze())
            
        # Isolar e registrar estritamente o Erro Translacional Absoluto (em milímetros)
        # O Fitness retornado pelo algoritmo contém penalizações artificiais misturadas.
        pos_extraida_calculada = obter_matriz_se3(melhor_vetor_q)[:3, 3]
        pos_pura_alvo = frame_alvo[:3, 3]
        desvio_euclidiano_mm = np.linalg.norm(pos_extraida_calculada - pos_pura_alvo)
        
        matriz_vetores_q.append(melhor_vetor_q)
        lista_erros_rmse.append(desvio_euclidiano_mm)
        lista_custos_fitness.append(melhor_custo_j)
        
        q_registro_passado = melhor_vetor_q
        
        if debug_logs:
            print(f"[{modelo_metaheuristica}] Ciclo {pt_idx+1:02d}/{passos_discretos} | "
                  f"RMSE Posição: {desvio_euclidiano_mm:.3f} mm | Função Custo: {melhor_custo_j:.3f}")

    processamento_decorrido = time.time() - clock_inicio
    
    return (np.array(matriz_vetores_q), np.array(lista_erros_rmse), 
            np.array(lista_custos_fitness), processamento_decorrido)

# =====================================================================
# 5. GERENCIADOR DE VISUALIZAÇÕES E RELATÓRIO CIENTÍFICO FINAL
# =====================================================================
if __name__ == "__main__":
    print("Processando Simulador Comparativo (PSO vs AG) - Trajetória Cinemática SE(3)...\n")
    
    # Avaliação Sequencial Contígua
    q_hist_pso, rmse_pso, custo_pso, crono_pso = executar_rotina_bioinspirada("PSO", debug_logs=True)
    print("-" * 65)
    q_hist_ga, rmse_ga, custo_ga, crono_ga = executar_rotina_bioinspirada("GA", debug_logs=True)
    
    # 5.1 Apuração Analítica no Terminal
    print("\n" + "="*65)
    print("SÍNTESE DOS RESULTADOS EMPÍRICOS (AVALIAÇÃO ESTATÍSTICA)")
    print("="*65)
    print(f" Convergência Global : {crono_pso:.2f} s")
    print(f" Desvio Médio Rastreio : {np.mean(rmse_pso):.4f} mm")
    print(f" Pico Crítico Máximo : {np.max(rmse_pso):.4f} mm")
    print("-" * 65)
    print(f"[AG]  Convergência Global : {crono_ga:.2f} s")
    print(f"[AG]  Desvio Médio Rastreio : {np.mean(rmse_ga):.4f} mm")
    print(f"[AG]  Pico Crítico Máximo : {np.max(rmse_ga):.4f} mm")
    print("="*65 + "\n")

    # 5.2 Plotagem Gráfica (Formatada para padrões de Engenharia e Artigos - CBA/IEEE)
    
    # Gráfico A: Discrepância Euclidiana Translacional (Aferição de Acurácia)
    plt.figure(figsize=(10, 5))
    plt.plot(rmse_pso, label='PSO (Otimização por Enxame)', color='royalblue', 
             linewidth=2.5, marker='o', markersize=5)
    plt.plot(rmse_ga, label='AG (Evolução Genética)', color='firebrick', 
             linewidth=2.5, marker='s', linestyle='--', markersize=5)
    
    plt.title('Comparativo de Acurácia de Rastreamento (Erro Translacional - RMSE)', 
              fontsize=12, fontweight='bold')
    plt.xlabel('Discretização da Trajetória (Waypoints)', fontsize=11)
    plt.ylabel('Erro Espacial Euclidiano Absoluto (mm)', fontsize=11)
    
    # A representação logarítmica é o padrão de facto na análise heurística 
    # devido a dispersão de magnitudes entre as fases iniciais e o ajuste fino.
    plt.yscale('log') 
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig('figura_01_precisao_comparativa.png', dpi=300)
    
    # Gráfico B: Dinâmica do Espaço de Configuração (Aferição de Fluidez / Jerk)
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    
    for i_junta in range(6):
        eixos[0].plot(np.degrees(q_hist_pso[:, i_junta]), 
                      label=f'Junta {i_junta+1}', linewidth=1.5)
        eixos[1].plot(np.degrees(q_hist_ga[:, i_junta]), 
                      label=f'Junta {i_junta+1}', linewidth=1.5)
        
    eixos[0].set_title('Evolução Cinemática Angular (Rad → Graus) - Perfil PSO', fontsize=11)
    eixos[0].set_xlabel('Pontos Espaciais')
    eixos[0].set_ylabel('Rotação Articular (Graus)')
    eixos[0].grid(True, linestyle=':', alpha=0.8)
    
    eixos[1].set_title('Evolução Cinemática Angular (Rad → Graus) - Perfil AG', fontsize=11)
    eixos[1].set_xlabel('Pontos Espaciais')
    eixos[1].grid(True, linestyle=':', alpha=0.8)
    eixos[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.suptitle('Estabilidade Topológica e Suavização de Movimento no Espaço Articular', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figura_02_fluidez_articular.png', dpi=300)
    
    plt.show()