#!/usr/bin/env python3
"""
=============================================================================
  CONFIGURAÇÃO CENTRALIZADA - ROBÔ BIMO (6-DOF)
  
  Módulo repositório de constantes globais, caminhos de diretórios,
  limites de juntas físicas e geração de trajetórias do robô.
=============================================================================
"""

import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. MAPEAMENTO E ESTRUTURAÇÃO DE PASTAS DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("resultados")

DIRS = {
    "convergencia": BASE_DIR / "convergencia",
    "suavidade": BASE_DIR / "suavidade",
    "estatistica": BASE_DIR / "estatistica",
    "diversidade": BASE_DIR / "diversidade",
    "warm_cold": BASE_DIR / "warm_cold_start",
    "dados_brutos": BASE_DIR / "dados_brutos",
    "trajetorias": BASE_DIR / "trajetorias"  # <--- Nova pasta integrada com sucesso!
}

# Cria automaticamente os diretórios no seu sistema de arquivos caso não existam
for folder in DIRS.values():
    folder.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURAÇÕES GLOBAIS DO EXPERIMENTO (METAS-HEURÍSTICAS)
# ─────────────────────────────────────────────────────────────────────────────
N_RUNS       = 10      # Número de execuções independentes por algoritmo
N_PASSOS     = 30      # Quantidade de pontos discretos na trajetória

# Parâmetros - Particle Swarm Optimization (PSO e PSO-DFL)
PSO_POP      = 40      # Tamanho da população (partículas)
PSO_ITER     = 150     # Limite máximo de iterações por ponto

# Parâmetros - Algoritmo Genético (AG / GA)
AG_POP       = 40      # Tamanho da população (indivíduos)
AG_GENS      = 150     # Limite de gerações por ponto
TAXA_CROSS   = 0.7     # Probabilidade de crossover (recombinação)
TAXA_MUT     = 0.15    # Probabilidade de mutação por gene

# Pesos da Função de Custo Multiobjetivo (Fitness)
PESO_POS     = 1.0     # Peso associado ao erro de translação cartesiana
PESO_JTA     = 0.05    # Peso associado à penalidade por deslocamento brusco

SEED_BASE    = 42      # Semente randômica primordial para reprodutibilidade


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELAGEM CINEMÁTICA E RESTRIÇÕES MECÂNICAS (DH PARAMS)
# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros de Denavit-Hartenberg Padrão do BIMO [theta, d, a, alpha]
DH_params = [
    [0, 184, 0,   np.pi/2],   # Junta 1
    [0, 0,   222, 0],          # Junta 2
    [0, 0,   205, 0],          # Junta 3
    [0, 0,   0,   -np.pi/2],   # Junta 4
    [0, 0,   0,   np.pi/2],    # Junta 5
    [0, 75,  0,   0]           # Junta 6
]

# Limites operacionais das juntas angulares em Radianos (Curso mecânico do BIMO)
LIMITES_INF = np.array([
    -np.pi, 
    -np.pi * 110 / 180, 
    -np.pi * 80 / 180,
    -np.pi, 
    -np.pi * 50 / 180,  
    -np.pi
])

LIMITES_SUP = np.array([
    np.pi,  
    np.pi * 155 / 180,  
    np.pi * 210 / 180,
    np.pi,  
    np.pi * 230 / 180,   
    np.pi
])


# ─────────────────────────────────────────────────────────────────────────────
# 4. GERAÇÃO ANALÍTICA DA TRAJETÓRIA DE REFERÊNCIA (ALVO)
# ─────────────────────────────────────────────────────────────────────────────
# Criação do perfil linear tridimensional a ser percorrido pelo efetuador
x_traj = np.linspace(150, 300, N_PASSOS)
y_traj = np.linspace(100, -100, N_PASSOS)
z_traj = np.ones(N_PASSOS) * 200.0

# Matriz compilada das coordenadas alvo [N_PASSOS x 3]
TRAJETORIA = np.column_stack([x_traj, y_traj, z_traj])