#!/usr/bin/env python3
"""
=============================================================================
  MÓDULO DE MÉTRICAS E CINEMÁTICA DIRETA - ROBÔ BIMO (6-DOF)
  
  Responsável por calcular o posicionamento do efetuador e extrair as
  métricas de performance (RMSE, suavidade, Jerk articular e convergência).
=============================================================================
"""

import numpy as np
from bimo.config import TRAJETORIA, N_PASSOS, DH_params

def dh_matrix(theta, d, a, alpha):
    """Matriz de transformação DH homogênea padrão."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0,  sa,     ca,    d],
        [0,  0,      0,     1]
    ])

def cinematica_direta(q):
    """Calcula a posição XYZ (mm) do efetuador final com base nos ângulos das juntas."""
    T = np.eye(4)
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T = T @ dh_matrix(q[i] + theta, d, a, alpha)
    return T[:3, 3]

def calcular_metricas(resultado_algo):
    """
    Processa o histórico de posições e erros brutos extraídos durante as runs
    e gera o dicionário consolidado de métricas estatísticas.
    """
    hist_q = resultado_algo["historico_q"]
    hist_err = resultado_algo["historico_erro"]
    
    # Reconstrução das posições cartesianas executadas pelo robô
    historico_pos = np.array([cinematica_direta(q) for q in hist_q])
    
    # Erros pontuais de posição em milímetros em relação ao alvo analítico
    erros_pos = np.array([
        np.linalg.norm(historico_pos[i] - TRAJETORIA[i])
        for i in range(N_PASSOS)
    ])
    
    # Métricas Básicas de Erro Cartesiano
    rmse = np.sqrt(np.mean(erros_pos**2))
    erro_medio = np.mean(erros_pos)
    erro_max = np.max(erros_pos)
    
    # Métricas de Suavidade Cinemática: Σ|Δθ|
    suavidade = np.sum(np.abs(np.diff(hist_q, axis=0)))
    
    # Jerk Articular: Σ|Δ²θ| (Mapeia a derivada segunda da velocidade/aceleração)
    vel_articular = np.diff(hist_q, axis=0)
    jerk_articular = np.sum(np.abs(np.diff(vel_articular, axis=0)))
    
    # Índice de suavidade normalizado por passo temporal
    indice_suavidade = suavidade / (N_PASSOS - 1) if N_PASSOS > 1 else 0
    
    # Taxa média de Convergência Heurística
    taxa_convergencia = 0.0
    if len(hist_err) > 1:
        melhorias = np.maximum(0, hist_err[:-1] - hist_err[1:])
        taxa_convergencia = np.mean(melhorias)
    
    # Métricas de Fitness do Landscape
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