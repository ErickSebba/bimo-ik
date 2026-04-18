"""
Métricas de Suavidade e Continuidade
Variação angular, jerk articular, índice de suavidade
"""

import numpy as np
from typing import Dict, List, Tuple


class MetricasSuavidade:
    """
    Calcula métricas de suavidade da trajetória articular.
    
    Crítico para movimento real do robô. AG geralmente produz trajetórias
    mais suaves devido ao elitismo.
    """
    
    @staticmethod
    def soma_variacoes_angulares(q_trajetoria: np.ndarray) -> float:
        """
        Calcula Σ|Δθ| - soma das variações angulares entre pontos consecutivos.
        
        Args:
            q_trajetoria: (n_pontos, n_juntas) - Trajetória de juntas em radianos
            
        Returns:
            float - Soma total de variações angulares
        """
        q = np.array(q_trajetoria)
        variacoes = np.diff(q, axis=0)  # Diferença entre pontos consecutivos
        soma_abs = np.sum(np.abs(variacoes))
        return float(soma_abs)
    
    @staticmethod
    def variacoes_por_junta(q_trajetoria: np.ndarray) -> np.ndarray:
        """
        Variação angular de cada junta individualmente.
        
        Returns:
            (n_juntas,) - Σ|Δθ| para cada junta
        """
        q = np.array(q_trajetoria)
        variacoes = np.diff(q, axis=0)
        soma_por_junta = np.sum(np.abs(variacoes), axis=0)
        return soma_por_junta
    
    @staticmethod
    def jerk_articular(q_trajetoria: np.ndarray) -> float:
        """
        Calcula jerk articular - variação da aceleração (terceira derivada).
        
        Jerk = Σ|d³θ/dt³|
        
        Valores altos indicam movimento áspero, baixos indicam suave.
        
        Args:
            q_trajetoria: (n_pontos, n_juntas)
            
        Returns:
            float - Jerk articular médio (normalizado)
        """
        q = np.array(q_trajetoria)
        
        if len(q) < 4:  # Precisa de pelo menos 4 pontos para 3ª derivada
            return 0.0
        
        # Primeira derivada (velocidade articular)
        vel = np.diff(q, axis=0)  # (n-1, n_juntas)
        
        # Segunda derivada (aceleração articular)
        acc = np.diff(vel, axis=0)  # (n-2, n_juntas)
        
        # Terceira derivada (jerk)
        jerk = np.diff(acc, axis=0)  # (n-3, n_juntas)
        
        # Jerk total normalizado
        jerk_total = np.sum(np.abs(jerk))
        n_pontos = len(q)
        jerk_normalizado = jerk_total / n_pontos if n_pontos > 0 else 0.0
        
        return float(jerk_normalizado)
    
    @staticmethod
    def indice_suavidade(q_trajetoria: np.ndarray) -> float:
        """
        Índice de Suavidade baseado em norma de derivadas (IROS standard).
        
        Ideia: trajetórias suaves têm derivadas pequenas
        
        Fórmula: S = 1 / (1 + sqrt(Σ(d²θ/dt²)²))
        
        Valores [0,1]: 1 = perfeitamente suave, 0 = muito áspero
        
        Args:
            q_trajetoria: (n_pontos, n_juntas)
            
        Returns:
            float - Score de suavidade [0, 1]
        """
        q = np.array(q_trajetoria)
        
        if len(q) < 3:
            return 1.0
        
        # Velocidades
        vel = np.diff(q, axis=0)
        
        # Acelerações
        acc = np.diff(vel, axis=0)
        
        # Norma Euclidiana de cada aceleração
        norma_acc = np.linalg.norm(acc, axis=1)
        
        # Soma dos quadrados
        soma_quad = np.sum(norma_acc ** 2)
        
        # Índice normalizado
        indice = 1.0 / (1.0 + np.sqrt(soma_quad))
        
        return float(np.clip(indice, 0, 1))
    
    @staticmethod
    def descontinuidade_trajetoria(q_trajetoria: np.ndarray) -> float:
        """
        Mede descontinuidades na trajetória (saltos não-naturais).
        
        Calcula picos de aceleração / aceleração média.
        
        Valores altos = trajetória com "saltos" ou transições abruptas
        Valores baixos = trajetória fluida
        
        Returns:
            float - Fator de descontinuidade (sem unidade)
        """
        q = np.array(q_trajetoria)
        
        if len(q) < 3:
            return 0.0
        
        vel = np.diff(q, axis=0)
        acc = np.diff(vel, axis=0)
        
        norma_acc = np.linalg.norm(acc, axis=1)
        
        if len(norma_acc) == 0 or np.mean(norma_acc) == 0:
            return 0.0
        
        pico = np.max(norma_acc)
        media = np.mean(norma_acc)
        
        descontinuidade = pico / media
        
        return float(descontinuidade)
    
    @staticmethod
    def energia_mecanica_trajetoria(q_trajetoria: np.ndarray, 
                                   massas_juntas: np.ndarray = None) -> float:
        """
        Aproximação da energia mecânica gasta na trajetória.
        
        E ≈ Σ m * v² (energia cinética aproximada)
        
        Trajetória eficiente tem energia baixa.
        
        Args:
            q_trajetoria: (n_pontos, n_juntas)
            massas_juntas: (n_juntas,) - Massas dos links
            
        Returns:
            float - Energia total normalizada
        """
        q = np.array(q_trajetoria)
        
        if massas_juntas is None:
            massas_juntas = np.ones(q.shape[1])
        
        # Velocidades articulares
        vel = np.diff(q, axis=0)
        
        # Energia cinética: E = Σ m_i * v_i²
        energia = 0.0
        for i in range(vel.shape[1]):
            energia += massas_juntas[i] * np.sum(vel[:, i] ** 2)
        
        # Normalizar
        energia_normalizada = energia / len(q)
        
        return float(energia_normalizada)
    
    @staticmethod
    def relatorio_suavidade(q_trajetoria: np.ndarray,
                           titulo_algoritmo: str = "") -> Dict:
        """
        Gera relatório completo de suavidade.
        
        Returns:
            Dict com todas as métricas calculadas
        """
        variacoes = MetricasSuavidade.variacoes_por_junta(q_trajetoria)
        
        return {
            'algoritmo': titulo_algoritmo,
            'soma_variacoes_totais': float(MetricasSuavidade.soma_variacoes_angulares(q_trajetoria)),
            'soma_variacoes_por_junta': variacoes.tolist(),
            'jerk_articular': float(MetricasSuavidade.jerk_articular(q_trajetoria)),
            'indice_suavidade': float(MetricasSuavidade.indice_suavidade(q_trajetoria)),
            'descontinuidade': float(MetricasSuavidade.descontinuidade_trajetoria(q_trajetoria)),
            'energia_mecanica': float(MetricasSuavidade.energia_mecanica_trajetoria(q_trajetoria))
        }
