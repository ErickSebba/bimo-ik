"""
Métricas de Preservação de Diversidade vs. Convergência Prematura
Análise do Desvio Padrão da População
"""

import numpy as np
from typing import Tuple, Dict, List


class MetricasDiversidade:
    """
    Calcula métricas de diversidade da população ao longo das iterações.
    
    Referência: Duymazlar & Gönenli (Cap. 17)
    - Desvio padrão zero indica convergência para mínimo local
    - PSO-DFL deve manter maior diversidade em landscape dinâmico
    """
    
    @staticmethod
    def desvio_padrao_populacao(populacoes: List[np.ndarray]) -> np.ndarray:
        """
        Calcula desvio padrão de cada variável ao longo das iterações.
        
        Args:
            populacoes: Lista de (n_particulas, n_dimensoes) em cada iteração
            
        Returns:
            (n_iteracoes, n_dimensoes) - Desvio padrão por dimensão
        """
        desvios = []
        for pop in populacoes:
            desvio = np.std(pop, axis=0)
            desvios.append(desvio)
        return np.array(desvios)
    
    @staticmethod
    def diversidade_media_iteracao(populacoes: List[np.ndarray]) -> np.ndarray:
        """
        Diversidade média por iteração (média dos 6 desvios padrões das juntas).
        
        Returns:
            (n_iteracoes,) - Diversidade escalar por iteração
        """
        desvios = MetricasDiversidade.desvio_padrao_populacao(populacoes)
        return np.mean(desvios, axis=1)
    
    @staticmethod
    def taxa_perda_diversidade(diversidades: np.ndarray) -> float:
        """
        Calcula taxa de queda da diversidade (inclinação da reta ajustada).
        
        Valores altos = convergência rápida (pode ser prematura)
        Valores baixos = mantém diversidade (exploração prolongada)
        
        Args:
            diversidades: (n_iteracoes,) - Diversidade média por iteração
            
        Returns:
            float - Taxa de queda (negativa = queda de diversidade)
        """
        x = np.arange(len(diversidades))
        coef = np.polyfit(x, diversidades, 1)[0]  # inclinação
        return coef
    
    @staticmethod
    def ponto_convergencia_prematura(diversidades: np.ndarray, 
                                     limiar_variacao: float = 0.001) -> int:
        """
        Detecta em qual iteração a diversidade "morreu" (ficou muito pequena).
        
        Args:
            diversidades: (n_iteracoes,) - Diversidade média
            limiar_variacao: float - Abaixo disso, considera convergência prematura
            
        Returns:
            int - Índice da iteração onde o algoritmo "travou"
                  Retorna -1 se nunca atingiu o limiar
        """
        idx = np.where(diversidades < limiar_variacao)[0]
        if len(idx) > 0:
            return int(idx[0])
        return -1
    
    @staticmethod
    def diametro_populacao(populacoes: List[np.ndarray]) -> np.ndarray:
        """
        Diâmetro do espaço ocupado pela população (max - min de cada dimensão).
        
        Medida alternativa de diversidade mais robusta que desvio padrão.
        
        Returns:
            (n_iteracoes, n_dimensoes) - Diâmetro por dimensão
        """
        diametros = []
        for pop in populacoes:
            diam = np.max(pop, axis=0) - np.min(pop, axis=0)
            diametros.append(diam)
        return np.array(diametros)
    
    @staticmethod
    def entropia_populacao(populacoes: List[np.ndarray], 
                          n_bins: int = 10) -> np.ndarray:
        """
        Entropia de Shannon da distribuição de cada dimensão.
        
        Medida informação-teórica de dispersão.
        
        Args:
            populacoes: Lista de populações
            n_bins: Número de bins para histograma
            
        Returns:
            (n_iteracoes,) - Entropia média por iteração
        """
        entropias = []
        for pop in populacoes:
            ent_dims = []
            for dim in range(pop.shape[1]):
                hist, _ = np.histogram(pop[:, dim], bins=n_bins)
                hist = hist[hist > 0]  # Remove zeros
                p = hist / hist.sum()
                entropia = -np.sum(p * np.log2(p + 1e-10))
                ent_dims.append(entropia)
            entropias.append(np.mean(ent_dims))
        return np.array(entropias)
    
    @staticmethod
    def relatorio_diversidade(populacoes: List[np.ndarray],
                             titulo_algoritmo: str = "") -> Dict:
        """
        Gera relatório completo de diversidade.
        
        Returns:
            Dict com todas as métricas calculadas
        """
        div_media = MetricasDiversidade.diversidade_media_iteracao(populacoes)
        taxa_queda = MetricasDiversidade.taxa_perda_diversidade(div_media)
        ponto_conv = MetricasDiversidade.ponto_convergencia_prematura(div_media)
        entropia = MetricasDiversidade.entropia_populacao(populacoes)
        
        return {
            'algoritmo': titulo_algoritmo,
            'diversidade_media_inicial': float(div_media[0]),
            'diversidade_media_final': float(div_media[-1]),
            'taxa_queda_diversidade': float(taxa_queda),
            'iteracao_convergencia_prematura': int(ponto_conv),
            'entropia_media': float(np.mean(entropia)),
            'entropia_final': float(entropia[-1]),
            'diversidade_historico': div_media,
            'entropia_historico': entropia
        }
