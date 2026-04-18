"""
Métricas de Convergência
Taxa de convergência, número de iterações, histórico de erro
"""

import numpy as np
from typing import Tuple, Dict, List


class MetricasConvergencia:
    """
    Calcula velocidade e qualidade da convergência dos algoritmos.
    
    PSO converge mais rápido inicialmente, AG mais consistentemente.
    """
    
    @staticmethod
    def taxa_convergencia(historico_erro: np.ndarray, 
                         metodo: str = 'linear') -> float:
        """
        Taxa média de melhoria por iteração (percentual).
        
        Args:
            historico_erro: (n_iteracoes,) - Fitness ao longo das iterações
            metodo: 'linear' ou 'exponencial'
            
        Returns:
            float - Taxa de convergência (% melhoria por iteração)
        """
        if len(historico_erro) < 2:
            return 0.0
        
        erros = np.array(historico_erro)
        inicial = erros[0]
        final = erros[-1]
        
        if inicial == 0:
            return 0.0
        
        if metodo == 'linear':
            # Melhoria total dividida pelo número de iterações
            melhoria = (inicial - final) / inicial
            taxa = (melhoria / len(erros)) * 100
        else:  # exponencial
            # Assumir decaimento exponencial: e(i) = e0 * exp(-k*i)
            # Extrair k do log
            if final > 0 and inicial > final:
                k = -np.log(final / inicial) / len(erros)
                taxa = k * 100
            else:
                taxa = 0.0
        
        return taxa
    
    @staticmethod
    def iteracoes_para_tolerancia(historico_erro: np.ndarray,
                                 tolerancia_percentual: float = 5.0) -> int:
        """
        Quantas iterações até atingir erro relativo < tolerância.
        
        Args:
            historico_erro: (n_iteracoes,)
            tolerancia_percentual: e.g., 5 significa erro_final ≤ 1.05 * erro_minimo
            
        Returns:
            int - Iteração onde tolerância foi atingida (ou -1 se nunca)
        """
        erros = np.array(historico_erro)
        erro_minimo = np.min(erros)
        limiar = erro_minimo * (1 + tolerancia_percentual / 100)
        
        idx = np.where(erros <= limiar)[0]
        if len(idx) > 0:
            return int(idx[0])
        return -1
    
    @staticmethod
    def melhoria_relativa(historico_erro: np.ndarray) -> np.ndarray:
        """
        Melhoria acumulada relativa ao valor inicial (%).
        
        Returns:
            (n_iteracoes,) - % de melhoria em relação ao erro inicial
        """
        erros = np.array(historico_erro)
        inicial = erros[0]
        if inicial == 0:
            return np.zeros_like(erros)
        melhoria = (inicial - erros) / inicial * 100
        return np.clip(melhoria, 0, 100)  # Garante que não fica negativo
    
    @staticmethod
    def velocidade_convergencia_fases(historico_erro: np.ndarray,
                                     n_fases: int = 3) -> Dict[str, float]:
        """
        Calcula taxa de convergência em diferentes fases (inicio, meio, fim).
        
        Útil para detectar PSO rápido no início vs AG estável no fim.
        
        Returns:
            Dict com taxa em cada fase
        """
        erros = np.array(historico_erro)
        n_iter = len(erros)
        size_fase = n_iter // n_fases
        
        taxas = {}
        for i in range(n_fases):
            inicio = i * size_fase
            fim = (i + 1) * size_fase if i < n_fases - 1 else n_iter
            fase_erros = erros[inicio:fim]
            taxa = MetricasConvergencia.taxa_convergencia(fase_erros)
            taxas[f'fase_{i+1}'] = taxa
        
        return taxas
    
    @staticmethod
    def suavidade_convergencia(historico_erro: np.ndarray) -> float:
        """
        Calcula suavidade da convergência (quanto menos oscilação, melhor).
        
        Baseado na variância das diferenças (segunda derivada).
        
        Returns:
            float - Score de suavidade [0, 1], onde 1 é perfeitamente suave
        """
        erros = np.array(historico_erro)
        if len(erros) < 3:
            return 1.0
        
        # Primeira derivada (variação)
        diffs = np.diff(erros)
        
        # Segunda derivada (curvatura)
        curvatura = np.abs(np.diff(diffs))
        
        # Score: inverso normalizado da curvatura média
        media_curvatura = np.mean(curvatura)
        max_curvatura = np.max(np.abs(diffs))
        
        if max_curvatura == 0:
            return 1.0
        
        score = 1.0 / (1.0 + (media_curvatura / max_curvatura))
        return float(np.clip(score, 0, 1))
    
    @staticmethod
    def relatorio_convergencia(historico_erro: np.ndarray,
                              titulo_algoritmo: str = "") -> Dict:
        """
        Gera relatório completo de convergência.
        
        Returns:
            Dict com todas as métricas calculadas
        """
        erros = np.array(historico_erro)
        
        return {
            'algoritmo': titulo_algoritmo,
            'erro_inicial': float(erros[0]),
            'erro_final': float(erros[-1]),
            'erro_minimo': float(np.min(erros)),
            'erro_media': float(np.mean(erros)),
            'taxa_convergencia_linear': float(MetricasConvergencia.taxa_convergencia(erros, 'linear')),
            'taxa_convergencia_exp': float(MetricasConvergencia.taxa_convergencia(erros, 'exponencial')),
            'iteracoes_para_tol_5pct': int(MetricasConvergencia.iteracoes_para_tolerancia(erros, 5.0)),
            'suavidade_convergencia': float(MetricasConvergencia.suavidade_convergencia(erros)),
            'melhoria_relativa': float((erros[0] - erros[-1]) / erros[0] * 100),
            'velocidade_por_fase': MetricasConvergencia.velocidade_convergencia_fases(erros)
        }
