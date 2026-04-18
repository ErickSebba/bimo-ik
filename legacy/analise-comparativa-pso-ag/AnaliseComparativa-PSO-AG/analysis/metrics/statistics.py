"""
Análises Estatísticas para Validação Científica
Testes t-Student, ANOVA, boxplots, gráficos de convergência
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import json


class AnalisaEstatistica:
    """
    Análises estatísticas para comparação válida de algoritmos.
    
    Essencial para artigos científicos: mostra diferenças significativas.
    """
    
    @staticmethod
    def teste_t_student(dados_alg1: np.ndarray, 
                       dados_alg2: np.ndarray,
                       bilateral: bool = True) -> Dict:
        """
        Teste t-Student para comparação de duas amostras.
        
        Args:
            dados_alg1, dados_alg2: Arrays com métricas de cada algoritmo
            bilateral: Se True, teste bilateral (diferente); False = unilateral
            
        Returns:
            Dict com t-statistic, p-value, significância
        """
        t_stat, p_value = stats.ttest_ind(dados_alg1, dados_alg2)
        
        alpha = 0.05
        significante = p_value < alpha
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significante_alpha_0.05': bool(significante),
            'alg1_media': float(np.mean(dados_alg1)),
            'alg2_media': float(np.mean(dados_alg2)),
            'alg1_desvio': float(np.std(dados_alg1)),
            'alg2_desvio': float(np.std(dados_alg2))
        }
    
    @staticmethod
    def teste_anova(dados_multiplos: Dict[str, np.ndarray]) -> Dict:
        """
        Teste ANOVA para comparação de 3+ amostras.
        
        Args:
            dados_multiplos: Dict { 'algoritmo_nome': array_dados }
            
        Returns:
            Dict com F-statistic, p-value, significância
        """
        amostras = list(dados_multiplos.values())
        f_stat, p_value = stats.f_oneway(*amostras)
        
        alpha = 0.05
        significante = p_value < alpha
        
        # Médias por grupo
        medias = {nome: float(np.mean(dados)) 
                 for nome, dados in dados_multiplos.items()}
        
        return {
            'F_statistic': float(f_stat),
            'p_value': float(p_value),
            'significante_alpha_0.05': bool(significante),
            'medias_por_grupo': medias
        }
    
    @staticmethod
    def teste_wilcoxon(dados_alg1: np.ndarray,
                      dados_alg2: np.ndarray) -> Dict:
        """
        Teste Wilcoxon (não-paramétrico) para dados que não seguem distribuição normal.
        
        Mais robusto que t-Student em alguns casos.
        
        Args:
            dados_alg1, dados_alg2: Arrays com métricas pareadas
            
        Returns:
            Dict com estatística e p-value
        """
        # Testa se dados têm mesmo tamanho (pareados)
        if len(dados_alg1) != len(dados_alg2):
            return {'erro': 'Dados devem ter mesmo tamanho (pareados)'}
        
        stat, p_value = stats.wilcoxon(dados_alg1, dados_alg2)
        
        return {
            'wilcoxon_statistic': float(stat),
            'p_value': float(p_value),
            'significante_alpha_0.05': bool(p_value < 0.05)
        }
    
    @staticmethod
    def normalidade_shapiro(dados: np.ndarray) -> Dict:
        """
        Teste Shapiro-Wilk para normalidade dos dados.
        
        Necessário para decidir entre testes paramétricos (t-test) e não-paramétricos.
        
        Args:
            dados: Array com dados
            
        Returns:
            Dict com estatística e p-value (p > 0.05 = distribuição normal)
        """
        stat, p_value = stats.shapiro(dados)
        
        normal = p_value > 0.05
        
        return {
            'shapiro_statistic': float(stat),
            'p_value': float(p_value),
            'eh_normal': bool(normal),
            'interpretacao': 'Dados seguem distribuição normal' if normal else 'Dados NÃO seguem distribuição normal'
        }
    
    @staticmethod
    def size_effect_cohens_d(dados_alg1: np.ndarray,
                            dados_alg2: np.ndarray) -> float:
        """
        Calcula d de Cohen - tamanho do efeito (magnitude da diferença).
        
        d < 0.2: negligenciável
        0.2 ≤ d < 0.5: pequeno
        0.5 ≤ d < 0.8: médio
        d ≥ 0.8: grande
        
        Args:
            dados_alg1, dados_alg2: Arrays com métricas
            
        Returns:
            float - d de Cohen
        """
        n1, n2 = len(dados_alg1), len(dados_alg2)
        mean1, mean2 = np.mean(dados_alg1), np.mean(dados_alg2)
        var1, var2 = np.var(dados_alg1, ddof=1), np.var(dados_alg2, ddof=1)
        
        # Desvio padrão pooled
        sp = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if sp == 0:
            return 0.0
        
        d = (mean1 - mean2) / sp
        return float(d)
    
    @staticmethod
    def intervalo_confianca(dados: np.ndarray, 
                           confianca: float = 0.95) -> Tuple[float, float]:
        """
        Calcula intervalo de confiança para a média.
        
        Args:
            dados: Array com dados
            confianca: Nível de confiança (default 95%)
            
        Returns:
            (limite_inferior, limite_superior) - Intervalo para a média
        """
        media = np.mean(dados)
        se = stats.sem(dados)  # Erro padrão
        margem = se * stats.t.ppf((1 + confianca) / 2, len(dados) - 1)
        
        return float(media - margem), float(media + margem)
    
    @staticmethod
    def resumo_estatistico_completo(resultados_por_algoritmo: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """
        Gera resumo estatístico completo para publicação.
        
        Args:
            resultados_por_algoritmo: Dict com resultado por algoritmo
                {
                    'PSO': { 'erro_final': [...], 'jerk': [...] },
                    'AG': { 'erro_final': [...], 'jerk': [...] },
                    'PSO-DFL': { 'erro_final': [...], 'jerk': [...] }
                }
        
        Returns:
            Dict com análises completas
        """
        resumo = {}
        
        # Para cada métrica
        metricas = list(list(resultados_por_algoritmo.values())[0].keys())
        
        for metrica in metricas:
            dados_metrica = {}
            for alg, resultados in resultados_por_algoritmo.items():
                if metrica in resultados:
                    dados_metrica[alg] = np.array(resultados[metrica])
            
            if len(dados_metrica) < 2:
                continue
            
            resumo_metrica = {'metrica': metrica}
            
            # Estatísticas descritivas por algoritmo
            descritivas = {}
            for alg, dados in dados_metrica.items():
                descritivas[alg] = {
                    'media': float(np.mean(dados)),
                    'mediana': float(np.median(dados)),
                    'desvio': float(np.std(dados)),
                    'min': float(np.min(dados)),
                    'max': float(np.max(dados))
                }
            resumo_metrica['descritivas'] = descritivas
            
            # Testes comparativos
            algs_list = list(dados_metrica.keys())
            
            if len(algs_list) == 2:
                # Teste t-Student bilateral
                resumo_metrica['t_student'] = AnalisaEstatistica.teste_t_student(
                    dados_metrica[algs_list[0]], 
                    dados_metrica[algs_list[1]]
                )
                # Tamanho do efeito
                resumo_metrica['cohens_d'] = float(AnalisaEstatistica.size_effect_cohens_d(
                    dados_metrica[algs_list[0]], 
                    dados_metrica[algs_list[1]]
                ))
            elif len(algs_list) >= 3:
                # ANOVA
                resumo_metrica['anova'] = AnalisaEstatistica.teste_anova(dados_metrica)
            
            resumo[metrica] = resumo_metrica
        
        return resumo
    
    @staticmethod
    def salvar_resumo_json(resumo: Dict, caminho_arquivo: str):
        """
        Salva resumo estatístico em arquivo JSON formatado.
        """
        with open(caminho_arquivo, 'w') as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)
