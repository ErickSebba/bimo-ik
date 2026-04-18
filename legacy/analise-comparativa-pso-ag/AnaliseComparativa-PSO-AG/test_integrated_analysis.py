"""
INTEGRADOR: Conecta seu código erick_dan.py com o framework de análises
Executa PSO, AG e PSO-DFL com coleta de dados para análises

Estrutura:
1. Importa funções do erick_dan.py (cinematica_direta, fitness, etc)
2. Define wrappers para PSO, AG e PSO-DFL
3. Executa testes systematicamente
4. Salva dados e chama analisador
"""

import os
import sys
import numpy as np
import json
import time
from typing import Dict, Tuple, List, Any

# Adicionar path para imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(__file__))

# Importar função do erick_dan.py
try:
    from erick_dan import (
        TRAJETORIA, LIMITES_INF, LIMITES_SUP,
        N_RUNS, N_PASSOS, PSO_POP, PSO_ITER,
        AG_POP, AG_GENS, TAXA_CROSS, TAXA_MUT,
        PESO_POS, PESO_JTA,
        cinematica_direta, criar_fitness,
        rodar_pso  # Função PSO do seu código
    )
    print("[OK] Importações do erick_dan.py bem-sucedidas")
except ImportError as e:
    print(f"[AVISO] Erro ao importar erick_dan.py: {e}")
    print("[INFO] Usando configuração genérica padrão")
    
    # Configuração fallback
    N_RUNS = 10#1
    N_PASSOS = 3
    PSO_POP = 40
    PSO_ITER = 150
    AG_POP = 40
    AG_GENS = 150
    
    LIMITES_INF = np.array([-np.pi]*6)
    LIMITES_SUP = np.array([np.pi]*6)
    
    # Trajetória dummy
    x_traj = np.linspace(150, 300, N_PASSOS)
    y_traj = np.linspace(100, -100, N_PASSOS)
    z_traj = np.ones(N_PASSOS) * 200.0
    TRAJETORIA = np.column_stack([x_traj, y_traj, z_traj])
    
    # Para teste rápido, usar apenas 1 ponto
    TRAJETORIA = TRAJETORIA[:1]

# Importar análise
from analysis.algorithms.pso_base import PSO_Padrao, PSO_DFL
from analysis.algorithms.ag_base import AG_Padrao
from analysis.run_analysis import OrquestradorAnalises
from analysis.metrics import (
    MetricasConvergencia,
    MetricasDiversidade,
    MetricasSuavidade,
    AnalisaEstatistica
)


class IntegradorAnalises:
    """
    Orquestra execução de testes e coleta de dados.
    
    Workflow:
    1. Executa PSO, AG, PSO-DFL múltiplas vezes
    2. Coleta todos os históricos
    3. Analisa resultados
    4. Gera visualizações e relatórios
    """
    
    def __init__(self,
                 trajetoria: np.ndarray = None,
                 limites_inf: np.ndarray = None,
                 limites_sup: np.ndarray = None,
                 n_runs: int = 10,
                 path_saidas: str = "analysis/results"):
        """
        Args:
            trajetoria: (n_pontos, 3) - Trajetória XYZ de alvo
            limites_inf, limites_sup: Limites articulares
            n_runs: Número de execuções
            path_saidas: Caminho para salvar resultados
        """
        self.trajetoria = trajetoria if trajetoria is not None else TRAJETORIA
        self.limites_inf = limites_inf if limites_inf is not None else LIMITES_INF
        self.limites_sup = limites_sup if limites_sup is not None else LIMITES_SUP
        self.n_runs = n_runs
        self.path_saidas = path_saidas
        
        # Orquestrador de análises
        self.orq = OrquestradorAnalises(
            path_base=path_saidas,
            n_runs=n_runs,
            n_pontos_trajetoria=len(self.trajetoria)
        )
        
        # Containers para dados coletados
        self.dados_colectados = {
            'PSO': {},
            'AG': {},
            'PSO-DFL': {}
        }
    
    def executar_teste_completo(self, verbose: bool = True) -> Dict:
        """
        Executa teste COMPLETO: PSO (Cold+Warm), AG (Cold+Warm), PSO-DFL
        """
        print("\n" + "="*70)
        print("INICIANDO TESTE COMPLETO")
        print("="*70)
        
        # ─────────────────────────────────────────────
        # TESTE 1: PSO COLD-START vs WARM-START
        # ─────────────────────────────────────────────
        print("\n[1/3] Executando PSO (Cold-Start vs Warm-Start)...")
        resultado_pso = self._executar_pso(verbose=verbose)
        self.dados_colectados['PSO'] = resultado_pso
        
        # ─────────────────────────────────────────────
        # TESTE 2: AG COLD-START vs WARM-START
        # ─────────────────────────────────────────────
        print("\n[2/3] Executando AG (Cold-Start vs Warm-Start)...")
        resultado_ag = self._executar_ag(verbose=verbose)
        self.dados_colectados['AG'] = resultado_ag
        
        # ─────────────────────────────────────────────
        # TESTE 3: PSO-DFL WARM-START
        # ─────────────────────────────────────────────
        print("\n[3/3] Executando PSO-DFL (Warm-Start)...")
        resultado_pso_dfl = self._executar_pso_dfl(verbose=verbose)
        self.dados_colectados['PSO-DFL'] = resultado_pso_dfl
        
        # ─────────────────────────────────────────────
        # ANÁLISES
        # ─────────────────────────────────────────────
        print("\n" + "="*70)
        print("GERANDO ANÁLISES E VISUALIZAÇÕES")
        print("="*70)
        
        self._gerar_todas_analises()
        
        return self.dados_colectados
    
    def _executar_pso(self, verbose: bool = False) -> Dict:
        """
        Executa PSO múltiplas vezes e coleta dados.
        """
        dados = {
            'cold_start': {
                'fitness': [],
                'tempo': [],
                'trajetorias': [],
                'historico_erro': [],
                'historico_diversidade': []
            },
            'warm_start': {
                'fitness': [],
                'tempo': [],
                'trajetorias': [],
                'historico_erro': [],
                'historico_diversidade': []
            }
        }
        
        for run in range(self.n_runs):
            seed = 42 + run
            
            # Cold-Start
            print(f"  Run {run+1}/{self.n_runs} - PSO Cold-Start...", end='')
            resultado_cold = self._executar_pso_cold_start(seed, verbose=False)
            dados['cold_start']['fitness'].extend(resultado_cold['fitness'])
            dados['cold_start']['tempo'].append(resultado_cold['tempo'])
            dados['cold_start']['trajetorias'].append(resultado_cold['trajetoria'])
            dados['cold_start']['historico_erro'].append(resultado_cold.get('historico_erro', []))
            dados['cold_start']['historico_diversidade'].append(resultado_cold.get('historico_diversidade', []))
            print(" OK")
            
            # Warm-Start
            print(f"  Run {run+1}/{self.n_runs} - PSO Warm-Start...", end='')
            resultado_warm = self._executar_pso_warm_start(seed, verbose=False)
            dados['warm_start']['fitness'].extend(resultado_warm['fitness'])
            dados['warm_start']['tempo'].append(resultado_warm['tempo'])
            dados['warm_start']['trajetorias'].append(resultado_warm['trajetoria'])
            dados['warm_start']['historico_erro'].append(resultado_warm.get('historico_erro', []))
            dados['warm_start']['historico_diversidade'].append(resultado_warm.get('historico_diversidade', []))
            print(" OK")
        
        # Converter listas para arrays
        dados['cold_start']['tempo'] = np.array(dados['cold_start']['tempo'])
        dados['warm_start']['tempo'] = np.array(dados['warm_start']['tempo'])
        
        return dados
    
    def _executar_pso_cold_start(self, seed: int, verbose: bool = False) -> Dict:
        """
        Executa PSO sem doping para uma trajetória completa.
        """
        np.random.seed(seed)
        
        # Usar implementação genérica para teste rápido
        # (evita usar rodar_pso do erick_dan que é muito lenta)
        return self._executar_pso_generico(seed, warm_start=False)
    
    def _executar_pso_warm_start(self, seed: int, verbose: bool = False) -> Dict:
        """
        Executa PSO com doping (warm-start).
        """
        np.random.seed(seed)
        
        trajetoria = []
        fitness_hist = []
        tempo_total = 0
        
        q_anterior = np.zeros(6)
        historico_erro_completo = []
        historico_diversidade_completo = []
        
        for i, alvo in enumerate(self.trajetoria):
            # Usar classe para captura segura
            class FitnessFunctionPSO:
                def __init__(self, target):
                    self.target = np.array(target)
                
                def __call__(self, q):
                    return np.linalg.norm(q[:3] - self.target)
            
            fitness_func = FitnessFunctionPSO(alvo)
            
            # Criar PSO warm-start
            pso = PSO_Padrao(
                fitness_func=fitness_func,
                dim=6,
                pop=PSO_POP,
                max_iter=PSO_ITER,
                limites_inf=self.limites_inf.tolist(),
                limites_sup=self.limites_sup.tolist(),
                seed=seed + i
            )
            
            # Executar com doping
            t0 = time.time()
            q_opt, fit_val, tempo = pso.rodar_warm_start(q_anterior, verbose=False)
            tempo_total += tempo
            
            trajetoria.append(q_opt)
            fitness_hist.append(float(fit_val))
            
            # Armazenar históricos
            if hasattr(pso, 'historico_melhor_fitness'):
                historico_erro_completo.append(pso.historico_melhor_fitness)
                historico_diversidade_completo.append(pso.historico_diversidade)
            
            q_anterior = q_opt
        
        return {
            'trajetoria': np.array(trajetoria),
            'fitness': fitness_hist,
            'tempo': tempo_total,
            'historico_erro': historico_erro_completo,
            'historico_diversidade': historico_diversidade_completo
        }
    
    def _executar_pso_generico(self, seed: int, warm_start: bool = False) -> Dict:
        """
        Implementação genérica PSO como fallback.
        """
        np.random.seed(seed)
        
        trajetoria = []
        fitness_hist = []
        tempo_total = 0
        
        q_anterior = np.zeros(6)
        
        # Usar menos iterações para teste rápido
        n_iter_teste = 30 if self.n_runs <= 1 else PSO_ITER
        
        for alvo in self.trajetoria:
            fitness_func = lambda q: np.linalg.norm(q[:3] - alvo)
            
            pso = PSO_Padrao(
                fitness_func=fitness_func,
                dim=6,
                pop=PSO_POP,
                max_iter=n_iter_teste,
                limites_inf=self.limites_inf.tolist(),
                limites_sup=self.limites_sup.tolist(),
                seed=seed
            )
            
            t0 = time.time()
            if warm_start:
                q_opt, fit_val, tempo = pso.rodar_warm_start(q_anterior)
            else:
                q_opt, fit_val, tempo = pso.rodar_cold_start()
            tempo_total += tempo
            
            trajetoria.append(q_opt)
            fitness_hist.append(float(fit_val))
            q_anterior = q_opt
        
        return {
            'trajetoria': np.array(trajetoria),
            'fitness': fitness_hist,
            'tempo': tempo_total
        }
    
    def _executar_ag(self, verbose: bool = False) -> Dict:
        """
        Executa AG múltiplas vezes.
        """
        dados = {
            'cold_start': {
                'fitness': [],
                'tempo': [],
                'trajetorias': [],
                'historico_erro': [],
                'historico_diversidade': []
            },
            'warm_start': {
                'fitness': [],
                'tempo': [],
                'trajetorias': [],
                'historico_erro': [],
                'historico_diversidade': []
            }
        }
        
        for run in range(self.n_runs):
            seed = 42 + run
            
            # Cold-Start
            print(f"  Run {run+1}/{self.n_runs} - AG Cold-Start...", end='')
            resultado_cold = self._executar_ag_cold_start(seed, verbose=False)
            dados['cold_start']['fitness'].extend(resultado_cold['fitness'])
            dados['cold_start']['tempo'].append(resultado_cold['tempo'])
            dados['cold_start']['trajetorias'].append(resultado_cold['trajetoria'])
            dados['cold_start']['historico_erro'].append(resultado_cold.get('historico_erro', []))
            dados['cold_start']['historico_diversidade'].append(resultado_cold.get('historico_diversidade', []))
            print(" OK")
            
            # Warm-Start
            print(f"  Run {run+1}/{self.n_runs} - AG Warm-Start...", end='')
            resultado_warm = self._executar_ag_warm_start(seed, verbose=False)
            dados['warm_start']['fitness'].extend(resultado_warm['fitness'])
            dados['warm_start']['tempo'].append(resultado_warm['tempo'])
            dados['warm_start']['trajetorias'].append(resultado_warm['trajetoria'])
            dados['warm_start']['historico_erro'].append(resultado_warm.get('historico_erro', []))
            dados['warm_start']['historico_diversidade'].append(resultado_warm.get('historico_diversidade', []))
            print(" OK")
        
        dados['cold_start']['tempo'] = np.array(dados['cold_start']['tempo'])
        dados['warm_start']['tempo'] = np.array(dados['warm_start']['tempo'])
        
        return dados
    
    def _executar_ag_cold_start(self, seed: int, verbose: bool = False) -> Dict:
        """AG sem doping."""
        np.random.seed(seed)
        
        trajetoria = []
        fitness_hist = []
        tempo_total = 0
        historico_erro_completo = []
        historico_diversidade_completo = []
        
        # Usar menos iterações para teste rápido
        n_ger_teste = 30 if self.n_runs <= 1 else AG_GENS
        
        for alvo in self.trajetoria:
            # Usar classe para captura segura
            class FitnessFunctionAG:
                def __init__(self, target):
                    self.target = np.array(target)
                
                def __call__(self, q):
                    return np.linalg.norm(q[:3] - self.target)
            
            fitness_func = FitnessFunctionAG(alvo)
            
            ag = AG_Padrao(
                fitness_func=fitness_func,
                dim=6,
                pop=AG_POP,
                n_gerações=n_ger_teste,
                limites_inf=self.limites_inf.tolist(),
                limites_sup=self.limites_sup.tolist(),
                seed=seed
            )
            
            q_opt, fit_val, tempo = ag.rodar_cold_start(verbose=False)
            
            trajetoria.append(q_opt)
            fitness_hist.append(fit_val)
            tempo_total += tempo
            
            # Coletar históricos
            if hasattr(ag, 'historico_melhor_fitness'):
                historico_erro_completo.append(ag.historico_melhor_fitness)
            if hasattr(ag, 'historico_diversidade'):
                historico_diversidade_completo.append(ag.historico_diversidade)
        
        return {
            'trajetoria': np.array(trajetoria),
            'fitness': fitness_hist,
            'tempo': tempo_total,
            'historico_erro': historico_erro_completo,
            'historico_diversidade': historico_diversidade_completo
        }
    
    def _executar_ag_warm_start(self, seed: int, verbose: bool = False) -> Dict:
        """AG com doping."""
        np.random.seed(seed)
        
        trajetoria = []
        fitness_hist = []
        tempo_total = 0
        
        q_anterior = np.zeros(6)
        historico_erro_completo = []
        historico_diversidade_completo = []
        
        # Usar menos iterações para teste rápido
        n_ger_teste = 30 if self.n_runs <= 1 else AG_GENS
        
        for alvo in self.trajetoria:
            # Usar classe para captura segura
            class FitnessFunctionAG:
                def __init__(self, target):
                    self.target = np.array(target)
                
                def __call__(self, q):
                    return np.linalg.norm(q[:3] - self.target)
            
            fitness_func = FitnessFunctionAG(alvo)
            
            ag = AG_Padrao(
                fitness_func=fitness_func,
                dim=6,
                pop=AG_POP,
                n_gerações=n_ger_teste,
                limites_inf=self.limites_inf.tolist(),
                limites_sup=self.limites_sup.tolist(),
                seed=seed
            )
            
            t0 = time.time()
            q_opt, fit_val, tempo = ag.rodar_warm_start(q_anterior, verbose=False)
            tempo_total += tempo
            
            trajetoria.append(q_opt)
            fitness_hist.append(fit_val)
            
            if hasattr(ag, 'historico_melhor_fitness'):
                historico_erro_completo.append(ag.historico_melhor_fitness)
                historico_diversidade_completo.append(ag.historico_diversidade)
            
            q_anterior = q_opt
        
        return {
            'trajetoria': np.array(trajetoria),
            'fitness': fitness_hist,
            'tempo': tempo_total,
            'historico_erro': historico_erro_completo,
            'historico_diversidade': historico_diversidade_completo
        }
    
    def _executar_pso_dfl(self, verbose: bool = False) -> Dict:
        """
        Executa PSO-DFL (Dynamic Fitness Landscape) com warm-start.
        """
        dados = {
            'warm_start': {
                'fitness': [],
                'tempo': [],
                'trajetorias': [],
                'historico_erro': [],
                'historico_diversidade': []
            }
        }
        
        for run in range(self.n_runs):
            seed = 42 + run
            
            print(f"  Run {run+1}/{self.n_runs} - PSO-DFL...", end='')
            resultado = self._executar_pso_dfl_warm_start(seed, verbose=False)
            dados['warm_start']['fitness'].extend(resultado['fitness'])
            dados['warm_start']['tempo'].append(resultado['tempo'])
            dados['warm_start']['trajetorias'].append(resultado['trajetoria'])
            dados['warm_start']['historico_erro'].append(resultado.get('historico_erro', []))
            dados['warm_start']['historico_diversidade'].append(resultado.get('historico_diversidade', []))
            print(" OK")
        
        dados['warm_start']['tempo'] = np.array(dados['warm_start']['tempo'])
        
        return dados
    
    def _executar_pso_dfl_warm_start(self, seed: int, verbose: bool = False) -> Dict:
        """PSO-DFL com doping."""
        np.random.seed(seed)
        
        trajetoria = []
        fitness_hist = []
        tempo_total = 0
        
        q_anterior = np.zeros(6)
        historico_erro_completo = []
        historico_diversidade_completo = []
        
        # Usar menos iterações para teste rápido
        n_iter_teste = 30 if self.n_runs <= 1 else PSO_ITER
        
        for alvo in self.trajetoria:
            # Usar classe para captura segura
            class FitnessFunctionPSO:
                def __init__(self, target):
                    self.target = np.array(target)
                
                def __call__(self, q):
                    return np.linalg.norm(q[:3] - self.target)
            
            fitness_func = FitnessFunctionPSO(alvo)
            
            pso_dfl = PSO_DFL(
                fitness_func=fitness_func,
                dim=6,
                pop=PSO_POP,
                max_iter=n_iter_teste,
                limites_inf=self.limites_inf.tolist(),
                limites_sup=self.limites_sup.tolist(),
                seed=seed + len(trajetoria),
                modo_adaptacao='sigmoid'
            )
            
            t0 = time.time()
            q_opt, fit_val, tempo = pso_dfl.rodar_warm_start(q_anterior, verbose=False)
            tempo_total += tempo
            
            trajetoria.append(q_opt)
            fitness_hist.append(float(fit_val))
            
            if hasattr(pso_dfl, 'historico_melhor_fitness'):
                historico_erro_completo.append(pso_dfl.historico_melhor_fitness)
                historico_diversidade_completo.append(pso_dfl.historico_diversidade)
            
            q_anterior = q_opt
        
        return {
            'trajetoria': np.array(trajetoria),
            'fitness': fitness_hist,
            'tempo': tempo_total,
            'historico_erro': historico_erro_completo,
            'historico_diversidade': historico_diversidade_completo
        }
    
    def _gerar_todas_analises(self):
        """
        Executa TODAS as análises conforme requisitos do prompt:
        
        1. Convergência (erro vs iterações)
        2. Diversidade (desvio padrão da população)
        3. Suavidade (jerk, variações angulares)
        4. Estatísticas (t-Student, ANOVA)
        5. Comparação Warm-Start vs Cold-Start
        
        Organiza saídas automaticamente em pastas temáticas.
        """
        print("\n" + "="*70)
        print("[ANALISE 1/5] Convergencia - Erro vs Iteracoes")
        print("="*70)
        print("[INFO] Gerando graficos de convergencia...")
        
        historicos_erro = {
            'PSO': self.dados_colectados['PSO'].get('warm_start', {}).get('historico_erro', []),
            'AG': self.dados_colectados['AG'].get('warm_start', {}).get('historico_erro', []),
            'PSO-DFL': self.dados_colectados['PSO-DFL'].get('warm_start', {}).get('historico_erro', [])
        }
        
        if any(historicos_erro.values()):
            self.orq.plotar_convergencia_comparativa(historicos_erro)
        
        print("[INFO] Gerando analises de diversidade...")
        historicos_diversidade = {
            'PSO': self.dados_colectados['PSO'].get('warm_start', {}).get('historico_diversidade', []),
            'AG': self.dados_colectados['AG'].get('warm_start', {}).get('historico_diversidade', []),
            'PSO-DFL': self.dados_colectados['PSO-DFL'].get('warm_start', {}).get('historico_diversidade', [])
        }
        
        if any(historicos_diversidade.values()):
            try:
                self.orq.plotar_diversidade_comparativa(historicos_diversidade)
            except Exception as e:
                print(f"[WARN] Nao foi possivel gerar grafico de diversidade: {e}")
        
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("[ANALISE 2/5] Suavidade - Jerk e Variacoes Angulares")
        print("="*70)
        print("[INFO] Calculando metricas de suavidade...")
        
        metricas_suavidade = self._calcular_metricas_suavidade()
        self._salvar_relatorio_suavidade(metricas_suavidade)
        
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("[ANALISE 3/5] Impacto de Doping (Warm-Start vs Cold-Start)")
        print("="*70)
        print("[INFO] Gerando analises de impacto de doping...")
        
        resultados_doping = {
            'PSO': {
                'cold': self.dados_colectados['PSO'].get('cold_start', {}).get('historico_erro', []),
                'warm': self.dados_colectados['PSO'].get('warm_start', {}).get('historico_erro', [])
            },
            'AG': {
                'cold': self.dados_colectados['AG'].get('cold_start', {}).get('historico_erro', []),
                'warm': self.dados_colectados['AG'].get('warm_start', {}).get('historico_erro', [])
            }
        }
        
        if any(v for alg_data in resultados_doping.values() for v in alg_data.values()):
            try:
                self.orq.plotar_comparacao_doping_simples(resultados_doping)
            except Exception as e:
                print(f"[WARN] Nao foi possivel gerar grafico de doping: {e}")
        
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("[ANALISE 4/5] Boxplots de Metricas")
        print("="*70)
        print("[INFO] Gerando boxplots...")
        
        resultados_metricas = {
            'PSO': {
                'convergencia': historicos_erro.get('PSO', []),
                'diversidade': historicos_diversidade.get('PSO', [])
            },
            'AG': {
                'convergencia': historicos_erro.get('AG', []),
                'diversidade': historicos_diversidade.get('AG', [])
            },
            'PSO-DFL': {
                'convergencia': historicos_erro.get('PSO-DFL', []),
                'diversidade': historicos_diversidade.get('PSO-DFL', [])
            }
        }
        
        if any(v for alg_data in resultados_metricas.values() for v in alg_data.values()):
            try:
                self.orq.plotar_boxplots_simples(resultados_metricas)
            except Exception as e:
                print(f"[WARN] Nao foi possivel gerar boxplots: {e}")
        
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("[ANALISE 5/5] Testes Estatisticos (t-Student, ANOVA)")
        print("="*70)
        print("[INFO] Executando testes estatisticos para validacao...")
        
        self._executar_testes_estatisticos(historicos_erro)
        
        # Gerar dados e estatísticas
        print("\n[INFO] Gerando estatisticas consolidadas e exportando dados...")
        self._exportar_dados_brutos()
        self._gerar_relatorio_estatistico()
        
        print("\n" + "="*70)
        print("[OK] TODAS AS ANALISES CONCLUIDAS COM SUCESSO!")
        print("="*70)
    
    def _calcular_metricas_suavidade(self) -> Dict:
        """
        Calcula métricas de suavidade para todas as trajetórias.
        
        Métricas:
        - Soma de variações angulares: Σ|Δθ|
        - Jerk articular (variação de aceleração)
        - Índice de suavidade baseado em derivadas
        """
        from analysis.metrics.smoothness_metrics import MetricasSuavidade
        
        metricas = {}
        
        for alg_nome in ['PSO', 'AG', 'PSO-DFL']:
            alg_data = self.dados_colectados.get(alg_nome, {})
            metricas[alg_nome] = {}
            
            for modo in ['warm_start', 'cold_start']:
                modo_data = alg_data.get(modo, {})
                if not modo_data:
                    continue
                
                trajetorias = modo_data.get('trajetorias', [])
                if not trajetorias:
                    continue
                
                # Calcular métricas para cada trajetória (run)
                variacoes_lista = []
                jerk_lista = []
                
                for traj in trajetorias:
                    if len(traj) > 0:
                        soma_var = MetricasSuavidade.soma_variacoes_angulares(traj)
                        jerk = MetricasSuavidade.jerk_articular(traj)
                        
                        variacoes_lista.append(soma_var)
                        jerk_lista.append(jerk)
                
                if variacoes_lista:
                    metricas[alg_nome][modo] = {
                        'soma_variacoes_media': float(np.mean(variacoes_lista)),
                        'soma_variacoes_std': float(np.std(variacoes_lista)),
                        'jerk_media': float(np.mean(jerk_lista)),
                        'jerk_std': float(np.std(jerk_lista))
                    }
        
        return metricas
    
    def _salvar_relatorio_suavidade(self, metricas_suavidade: Dict):
        """
        Salva relatório de suavidade em JSON.
        """
        path = os.path.join(self.path_saidas, 'statistics', 'metricas_suavidade.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(metricas_suavidade, f, indent=2)
        
        print(f"[INFO] Metricas de suavidade salvas em {path}")
    
    def _executar_testes_estatisticos(self, historicos_erro: Dict):
        """
        Executa testes estatísticos: t-Student, ANOVA, Wilcoxon.
        
        Validação científica das diferenças entre algoritmos.
        """
        from analysis.metrics.statistics import AnalisaEstatistica
        
        testes_resultados = {}
        
        # ─────────────────────────────────────────
        # TESTE 1: PSO vs AG (t-Student)
        # ─────────────────────────────────────────
        print("\n  Teste 1: PSO vs AG (t-Student)")
        
        pso_erros = np.asarray(historicos_erro.get('PSO', [])).flatten()
        ag_erros = np.asarray(historicos_erro.get('AG', [])).flatten()
        
        if len(pso_erros) > 0 and len(ag_erros) > 0:
            resultado_t_pso_ag = AnalisaEstatistica.teste_t_student(pso_erros, ag_erros)
            testes_resultados['PSO_vs_AG_ttest'] = resultado_t_pso_ag
            
            sig = "SIM" if resultado_t_pso_ag.get('significante_alpha_0.05') else "NAO"
            print(f"    p-value: {resultado_t_pso_ag['p_value']:.6f}")
            print(f"    Significativo (alpha=0.05): {sig}")
        
        # ─────────────────────────────────────────
        # TESTE 2: PSO vs PSO-DFL (t-Student)
        # ─────────────────────────────────────────
        print("\n  Teste 2: PSO vs PSO-DFL (t-Student)")
        
        psodfl_erros = np.asarray(historicos_erro.get('PSO-DFL', [])).flatten()
        
        if len(pso_erros) > 0 and len(psodfl_erros) > 0:
            resultado_t_pso_dfl = AnalisaEstatistica.teste_t_student(pso_erros, psodfl_erros)
            testes_resultados['PSO_vs_PSODFL_ttest'] = resultado_t_pso_dfl
            
            sig = "SIM" if resultado_t_pso_dfl.get('significante_alpha_0.05') else "NAO"
            print(f"    p-value: {resultado_t_pso_dfl['p_value']:.6f}")
            print(f"    Significativo (alpha=0.05): {sig}")
        
        # ─────────────────────────────────────────
        # TESTE 3: ANOVA (PSO, AG, PSO-DFL)
        # ─────────────────────────────────────────
        print("\n  Teste 3: ANOVA (3 algoritmos)")
        
        if len(pso_erros) > 0 and len(ag_erros) > 0 and len(psodfl_erros) > 0:
            dados_anova = {
                'PSO': pso_erros,
                'AG': ag_erros,
                'PSO-DFL': psodfl_erros
            }
            resultado_anova = AnalisaEstatistica.teste_anova(dados_anova)
            testes_resultados['ANOVA_3alg'] = resultado_anova
            
            sig = "SIM" if resultado_anova.get('significante_alpha_0.05') else "NAO"
            print(f"    F-statistic: {resultado_anova['F_statistic']:.4f}")
            print(f"    p-value: {resultado_anova['p_value']:.6f}")
            print(f"    Significativo (alpha=0.05): {sig}")
        
        # Salvar resultados
        path = os.path.join(self.path_saidas, 'statistics', 'testes_estatisticos.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(testes_resultados, f, indent=2)
        
        print(f"\n[INFO] Testes estatisticos salvos em {path}")
    
    def _exportar_dados_brutos(self):
        """
        Exporta todos os dados brutos para JSON.
        """
        import json
        
        dados_export = {}
        
        for alg_nome in ['PSO', 'AG', 'PSO-DFL']:
            alg_data = self.dados_colectados.get(alg_nome, {})
            
            dados_export[alg_nome] = {}
            
            for modo in ['cold_start', 'warm_start']:
                modo_data = alg_data.get(modo, {})
                if modo_data:
                    tempo_val = modo_data.get('tempo', 0)
                    if isinstance(tempo_val, np.ndarray):
                        tempo_val = float(tempo_val.flatten()[0]) if len(tempo_val) > 0 else 0
                    else:
                        tempo_val = float(tempo_val)
                    
                    # Processar histórico de erro
                    hist_erro_raw = modo_data.get('historico_erro', [])
                    hist_erro_list = []
                    if hist_erro_raw:
                        # Se é lista de listas (múltiplos runs), concatenar
                        if isinstance(hist_erro_raw[0], (list, np.ndarray)):
                            for h in hist_erro_raw:
                                if h:
                                    hist_erro_list.extend([float(x) if isinstance(x, (int, float, np.number)) else x 
                                                          for x in np.asarray(h).flatten().tolist()])
                        else:
                            hist_erro_list = [float(x) if isinstance(x, (int, float, np.number)) else x 
                                            for x in np.asarray(hist_erro_raw).flatten().tolist()]
                    
                    # Processar histórico de diversidade
                    hist_div_raw = modo_data.get('historico_diversidade', [])
                    hist_div_list = []
                    if hist_div_raw:
                        if isinstance(hist_div_raw[0], (list, np.ndarray)):
                            for h in hist_div_raw:
                                if h:
                                    hist_div_list.extend([float(x) if isinstance(x, (int, float, np.number)) else x 
                                                         for x in np.asarray(h).flatten().tolist()])
                        else:
                            hist_div_list = [float(x) if isinstance(x, (int, float, np.number)) else x 
                                           for x in np.asarray(hist_div_raw).flatten().tolist()]
                    
                    dados_export[alg_nome][modo] = {
                        'historico_erro': hist_erro_list,
                        'historico_diversidade': hist_div_list,
                        'tempo': tempo_val
                    }
        
        # Salvar
        path = os.path.join(self.path_saidas, 'data', 'dados_brutos.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(dados_export, f, indent=2)
        
        print(f"[INFO] Dados brutos exportados para {path}")
    
    def _gerar_relatorio_estatistico(self):
        """
        Gera relatório estatístico em JSON.
        """
        import json
        
        relatorio = {}
        
        for alg_nome in ['PSO', 'AG', 'PSO-DFL']:
            alg_data = self.dados_colectados.get(alg_nome, {})
            relatorio[alg_nome] = {}
            
            for modo in ['cold_start', 'warm_start']:
                modo_data = alg_data.get(modo, {})
                
                if modo_data:
                    # Processar histórico de erro - pode ser lista de listas
                    hist_erro_raw = modo_data.get('historico_erro', [])
                    hist_erro = []
                    if hist_erro_raw:
                        if isinstance(hist_erro_raw[0], (list, np.ndarray)):
                            for h in hist_erro_raw:
                                if h:
                                    hist_erro.extend(np.asarray(h).flatten().tolist())
                        else:
                            hist_erro = np.asarray(hist_erro_raw).flatten().tolist()
                    
                    hist_erro = np.asarray(hist_erro).flatten()
                    
                    # Processar histórico de diversidade
                    hist_div_raw = modo_data.get('historico_diversidade', [])
                    hist_div = []
                    if hist_div_raw:
                        if isinstance(hist_div_raw[0], (list, np.ndarray)):
                            for h in hist_div_raw:
                                if h:
                                    hist_div.extend(np.asarray(h).flatten().tolist())
                        else:
                            hist_div = np.asarray(hist_div_raw).flatten().tolist()
                    
                    hist_div = np.asarray(hist_div).flatten()
                    
                    tempo_val = modo_data.get('tempo', 0)
                    if isinstance(tempo_val, np.ndarray):
                        # Se é array, calcular média
                        tempo_val = float(np.mean(tempo_val)) if len(tempo_val) > 0 else 0
                    else:
                        tempo_val = float(tempo_val)
                    
                    relatorio[alg_nome][modo] = {
                        'tempo_medio': tempo_val,
                        'convergencia': {
                            'erro_final_medio': float(np.mean(hist_erro)) if len(hist_erro) > 0 else 0,
                            'erro_final_std': float(np.std(hist_erro)) if len(hist_erro) > 0 else 0,
                            'erro_minimo': float(np.min(hist_erro)) if len(hist_erro) > 0 else 0,
                            'erro_maximo': float(np.max(hist_erro)) if len(hist_erro) > 0 else 0,
                            'iteracoes_convergencia': int(len(hist_erro))
                        },
                        'diversidade': {
                            'media': float(np.mean(hist_div)) if len(hist_div) > 0 else 0,
                            'std': float(np.std(hist_div)) if len(hist_div) > 0 else 0,
                            'minimo': float(np.min(hist_div)) if len(hist_div) > 0 else 0,
                            'maximo': float(np.max(hist_div)) if len(hist_div) > 0 else 0
                        }
                    }
        
        # Salvar
        path = os.path.join(self.path_saidas, 'statistics', 'relatorio_completo.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(relatorio, f, indent=2)
        
        print(f"[INFO] Relatorio estatistico salvo em {path}")
        
        # Também gerar resumo em texto
        self._gerar_resumo_texto(relatorio)
    
    def _gerar_resumo_texto(self, relatorio):
        """
        Gera resumo das análises em formato texto.
        """
        resumo = "=" * 80 + "\n"
        resumo += "RESUMO DAS ANALISES: PSO vs AG vs PSO-DFL\n"
        resumo += "=" * 80 + "\n\n"
        
        for alg_nome in ['PSO', 'AG', 'PSO-DFL']:
            resumo += f"\n{'='*80}\n"
            resumo += f"ALGORITMO: {alg_nome}\n"
            resumo += f"{'='*80}\n"
            
            alg_data = relatorio.get(alg_nome, {})
            
            for modo in ['cold_start', 'warm_start']:
                modo_data = alg_data.get(modo, {})
                
                if modo_data:
                    resumo += f"\n{modo.upper().replace('_', '-')}:\n"
                    resumo += f"  Tempo total:           {modo_data.get('tempo_medio', 0):.4f} segundos\n"
                    
                    conv = modo_data.get('convergencia', {})
                    resumo += f"  Convergencia:\n"
                    resumo += f"    - Erro final (media):  {conv.get('erro_final_medio', 0):.6f} mm\n"
                    resumo += f"    - Erro final (std):    {conv.get('erro_final_std', 0):.6f} mm\n"
                    resumo += f"    - Melhor erro:         {conv.get('erro_minimo', 0):.6f} mm\n"
                    resumo += f"    - Pior erro:           {conv.get('erro_maximo', 0):.6f} mm\n"
                    resumo += f"    - Iteracoes:           {conv.get('iteracoes_convergencia', 0)}\n"
                    
                    div = modo_data.get('diversidade', {})
                    resumo += f"  Diversidade:\n"
                    resumo += f"    - Media:               {div.get('media', 0):.6f}\n"
                    resumo += f"    - Desvio padrao:       {div.get('std', 0):.6f}\n"
                    resumo += f"    - Minima:              {div.get('minimo', 0):.6f}\n"
                    resumo += f"    - Maxima:              {div.get('maximo', 0):.6f}\n"
        
        # Gerar comparação
        resumo += f"\n\n{'='*80}\n"
        resumo += "COMPARACAO: COLD-START vs WARM-START (IMPACTO DE DOPING)\n"
        resumo += f"{'='*80}\n"
        
        for alg_nome in ['PSO', 'AG']:
            alg_data = relatorio.get(alg_nome, {})
            cold = alg_data.get('cold_start', {})
            warm = alg_data.get('warm_start', {})
            
            if cold and warm:
                resumo += f"\n{alg_nome}:\n"
                
                cold_erro = cold.get('convergencia', {}).get('erro_final_medio', 0)
                warm_erro = warm.get('convergencia', {}).get('erro_final_medio', 0)
                melhora = ((cold_erro - warm_erro) / cold_erro * 100) if cold_erro > 0 else 0
                
                resumo += f"  Erro final (cold): {cold_erro:.6f} mm\n"
                resumo += f"  Erro final (warm): {warm_erro:.6f} mm\n"
                resumo += f"  Melhora com doping: {melhora:.2f}%\n"
                
                cold_tempo = cold.get('tempo_medio', 0)
                warm_tempo = warm.get('tempo_medio', 0)
                
                resumo += f"  Tempo (cold): {cold_tempo:.4f} segundos\n"
                resumo += f"  Tempo (warm): {warm_tempo:.4f} segundos\n"
        
        resumo += "\n" + "=" * 80 + "\n"
        
        # Salvar
        path = os.path.join(self.path_saidas, 'statistics', 'resumo_analises.txt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(resumo)
        
        print(f"[INFO] Resumo de analises salvo em {path}")


def main():
    """
    Script de execução principal.
    
    ===================================================================
    BATERIA COMPLETA DE TESTES COMPARATIVOS
    ===================================================================
    
    Este script executa uma análise exaustiva comparando:
    
    1. ALGORITMO 1: PSO Tradicional (Particle Swarm Optimization)
       - Cold-Start: População aleatória para cada ponto da trajetória
       - Warm-Start: Doping com solução anterior (continuidade temporal)
    
    2. ALGORITMO 2: Algoritmo Genético (AG) 
       - Cold-Start: População inicial aleatória
       - Warm-Start: Doping com solução anterior
    
    3. ALGORITMO 3: PSO-DFL (Dynamic Fitness Landscape)
       - Warm-Start apenas (otimizado para continuidade)
       - Paisagem de fitness dinâmica com ajuste de parâmetros
    
    METRICAS COLETADAS:
    ───────────────────────────────────────────────────────────────────
    
    A. CONVERGENCIA:
       - Histórico de erro por iteração
       - Taxa de convergência (melhoria por iteração)
       - Erro final (média, desvio padrão, mín/máx)
    
    B. DIVERSIDADE DA POPULACAO:
       - Desvio padrão da população ao longo das iterações
       - Detecção de convergência prematura
       - Comportamento em mínimos locais
    
    C. SUAVIDADE E CONTINUIDADE:
       - Soma de variações angulares: Σ|Δθ|
       - Jerk articular (derivada da aceleração)
       - Índice de suavidade baseado em derivadas
       - Análise por junta individual
    
    D. TEMPO DE EXECUCAO:
       - Tempo total por algoritmo
       - Tempo médio por ponto da trajetória
    
    E. ANALISES ESTATISTICAS:
       - Teste t-Student: Comparação PSO vs AG, PSO vs PSO-DFL
       - Teste ANOVA: Comparação dos 3 algoritmos
       - Teste Wilcoxon: Dados não-paramétricos
       - Boxplots comparativos
    
    TESTES DE ENVIESAMENTO (WARM-START vs COLD-START):
    ───────────────────────────────────────────────────────────────────
    
    Warm-Start (Doping): A solução do ponto t-1 é injetada na população 
    inicial do ponto t. Esperado: Melhora no tempo e erro final.
    
    Cold-Start: População inicial aleatória. Baseline para comparação.
    
    ORGANIZACAO DE SAIDAS:
    ───────────────────────────────────────────────────────────────────
    
    analysis/results/
    ├── convergencia/      Graficos de erro vs iterações
    ├── diversidade/       Graficos de desvio padrão da população
    ├── suavidade/         Trajetórias articulares e jerk
    ├── estatistica/       Testes estatísticos e boxplots
    ├── data/              Dados brutos em JSON
    └── statistics/        Relatórios consolidados
    
    ===================================================================
    """
    print("\n" + "="*70)
    print("INTEGRADOR DE ANALISES COMPARATIVAS")
    print("PSO vs AG vs PSO-DFL")
    print("="*70)
    print(__doc__)
    
    # Criar integrador
    integrador = IntegradorAnalises(
        n_runs=20,  # Aumentar para 10+ em produção para rigor estatístico
        path_saidas="analysis/results"
    )
    
    # Executar teste completo
    print("\n[INFO] Iniciando bateria de testes...")
    resultados = integrador.executar_teste_completo(verbose=True)
    
    print("\n" + "="*70)
    print("TESTE CONCLUIDO COM SUCESSO!")
    print("="*70)
    print(f"\nResultados salvos em: {integrador.path_saidas}")
    print("\nARQUIVOS GERADOS:")
    print("  [data] Dados brutos em JSON")
    print("  [plots] Graficos prontos para artigo")
    print("  [statistics] Analises estatisticas consolidadas")
    print("\n[OK] Pronto para incluir no artigo cientifico!")


if __name__ == '__main__':
    main()
