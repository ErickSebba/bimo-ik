"""
Configurador de Experimentos
Define os setups de teste: Cold-Start, Warm-Start, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable


class ConfiguradorExperimentos:
    """
    Classe para configurar e executar diferentes tipos de experimentos.
    
    Tipos:
    1. Cold-Start: Sem doping, solução anterior não influencia
    2. Warm-Start: Com doping, solução anterior "aquece" população
    3. Comparação de Métodos: PSO vs AG vs PSO-DFL
    4. Análise de Convergência: Com/sem doping
    """
    
    def __init__(self,
                 fitness_func: Callable,
                 trajetoria: np.ndarray,
                 limites_inf: np.ndarray = None,
                 limites_sup: np.ndarray = None,
                 n_runs: int = 10,
                 seed_base: int = 42):
        """
        Args:
            fitness_func: Função que retorna fitness dado alvo
            trajetoria: (n_pontos, 3) - Trajetória de alvo XYZ
            limites_inf, limites_sup: Limites articulares
            n_runs: Número de execuções independentes
            seed_base: Semente-base (cada run usa seed_base + run_id)
        """
        self.fitness_func = fitness_func
        self.trajetoria = np.array(trajetoria)
        self.n_pontos = len(trajetoria)
        self.limites_inf = limites_inf
        self.limites_sup = limites_sup
        self.n_runs = n_runs
        self.seed_base = seed_base
    
    def executar_teste_cold_vs_warm_pso(self,
                                        algoritmo_pso: Any,
                                        verbose: bool = False) -> Dict:
        """
        Compara PSO Cold-Start vs Warm-Start.
        
        Executa múltiplas vezes e coleta estatísticas.
        
        Returns:
            Dict com resultados separados de cold e warm
        """
        resultados_cold = {
            'tempo_por_ponto': [],
            'fitness_por_ponto': [],
            'trajetoria_completa': [],
            'tempo_total': 0
        }
        
        resultados_warm = {
            'tempo_por_ponto': [],
            'fitness_por_ponto': [],
            'trajetoria_completa': [],
            'tempo_total': 0
        }
        
        for run in range(self.n_runs):
            seed = self.seed_base + run
            
            # === COLD-START ===
            q_anterior_cold = np.zeros(6)
            tempo_total_cold = 0
            fitness_total_cold = []
            trajetoria_cold = []
            
            for i, alvo in enumerate(self.trajetoria):
                # Cria fitness para este ponto
                fitness_alvo = self.fitness_func(alvo, q_anterior_cold)
                
                # Cria instância PSO
                pso_cold = algoritmo_pso(fitness_alvo, seed=seed)
                
                # Executa sem doping
                q_opt, fit, tempo = pso_cold.rodar_cold_start(verbose=False)
                
                fitness_total_cold.append(float(fit))
                tempo_total_cold += tempo
                trajetoria_cold.append(q_opt)
                
                q_anterior_cold = q_opt
            
            resultados_cold['tempo_por_ponto'].append(np.array(fitness_total_cold))
            resultados_cold['fitness_por_ponto'].append(np.array(fitness_total_cold))
            resultados_cold['trajetoria_completa'].append(np.array(trajetoria_cold))
            resultados_cold['tempo_total'] += tempo_total_cold
            
            # === WARM-START ===
            q_anterior_warm = np.zeros(6)
            tempo_total_warm = 0
            fitness_total_warm = []
            trajetoria_warm = []
            
            for i, alvo in enumerate(self.trajetoria):
                fitness_alvo = self.fitness_func(alvo, q_anterior_warm)
                
                pso_warm = algoritmo_pso(fitness_alvo, seed=seed)
                
                # Executa COM doping
                q_opt, fit, tempo = pso_warm.rodar_warm_start(
                    solucao_anterior=q_anterior_warm,
                    verbose=False
                )
                
                fitness_total_warm.append(float(fit))
                tempo_total_warm += tempo
                trajetoria_warm.append(q_opt)
                
                q_anterior_warm = q_opt
            
            resultados_warm['tempo_por_ponto'].append(np.array(fitness_total_warm))
            resultados_warm['fitness_por_ponto'].append(np.array(fitness_total_warm))
            resultados_warm['trajetoria_completa'].append(np.array(trajetoria_warm))
            resultados_warm['tempo_total'] += tempo_total_warm
            
            if verbose:
                print(f"  Run {run+1}/{self.n_runs} completo")
        
        return {
            'cold_start': resultados_cold,
            'warm_start': resultados_warm,
            'n_runs': self.n_runs
        }
    
    def executar_comparacao_tres_algoritmos(self,
                                           algoritmo_pso: Any,
                                           algoritmo_ag: Any,
                                           algoritmo_pso_dfl: Any,
                                           verbose: bool = False) -> Dict:
        """
        Compara PSO, AG e PSO-DFL simultaneamente com warm-start.
        
        Returns:
            Dict organizado por algoritmo
        """
        
        resultados = {
            'PSO': {'fitness': [], 'tempo': [], 'diversidade': []},
            'AG': {'fitness': [], 'tempo': [], 'diversidade': []},
            'PSO-DFL': {'fitness': [], 'tempo': [], 'diversidade': []}
        }
        
        for run in range(self.n_runs):
            seed = self.seed_base + run
            
            # === PSO Padrão ===
            q_ant_pso = np.zeros(6)
            fit_pso = []
            tempo_pso = 0
            
            for alvo in self.trajetoria:
                fitness_alvo = self.fitness_func(alvo, q_ant_pso)
                pso = algoritmo_pso(fitness_alvo, seed=seed)
                q_opt, fit_val, t = pso.rodar_warm_start(q_ant_pso)
                fit_pso.append(float(fit_val))
                tempo_pso += t
                q_ant_pso = q_opt
            
            resultados['PSO']['fitness'].append(fit_pso)
            resultados['PSO']['tempo'].append(tempo_pso)
            
            # === AG ===
            q_ant_ag = np.zeros(6)
            fit_ag = []
            tempo_ag = 0
            
            for alvo in self.trajetoria:
                fitness_alvo = self.fitness_func(alvo, q_ant_ag)
                ag = algoritmo_ag(fitness_alvo, seed=seed)
                q_opt, fit_val, t = ag.rodar_warm_start(q_ant_ag)
                fit_ag.append(float(fit_val))
                tempo_ag += t
                q_ant_ag = q_opt
            
            resultados['AG']['fitness'].append(fit_ag)
            resultados['AG']['tempo'].append(tempo_ag)
            
            # === PSO-DFL ===
            q_ant_dfl = np.zeros(6)
            fit_dfl = []
            tempo_dfl = 0
            
            for alvo in self.trajetoria:
                fitness_alvo = self.fitness_func(alvo, q_ant_dfl)
                pso_dfl = algoritmo_pso_dfl(fitness_alvo, seed=seed)
                q_opt, fit_val, t = pso_dfl.rodar_warm_start(q_ant_dfl)
                fit_dfl.append(float(fit_val))
                tempo_dfl += t
                q_ant_dfl = q_opt
            
            resultados['PSO-DFL']['fitness'].append(fit_dfl)
            resultados['PSO-DFL']['tempo'].append(tempo_dfl)
            
            if verbose:
                print(f"  Run {run+1}/{self.n_runs} | "
                      f"PSO: {tempo_pso:.2f}s, AG: {tempo_ag:.2f}s, "
                      f"PSO-DFL: {tempo_dfl:.2f}s")
        
        return resultados
