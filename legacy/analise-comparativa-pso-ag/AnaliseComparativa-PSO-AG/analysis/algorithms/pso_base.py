"""
Implementação do PSO (Particle Swarm Optimization) com e sem DFL
Baseado em scikit-opt com extensões para análise
"""

import numpy as np
import time
from typing import Callable, Tuple, List, Dict, Any
from sko.PSO import PSO


class PSO_Padrao:
    """
    PSO padrão com warm-start (doping).
    
    Parâmetros:
    - w (inércia): 0.9 inicialmente, decresce para 0.4 ao final
    - c1, c2 (coeficientes): 0.3 cada (favorece exploração)
    """
    
    def __init__(self, 
                 fitness_func: Callable,
                 dim: int = 6,
                 pop: int = 40,
                 max_iter: int = 150,
                 limites_inf: List[float] = None,
                 limites_sup: List[float] = None,
                 seed: int = 42):
        """
        Inicializa PSO.
        
        Args:
            fitness_func: Função de avaliação (minimização)
            dim: Número de dimensões (juntas do robô)
            pop: Tamanho da população (partículas)
            max_iter: Número de iterações
            limites_inf, limites_sup: Limites articulares
            seed: Semente aleatória para reprodutibilidade
        """
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop = pop
        self.max_iter = max_iter
        self.seed = seed
        
        self.limites_inf = limites_inf if limites_inf is not None else [-np.pi] * dim
        self.limites_sup = limites_sup if limites_sup is not None else [np.pi] * dim
        
        self.historico_melhor_fitness = []
        self.historico_diversidade = []
        self.historico_populacao = []
        
    def rodar_cold_start(self, verbose: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Executa PSO SEM doping (cold-start).
        
        Returns:
            (melhor_solucao, melhor_fitness, tempo_execucao)
        """
        np.random.seed(self.seed)
        
        t0 = time.time()
        
        pso = PSO(func=self.fitness_func,
                 dim=self.dim,
                 pop=self.pop,
                 max_iter=self.max_iter,
                 lb=self.limites_inf,
                 ub=self.limites_sup,
                 w=0.9,
                 c1=0.3,
                 c2=0.3)
        
        pso.run()
        
        tempo = time.time() - t0
        
        # Garantir que gbest_y é um float escalar
        gbest_y_scalar = float(np.asarray(pso.gbest_y).flatten()[0])
        
        return pso.gbest_x, gbest_y_scalar, tempo
    
    def rodar_warm_start(self, 
                        solucao_anterior: np.ndarray = None,
                        verbose: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Executa PSO COM doping (warm-start).
        
        Args:
            solucao_anterior: Solução do passo anterior para "aquecer" a população
            verbose: Imprime progresso
            
        Returns:
            (melhor_solucao, melhor_fitness, tempo_execucao)
        """
        np.random.seed(self.seed)
        
        t0 = time.time()
        
        # Cria PSO com warm-start customizado
        class PSO_WarmStart(PSO):
            def __init__(self, *args, solucao_anterior=None, limites_inf=None, 
                        limites_sup=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.solucao_anterior = solucao_anterior
                self.limites_inf = limites_inf
                self.limites_sup = limites_sup
                
            def initialization(self):
                """Override para inicializar com doping"""
                super().initialization()
                
                if self.solucao_anterior is not None:
                    # Planta solução anterior nas primeiras partículas
                    self.X[0] = np.clip(self.solucao_anterior, 
                                       self.limites_inf, self.limites_sup)
                    
                    # Variações pequenas ao redor da solução anterior
                    for j in range(1, min(6, self.pop)):
                        variacao = np.random.uniform(-0.05, 0.05, self.dim)
                        self.X[j] = np.clip(self.solucao_anterior + variacao, 
                                           self.limites_inf, self.limites_sup)
        
        pso = PSO_WarmStart(func=self.fitness_func,
                           dim=self.dim,
                           pop=self.pop,
                           max_iter=self.max_iter,
                           lb=self.limites_inf,
                           ub=self.limites_sup,
                           w=0.9,
                           c1=0.3,
                           c2=0.3,
                           solucao_anterior=solucao_anterior,
                           limites_inf=self.limites_inf,
                           limites_sup=self.limites_sup)
        
        pso.run()
        
        tempo = time.time() - t0
        
        # Garantir que gbest_y é um float escalar
        gbest_y_scalar = float(np.asarray(pso.gbest_y).flatten()[0])
        
        return pso.gbest_x, gbest_y_scalar, tempo


class PSO_DFL(PSO_Padrao):
    """
    PSO com Dynamic Fitness Landscape (Paisagem de Fitness Dinâmica).
    
    Ajusta parâmetros (w, c1, c2) ao longo das iterações para manter
    diversidade em problemas com fitness landscape dinâmico (trajetória).
    
    Conceito: No espaço de trajetória, o landscape não é fixo - se deforma
    a cada ponto, então a PSO deve ser adaptativa também.
    
    Referência: "Temporal Continuity" - trata a trajetória como UM ÚNICO problema
    cujo mapa de erro se deforma no tempo.
    """
    
    def __init__(self, 
                 fitness_func: Callable,
                 dim: int = 6,
                 pop: int = 40,
                 max_iter: int = 150,
                 limites_inf: List[float] = None,
                 limites_sup: List[float] = None,
                 seed: int = 42,
                 modo_adaptacao: str = 'exponencial'):
        """
        Args:
            modo_adaptacao: 'exponencial', 'linear' ou 'sigmoid'
        """
        super().__init__(fitness_func, dim, pop, max_iter, 
                        limites_inf, limites_sup, seed)
        
        self.modo_adaptacao = modo_adaptacao
    
    def _calcular_parametros_adaptativos(self, iteracao_atual: int) -> Tuple[float, float, float]:
        """
        Calcula w, c1, c2 dinamicamente baseado na iteração.
        
        Estratégia:
        - Início (exploração): w alto, c1 alto (autoconfiança)
        - Meio (transição): w diminui, c1 e c2 equilibram
        - Fim (exploração local): w baixo, c2 alto (confiança social)
        
        Returns:
            (w, c1, c2) - Parâmetros para esta iteração
        """
        t = iteracao_atual / self.max_iter  # Progresso normalizado [0, 1]
        
        if self.modo_adaptacao == 'exponencial':
            # Inércia decresce exponencialmente
            w = 0.9 * np.exp(-2.0 * t)
            w = max(w, 0.3)  # Mínimo de 0.3
            
            # Autoconfiança decresce, confiança social cresce
            c1 = 0.4 * (1.0 - t)
            c2 = 0.1 + 0.4 * t
            
        elif self.modo_adaptacao == 'linear':
            w = 0.9 - 0.6 * t  # De 0.9 para 0.3
            c1 = 0.3 + 0.2 * (1.0 - t)
            c2 = 0.3 + 0.2 * t
            
        else:  # sigmoid
            # Transição suave (mais natural para robótica)
            sigmoid = 1.0 / (1.0 + np.exp(-6.0 * (t - 0.5)))
            w = 0.3 + 0.6 * (1.0 - sigmoid)
            c1 = 0.2 + 0.2 * (1.0 - sigmoid)
            c2 = 0.2 + 0.2 * sigmoid
        
        return w, c1, c2
    
    def rodar_warm_start(self, 
                        solucao_anterior: np.ndarray = None,
                        verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executa PSO-DFL COM doping (warm-start).
        
        O landscape dinâmico é simulado através da mudança de parâmetros,
        mantendo a diversidade mesmo conforme o paisagem "se deforma".
        
        Returns:
            (melhor_solucao, melhor_fitness, tempo_execucao)
        """
        np.random.seed(self.seed)
        
        t0 = time.time()
        
        # Inicializa estruturas de PSO
        X = np.random.uniform(self.limites_inf, self.limites_sup, (self.pop, self.dim))
        V = np.random.uniform(-1, 1, (self.pop, self.dim)) * 0.1
        
        # === DOPING (Warm-Start) ===
        if solucao_anterior is not None:
            X[0] = np.clip(solucao_anterior, self.limites_inf, self.limites_sup)
            for j in range(1, min(6, self.pop)):
                variacao = np.random.uniform(-0.05, 0.05, self.dim)
                X[j] = np.clip(solucao_anterior + variacao, 
                              self.limites_inf, self.limites_sup)
        # =========================
        
        # Melhor pessoal e global
        pbest_x = X.copy()
        pbest_y = np.array([self.fitness_func(x) for x in X])
        
        gbest_idx = np.argmin(pbest_y)
        gbest_x = X[gbest_idx].copy()
        gbest_y = pbest_y[gbest_idx]
        
        # Históricos
        self.historico_melhor_fitness = [float(gbest_y)]
        self.historico_diversidade = []
        self.historico_populacao = [X.copy()]
        
        # Loop principal com parâmetros adaptativos
        for i in range(self.max_iter):
            # Parâmetros dinâmicos para esta iteração
            w, c1, c2 = self._calcular_parametros_adaptativos(i)
            
            # Atualiza velocidades e posições
            r1 = np.random.rand(self.pop, self.dim)
            r2 = np.random.rand(self.pop, self.dim)
            
            V = (w * V + 
                c1 * r1 * (pbest_x - X) + 
                c2 * r2 * (gbest_x - X))
            
            X = X + V
            X = np.clip(X, self.limites_inf, self.limites_sup)
            
            # Avalia população
            Y = np.array([self.fitness_func(x) for x in X])
            
            # Atualiza melhor pessoal
            melhorou = Y < pbest_y
            pbest_x[melhorou] = X[melhorou]
            pbest_y[melhorou] = Y[melhorou]
            
            # Atualiza melhor global
            if np.min(Y) < gbest_y:
                gbest_y = np.min(Y)
                gbest_x = X[np.argmin(Y)].copy()
            
            # Armazena histórico
            self.historico_melhor_fitness.append(float(gbest_y))
            self.historico_populacao.append(X.copy())
            
            diversidade = np.std(X, axis=0).mean()
            self.historico_diversidade.append(float(diversidade))
            
            if verbose:
                print(f"  Iteração {i+1}/{self.max_iter} | Fitness: {gbest_y:.6f} | "
                      f"w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")
        
        tempo = time.time() - t0
        
        return gbest_x, np.array(gbest_y), tempo
