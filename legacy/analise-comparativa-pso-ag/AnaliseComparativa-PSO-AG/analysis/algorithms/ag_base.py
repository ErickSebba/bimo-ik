"""
Implementação do Algoritmo Genético (AG) com doping
Baseado em Farnese et al. - Seleção, Crossover Aritmético, Mutação Gaussiana
"""

import numpy as np
import time
from typing import Callable, Tuple, List, Dict, Any


class AG_Padrao:
    """
    Algoritmo Genético com crossover aritmético e mutação gaussiana.
    
    Configuração simétrica ao PSO para comparação justa.
    
    Parâmetros:
    - Taxa de Crossover: 0.7 (Farnese et al.)
    - Taxa de Mutação: 0.15 por gene (Farnese et al.)
    - Seleção: Roleta (proporcional ao fitness inverso)
    """
    
    def __init__(self,
                 fitness_func: Callable,
                 dim: int = 6,
                 pop: int = 40,
                 n_gerações: int = 150,
                 limites_inf: List[float] = None,
                 limites_sup: List[float] = None,
                 taxa_crossover: float = 0.7,
                 taxa_mutacao: float = 0.15,
                 elite_ratio: float = 0.1,
                 seed: int = 42):
        """
        Inicializa AG.
        
        Args:
            fitness_func: Função de avaliação (minimização)
            dim: Número de dimensões (juntas)
            pop: Tamanho da população
            n_gerações: Número de gerações
            limites_inf, limites_sup: Limites articulares
            taxa_crossover: Probabilidade de crossover
            taxa_mutacao: Probabilidade de mutação por gene
            elite_ratio: Proporção da população que é elitista (cópia para próxima geração)
            seed: Semente aleatória
        """
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop = pop
        self.n_gerações = n_gerações
        self.seed = seed
        
        self.limites_inf = np.array(limites_inf if limites_inf is not None else [-np.pi] * dim)
        self.limites_sup = np.array(limites_sup if limites_sup is not None else [np.pi] * dim)
        
        self.taxa_crossover = taxa_crossover
        self.taxa_mutacao = taxa_mutacao
        self.elite_ratio = elite_ratio
        self.n_elite = max(1, int(pop * elite_ratio))
        
        self.historico_melhor_fitness = []
        self.historico_diversidade = []
        self.historico_populacao = []
    
    def _inicializar_populacao(self, solucao_anterior: np.ndarray = None) -> np.ndarray:
        """
        Cria população inicial aleatória com opção de doping.
        
        Returns:
            (pop, dim) - População inicial
        """
        populacao = np.random.uniform(self.limites_inf, self.limites_sup, 
                                     (self.pop, self.dim))
        
        # === DOPING (Warm-Start) ===
        if solucao_anterior is not None:
            populacao[0] = np.clip(solucao_anterior, self.limites_inf, self.limites_sup)
            
            for j in range(1, min(6, self.pop)):
                variacao = np.random.uniform(-0.05, 0.05, self.dim)
                populacao[j] = np.clip(solucao_anterior + variacao,
                                      self.limites_inf, self.limites_sup)
        # =========================
        
        return populacao
    
    def _avaliar(self, populacao: np.ndarray) -> np.ndarray:
        """
        Avalia fitness de toda a população.
        
        Returns:
            (pop,) - Fitness de cada indivíduo
        """
        return np.array([self.fitness_func(ind) for ind in populacao])
    
    def _selecao_roleta(self, populacao: np.ndarray, 
                       fitness: np.ndarray,
                       n_selecionados: int) -> np.ndarray:
        """
        Seleção por roleta (proporcional ao fitness inverso).
        
        Indivíduos com melhor fitness (menor valor) têm maior probabilidade.
        
        Returns:
            (n_selecionados, dim) - Indivíduos selecionados
        """
        # Fitness inverso (para minimização: 1/f + offset para evitar divisão por zero)
        fitness_positivo = np.abs(fitness) + 1e-10
        aptidao = 1.0 / fitness_positivo
        
        # Normalizar probabilidades
        prob = aptidao / np.sum(aptidao)
        
        # Selecionar índices baseado em probabilidade
        idx = np.random.choice(len(populacao), size=n_selecionados, 
                              replace=True, p=prob)
        
        return populacao[idx]
    
    def _crossover_aritmetico(self, pai1: np.ndarray, pai2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crossover aritmético (blend).
        
        filho1 = α*pai1 + (1-α)*pai2
        filho2 = (1-α)*pai1 + α*pai2
        
        Onde α é aleatório em [0, 1].
        
        Melhor para espaços contínuos (Farnese et al. AG4).
        
        Returns:
            (filho1, filho2) - Dois filhos gerados
        """
        alpha = np.random.rand()
        
        filho1 = alpha * pai1 + (1 - alpha) * pai2
        filho2 = (1 - alpha) * pai1 + alpha * pai2
        
        # Clipar limites
        filho1 = np.clip(filho1, self.limites_inf, self.limites_sup)
        filho2 = np.clip(filho2, self.limites_inf, self.limites_sup)
        
        return filho1, filho2
    
    def _mutacao_gaussiana(self, individuo: np.ndarray) -> np.ndarray:
        """
        Mutação Gaussiana (gauss com desvio padrão adaptativo).
        
        Para cada gene, com probabilidade taxa_mutacao, aplicar:
        novo_gene = gene + N(0, sigma)
        
        onde sigma é uma fração do range disponível.
        
        Returns:
            individuo - Mutante
        """
        mutante = individuo.copy()
        
        # Calcula sigma como 1% do range disponível
        range_genes = self.limites_sup - self.limites_inf
        sigma = 0.01 * range_genes
        
        # Aplica mutação por gene
        for i in range(self.dim):
            if np.random.rand() < self.taxa_mutacao:
                mutante[i] += np.random.normal(0, sigma[i])
                mutante[i] = np.clip(mutante[i], self.limites_inf[i], self.limites_sup[i])
        
        return mutante
    
    def _criar_proxima_geracao(self, populacao: np.ndarray,
                              fitness: np.ndarray) -> np.ndarray:
        """
        Cria próxima geração via seleção, crossover e mutação.
        
        Implementa ELITISMO: os melhores n_elite indivíduos são copiados.
        
        Returns:
            nova_populacao - Próxima geração
        """
        nova_pop = []
        
        # === ELITISMO ===
        # Copia os melhores indivíduos
        idx_elite = np.argsort(fitness)[:self.n_elite]
        nova_pop.extend(populacao[idx_elite])
        
        # === REPRODUÇÃO ===
        # Preenche o resto da população com filhos
        while len(nova_pop) < self.pop:
            # Seleciona 2 pais
            pai1 = self._selecao_roleta(populacao, fitness, 1)[0]
            pai2 = self._selecao_roleta(populacao, fitness, 1)[0]
            
            # Crossover
            if np.random.rand() < self.taxa_crossover:
                filho1, filho2 = self._crossover_aritmetico(pai1, pai2)
            else:
                filho1, filho2 = pai1.copy(), pai2.copy()
            
            # Mutação
            filho1 = self._mutacao_gaussiana(filho1)
            filho2 = self._mutacao_gaussiana(filho2)
            
            # Adiciona filhos (controla tamanho)
            nova_pop.append(filho1)
            if len(nova_pop) < self.pop:
                nova_pop.append(filho2)
        
        return np.array(nova_pop[:self.pop])
    
    def rodar_cold_start(self, verbose: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Executa AG SEM doping (cold-start).
        
        Returns:
            (melhor_solucao, melhor_fitness, tempo_execucao)
        """
        np.random.seed(self.seed)
        
        t0 = time.time()
        
        populacao = self._inicializar_populacao(None)
        
        melhor_global = None
        melhor_fitness_global = float('inf')
        
        for geracao in range(self.n_gerações):
            # Avalia população
            fitness = self._avaliar(populacao)
            
            # Atualiza melhor
            idx_melhor = np.argmin(fitness)
            if fitness[idx_melhor] < melhor_fitness_global:
                melhor_fitness_global = fitness[idx_melhor]
                melhor_global = populacao[idx_melhor].copy()
            
            # Cria próxima geração
            populacao = self._criar_proxima_geracao(populacao, fitness)
        
        tempo = time.time() - t0
        
        return melhor_global, melhor_fitness_global, tempo
    
    def rodar_warm_start(self, 
                        solucao_anterior: np.ndarray = None,
                        verbose: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Executa AG COM doping (warm-start).
        
        Args:
            solucao_anterior: Solução do passo anterior para "aquecer" a população
            verbose: Imprime progresso
            
        Returns:
            (melhor_solucao, melhor_fitness, tempo_execucao)
        """
        np.random.seed(self.seed)
        
        t0 = time.time()
        
        # Inicializa com doping
        populacao = self._inicializar_populacao(solucao_anterior)
        
        # Históricos
        melhor_global = None
        melhor_fitness_global = float('inf')
        
        self.historico_melhor_fitness = []
        self.historico_diversidade = []
        self.historico_populacao = []
        
        for geracao in range(self.n_gerações):
            # Avalia população
            fitness = self._avaliar(populacao)
            
            # Atualiza melhor
            idx_melhor = np.argmin(fitness)
            if fitness[idx_melhor] < melhor_fitness_global:
                melhor_fitness_global = fitness[idx_melhor]
                melhor_global = populacao[idx_melhor].copy()
            
            # Armazena histórico
            self.historico_melhor_fitness.append(float(melhor_fitness_global))
            self.historico_populacao.append(populacao.copy())
            
            diversidade = np.std(populacao, axis=0).mean()
            self.historico_diversidade.append(float(diversidade))
            
            if verbose:
                print(f"  Geração {geracao+1}/{self.n_gerações} | "
                      f"Fitness: {melhor_fitness_global:.6f}")
            
            # Cria próxima geração
            populacao = self._criar_proxima_geracao(populacao, fitness)
        
        tempo = time.time() - t0
        
        return melhor_global, melhor_fitness_global, tempo
