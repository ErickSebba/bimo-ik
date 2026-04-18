"""
SCRIPT PRINCIPAL DE ANÁLISES COMPARATIVAS
Executa todos os testes: PSO vs AG vs PSO-DFL
Gera relatórios e visualizações para artigo científico

Organização de saídas:
  analysis/results/
    ├── data/           (dados brutos em JSON/NPZ)
    ├── statistics/     (resumos estatísticos)
    └── plots/          (gráficos de alta qualidade)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import json
import time
from typing import Dict, List

# Imports dos módulos criados
from analysis.metrics import (
    MetricasConvergencia,
    MetricasDiversidade,
    MetricasSuavidade,
    AnalisaEstatistica
)

# Importar do arquivo principal (erick_dan.py)
# Ajustar imports conforme necessário
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class OrquestradorAnalises:
    """
    Classe que coordena TODOS os testes e gera relatórios.
    """
    
    def __init__(self, 
                 path_base: str = "analysis/results",
                 n_runs: int = 10,
                 n_pontos_trajetoria: int = 30):
        """
        Args:
            path_base: Caminho para salvar resultados
            n_runs: Número de execuções independentes
            n_pontos_trajetoria: Pontos na trajetória de teste
        """
        self.path_base = path_base
        self.n_runs = n_runs
        self.n_pontos = n_pontos_trajetoria
        
        # Criar subpastas necessárias
        os.makedirs(os.path.join(path_base, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(path_base, 'statistics'), exist_ok=True)
        os.makedirs(os.path.join(path_base, 'data'), exist_ok=True)
        
        print(f"[INFO] Estrutura de diretórios criada em {path_base}")
    
    # ──────────────────────────────────────────────────────────────
    # TESTE 1: PRESERVAÇÃO DE DIVERSIDADE
    # ──────────────────────────────────────────────────────────────
    
    def analisar_preservacao_diversidade(self, 
                                        resultados_algoritmos: Dict) -> Dict:
        """
        Analisa desvio padrão da população ao longo das iterações.
        
        Compara PSO, AG e PSO-DFL.
        
        Args:
            resultados_algoritmos: Dict com históricos de população
                {
                    'PSO': [populacao_iter1, populacao_iter2, ...],
                    'AG': [...],
                    'PSO-DFL': [...]
                }
        """
        relatorios = {}
        
        for nome_alg, historico_pop in resultados_algoritmos.items():
            # Retirar média de múltiplos runs
            # Assumir historico_pop é lista de (n_iteracoes, populacoes)
            
            relatorio = MetricasDiversidade.relatorio_diversidade(
                historico_pop,
                titulo_algoritmo=nome_alg
            )
            relatorios[nome_alg] = relatorio
        
        # Salvar relatório
        caminho_rel = os.path.join(self.path_base, 'statistics', 
                                   'diversidade_preservacao.json')
        with open(caminho_rel, 'w') as f:
            json.dump(relatorios, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Relatório de diversidade salvo em {caminho_rel}")
        
        return relatorios
    
    # ──────────────────────────────────────────────────────────────
    # TESTE 2: IMPACTO DO DOPING (WARM-START vs COLD-START)
    # ──────────────────────────────────────────────────────────────
    
    def analisar_impacto_doping(self,
                               resultados_cold_warm: Dict) -> Dict:
        """
        Analisa como cada algoritmo reage ao doping.
        
        Compara métricas de Cold-Start vs Warm-Start.
        
        Args:
            resultados_cold_warm: Dict
                {
                    'PSO': {
                        'cold_start': { fitness: [...], tempo: [...] },
                        'warm_start': { fitness: [...], tempo: [...] }
                    },
                    'AG': {...},
                    ...
                }
        """
        resumo_doping = {}
        
        for nome_alg, dados in resultados_cold_warm.items():
            cold = np.array(dados['cold_start']['fitness'])  # (n_runs, n_pontos)
            warm = np.array(dados['warm_start']['fitness'])  # (n_runs, n_pontos)
            
            tempo_cold = np.array(dados['cold_start']['tempo'])  # (n_runs,)
            tempo_warm = np.array(dados['warm_start']['tempo'])  # (n_runs,)
            
            # Cálculos
            melhoria_tempo = (tempo_cold.mean() - tempo_warm.mean()) / tempo_cold.mean() * 100
            melhoria_erro = (cold.mean() - warm.mean()) / cold.mean() * 100
            
            resumo_doping[nome_alg] = {
                'tempo_cold_avg': float(tempo_cold.mean()),
                'tempo_warm_avg': float(tempo_warm.mean()),
                'melhoria_tempo_pct': float(melhoria_tempo),
                'fitness_cold_avg': float(cold.mean()),
                'fitness_warm_avg': float(warm.mean()),
                'melhoria_fitness_pct': float(melhoria_erro),
                'teste_t_tempo': AnalisaEstatistica.teste_t_student(tempo_cold, tempo_warm),
                'teste_t_fitness': AnalisaEstatistica.teste_t_student(
                    cold.flatten(), warm.flatten()
                )
            }
        
        # Salvar
        caminho_rel = os.path.join(self.path_base, 'statistics', 'impacto_doping.json')
        with open(caminho_rel, 'w') as f:
            json.dump(resumo_doping, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Análise de doping salva em {caminho_rel}")
        
        return resumo_doping
    
    # ──────────────────────────────────────────────────────────────
    # TESTE 3: MÉTRICAS DE SUAVIDADE
    # ──────────────────────────────────────────────────────────────
    
    def analisar_suavidade_trajetorias(self,
                                       trajetorias_por_alg: Dict[str, np.ndarray]) -> Dict:
        """
        Calcula suavidade de cada trajetória gerada.
        
        Args:
            trajetorias_por_alg: Dict
                {
                    'PSO': (n_runs, n_pontos, 6),
                    'AG': (n_runs, n_pontos, 6),
                    'PSO-DFL': (n_runs, n_pontos, 6)
                }
        """
        relatorios = {}
        
        for nome_alg, trajetorias in trajetorias_por_alg.items():
            metricas_por_run = []
            
            for run_idx in range(trajetorias.shape[0]):
                traj = trajetorias[run_idx]
                metrica = MetricasSuavidade.relatorio_suavidade(traj, nome_alg)
                metricas_por_run.append(metrica)
            
            # Agregação: média e desvio dos runs
            relatorio_agregado = {
                'algoritmo': nome_alg,
                'soma_variacoes_totais': np.mean([m['soma_variacoes_totais'] for m in metricas_por_run]),
                'jerk_articular': np.mean([m['jerk_articular'] for m in metricas_por_run]),
                'indice_suavidade': np.mean([m['indice_suavidade'] for m in metricas_por_run]),
                'descontinuidade': np.mean([m['descontinuidade'] for m in metricas_por_run]),
                'energia_mecanica': np.mean([m['energia_mecanica'] for m in metricas_por_run]),
                'desvios': {
                    'soma_variacoes_totais': np.std([m['soma_variacoes_totais'] for m in metricas_por_run]),
                    'jerk_articular': np.std([m['jerk_articular'] for m in metricas_por_run]),
                    'indice_suavidade': np.std([m['indice_suavidade'] for m in metricas_por_run]),
                    'descontinuidade': np.std([m['descontinuidade'] for m in metricas_por_run]),
                    'energia_mecanica': np.std([m['energia_mecanica'] for m in metricas_por_run])
                }
            }
            
            relatorios[nome_alg] = relatorio_agregado
        
        # Salvar
        caminho_rel = os.path.join(self.path_base, 'statistics', 'suavidade_trajetorias.json')
        with open(caminho_rel, 'w') as f:
            json.dump(relatorios, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Análise de suavidade salva em {caminho_rel}")
        
        return relatorios
    
    # ──────────────────────────────────────────────────────────────
    # TESTE 4: MÉTRICAS DE CONVERGÊNCIA
    # ──────────────────────────────────────────────────────────────
    
    def analisar_convergencia(self,
                             historico_erro_por_alg: Dict[str, List[np.ndarray]]) -> Dict:
        """
        Analisa velocidade e suavidade da convergência.
        
        Args:
            historico_erro_por_alg: Dict
                {
                    'PSO': [hist_err_run1, hist_err_run2, ...],
                    'AG': [...],
                    'PSO-DFL': [...]
                }
        """
        relatorios = {}
        
        for nome_alg, historicos in historico_erro_por_alg.items():
            # Usar primeira iteração de cada run (ou média)
            historico_medio = np.mean(historicos, axis=0)
            
            relatorio = MetricasConvergencia.relatorio_convergencia(
                historico_medio,
                titulo_algoritmo=nome_alg
            )
            relatorios[nome_alg] = relatorio
        
        # Salvar
        caminho_rel = os.path.join(self.path_base, 'statistics', 'convergencia.json')
        with open(caminho_rel, 'w') as f:
            json.dump(relatorios, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Análise de convergência salva em {caminho_rel}")
        
        return relatorios
    
    # ──────────────────────────────────────────────────────────────
    # TESTE 5: ANÁLISES ESTATÍSTICAS
    # ──────────────────────────────────────────────────────────────
    
    def gerar_analises_estatisticas(self,
                                   resultados_por_alg: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """
        Testes estatísticos: t-Student, ANOVA, Wilcoxon, etc.
        
        Args:
            resultados_por_alg: Dict
                {
                    'PSO': { 'erro_final': [...], 'jerk': [...], ... },
                    'AG': { ... },
                    'PSO-DFL': { ... }
                }
        """
        resumo = AnalisaEstatistica.resumo_estatistico_completo(resultados_por_alg)
        
        # Salvar
        caminho_rel = os.path.join(self.path_base, 'statistics', 'resumo_estatistico.json')
        AnalisaEstatistica.salvar_resumo_json(resumo, caminho_rel)
        
        print(f"[INFO] Resumo estatístico salvo em {caminho_rel}")
        
        return resumo
    
    # ──────────────────────────────────────────────────────────────
    # VISUALIZAÇÕES
    # ──────────────────────────────────────────────────────────────
    
    def plotar_convergencia_comparativa(self,
                                       historico_erro_por_alg: Dict[str, List[np.ndarray]]):
        """
        Plota convergência de PSO, AG e PSO-DFL sobrepostas.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = {'PSO': '#1f77b4', 'AG': '#ff7f0e', 'PSO-DFL': '#2ca02c'}
        
        for nome_alg, historicos in historico_erro_por_alg.items():
            # Converte para array, lidando com tamanhos diferentes
            # Pega apenas o primeiro elemento se for lista de listas
            historicos_flat = []
            for h in historicos:
                if isinstance(h, list):
                    # Se for lista, pega apenas o primeiro valor (representativo)
                    if len(h) > 0:
                        historicos_flat.append(h[0] if isinstance(h[0], (int, float)) else np.mean(h[0]))
                else:
                    historicos_flat.append(float(h))
            
            historicos_array = np.array(historicos_flat)
            
            # Verifica se temos dados válidos
            if len(historicos_array) == 0 or np.all(np.isnan(historicos_array)):
                continue
            
            # Cria array simples sem dimensões extras
            media = np.mean(historicos_array)
            desvio = np.std(historicos_array)
            
            # Para visualização, usa índice simples
            x = np.arange(len(historicos_array))
            y_values = historicos_array
            
            # Subplot 1: Convergência
            ax1.plot(x, y_values, 'o-', label=nome_alg, color=colors.get(nome_alg, 'gray'), 
                    linewidth=2, markersize=5, alpha=0.7)
            ax1.axhline(media, color=colors.get(nome_alg, 'gray'), linestyle='--', alpha=0.5)
            
            # Subplot 2: Log scale
            y_log = np.maximum(y_values, 1e-6)  # Evita log(0)
            ax2.semilogy(x, y_log, 'o-', label=nome_alg, color=colors.get(nome_alg, 'gray'), 
                        linewidth=2, markersize=5, alpha=0.7)
        
        ax1.set_xlabel('Run / Amostra', fontsize=11)
        ax1.set_ylabel('Fitness (erro em mm)', fontsize=11)
        ax1.set_title('Convergência Comparativa (Escala Linear)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Run / Amostra', fontsize=11)
        ax2.set_ylabel('Fitness (erro em mm, log scale)', fontsize=11)
        ax2.set_title('Convergência Comparativa (Escala Logarítmica)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        path = os.path.join(self.path_base, 'plots', '01_convergencia_comparativa.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Gráfico de convergência salvo em {path}")
    
    def plotar_diversidade_comparativa(self,
                                      resultados_diversidade: Dict):
        """
        Plota histórico de diversidade.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'PSO': '#1f77b4', 'AG': '#ff7f0e', 'PSO-DFL': '#2ca02c'}
        
        for nome_alg, historico in resultados_diversidade.items():
            # Tratar tanto listas simples como estruturas aninhadas
            if isinstance(historico, dict):
                historico = historico.get('diversidade_historico', [])
            
            # Converter para numpy array e lidar com dados vazios
            if historico:
                historico_array = np.asarray(historico).flatten()
                if len(historico_array) > 0:
                    ax.plot(historico_array, label=nome_alg, 
                           color=colors.get(nome_alg, 'gray'), linewidth=2)
        
        ax.set_xlabel('Iteração', fontsize=11)
        ax.set_ylabel('Desvio Padrão da População', fontsize=11)
        ax.set_title('Preservação de Diversidade ao Longo das Iterações', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.path_base, 'plots', '02_diversidade_preservacao.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Gráfico de diversidade salvo em {path}")
    
    def plotar_impacto_doping(self, resultados_doping: Dict):
        """
        Plota impacto de cold-start vs warm-start em tempo e fitness.
        """
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        nomes_alg = list(resultados_doping.keys())
        x_pos = np.arange(len(nomes_alg))
        width = 0.35
        
        # Tempo
        tempo_cold = [resultados_doping[alg]['tempo_cold_avg'] for alg in nomes_alg]
        tempo_warm = [resultados_doping[alg]['tempo_warm_avg'] for alg in nomes_alg]
        
        ax1.bar(x_pos - width/2, tempo_cold, width, label='Cold-Start', alpha=0.8, color='#d62728')
        ax1.bar(x_pos + width/2, tempo_warm, width, label='Warm-Start', alpha=0.8, color='#2ca02c')
        
        ax1.set_xlabel('Algoritmo', fontsize=11)
        ax1.set_ylabel('Tempo Total (segundos)', fontsize=11)
        ax1.set_title('Impacto do Doping no Tempo de Execução', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(nomes_alg)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Fitness médio
        fit_cold = [resultados_doping[alg]['fitness_cold_avg'] for alg in nomes_alg]
        fit_warm = [resultados_doping[alg]['fitness_warm_avg'] for alg in nomes_alg]
        
        ax2.bar(x_pos - width/2, fit_cold, width, label='Cold-Start', alpha=0.8, color='#d62728')
        ax2.bar(x_pos + width/2, fit_warm, width, label='Warm-Start', alpha=0.8, color='#2ca02c')
        
        ax2.set_xlabel('Algoritmo', fontsize=11)
        ax2.set_ylabel('Fitness Médio (erro em mm)', fontsize=11)
        ax2.set_title('Impacto do Doping na Qualidade da Solução', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(nomes_alg)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = os.path.join(self.path_base, 'plots', '03_impacto_doping.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Gráfico de impacto de doping salvo em {path}")
    
    def plotar_boxplots_metricas(self, resultados_por_alg: Dict[str, Dict[str, np.ndarray]]):
        """
        Gera boxplots das principais métricas para comparação estatística visual.
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        metricas = ['erro_final', 'jerk', 'indice_suavidade', 
                   'descontinuidade', 'taxa_convergencia']
        
        for idx, metrica in enumerate(metricas):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            
            dados_plotar = []
            labels = []
            
            for nome_alg, resultados in resultados_por_alg.items():
                if metrica in resultados:
                    dados_plotar.append(resultados[metrica])
                    labels.append(nome_alg)
            
            if dados_plotar:
                bp = ax.boxplot(dados_plotar, labels=labels, patch_artist=True)
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_ylabel(metrica, fontsize=10)
                ax.set_title(f'Boxplot: {metrica}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = os.path.join(self.path_base, 'plots', '04_boxplots_metricas.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Boxplots salvo em {path}")
    
    def plotar_comparacao_doping_simples(self, resultados_doping: Dict):
        """
        Plota comparação de cold-start vs warm-start de forma simples.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = {'PSO': '#1f77b4', 'AG': '#ff7f0e'}
        markers = {'cold': 'o-', 'warm': 's--'}
        
        for idx, (nome_alg, dados) in enumerate(resultados_doping.items()):
            ax = axes[idx]
            
            for tipo in ['cold', 'warm']:
                hist = dados.get(tipo, [])
                if hist:
                    hist_array = np.asarray(hist).flatten()
                    if len(hist_array) > 0:
                        label = 'Cold-Start' if tipo == 'cold' else 'Warm-Start (Doping)'
                        ax.plot(hist_array, markers.get(tipo, 'o-'), 
                               label=label, linewidth=2, 
                               color=colors.get(nome_alg, '#2ca02c'),
                               markersize=5)
            
            ax.set_xlabel('Iteração', fontsize=11)
            ax.set_ylabel('Erro de IK (mm)', fontsize=11)
            ax.set_title(f'Impacto de Doping - {nome_alg}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        path = os.path.join(self.path_base, 'plots', '03_impacto_doping.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Gráfico de impacto de doping salvo em {path}")
    
    def plotar_boxplots_simples(self, resultados_metricas: Dict):
        """
        Plota boxplots simples com os dados disponíveis.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metricas = ['convergencia', 'diversidade']
        colors = {'PSO': '#1f77b4', 'AG': '#ff7f0e', 'PSO-DFL': '#2ca02c'}
        
        for idx, metrica in enumerate(metricas):
            ax = axes[idx]
            
            dados_box = []
            labels_box = []
            box_colors = []
            
            for nome_alg in ['PSO', 'AG', 'PSO-DFL']:
                if nome_alg in resultados_metricas:
                    val = resultados_metricas[nome_alg].get(metrica, [])
                    if val:
                        val_array = np.asarray(val).flatten()
                        if len(val_array) > 0:
                            val_valid = val_array[np.isfinite(val_array)]
                            if len(val_valid) > 0:
                                dados_box.append(val_valid)
                                labels_box.append(nome_alg)
                                box_colors.append(colors.get(nome_alg, 'gray'))
            
            if dados_box:
                bp = ax.boxplot(dados_box, labels=labels_box, patch_artist=True)
                for patch, cor in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(cor)
                    patch.set_alpha(0.7)
                
                ax.set_ylabel(metrica.capitalize(), fontsize=11)
                ax.set_title(f'Distribuição de {metrica.capitalize()}', 
                            fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, f'Sem dados para {metrica}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        path = os.path.join(self.path_base, 'plots', '04_boxplots_metricas.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Boxplots salvo em {path}")
    
    def gerar_tabela_comparativa(self, relatorios: Dict) -> str:
        """
        Gera tabela em Markdown para incluir no artigo.
        """
        tabela = "# Tabela Comparativa: PSO vs AG vs PSO-DFL\n\n"
        tabela += "| Métrica | PSO | AG | PSO-DFL | Unidade |\n"
        tabela += "|---------|-----|----|---------|---------|\n"
        
        for nome_metrica, dados in relatorios.items():
            valores = {alg: dados.get('media', dados.get(alg, '?')) 
                      for alg, dados in dados.items()}
            
            linha = f"| {nome_metrica} | "
            for alg in ['PSO', 'AG', 'PSO-DFL']:
                val = valores.get(alg, '?')
                if isinstance(val, (int, float)):
                    linha += f"{val:.4f} | "
                else:
                    linha += f"{val} | "
            linha += " |\n"
            
            tabela += linha
        
        # Salvar
        path = os.path.join(self.path_base, 'statistics', 'tabela_comparativa.md')
        with open(path, 'w') as f:
            f.write(tabela)
        
        print(f"[INFO] Tabela comparativa salva em {path}")
        
        return tabela


def main():
    """
    Exemplo de execução completa.
    
    Para usar com seus dados, ajuste a importação de erick_dan.py
    e passe os resultados para o orquestrador.
    """
    print("=" * 70)
    print("ANÁLISE COMPARATIVA: PSO vs AG vs PSO-DFL")
    print("=" * 70)
    
    # Inicializar orquestrador
    orq = OrquestradorAnalises(
        path_base="analysis/results",
        n_runs=10,
        n_pontos_trajetoria=30
    )
    
    print("\n[INFO] Estrutura criada. Aguardando dados de execução dos algoritmos...")
    print("[INFO] Para usar este script:")
    print("   1. Execute seus testes com PSO, AG e PSO-DFL")
    print("   2. Colete históricos de população, fitness e trajetórias")
    print("   3. Passe os dados para os métodos do orquestrador")
    
    print("\nExemplo de uso:")
    print(">>> resultados_diversidade = orq.analisar_preservacao_diversidade(historico_pop)")
    print(">>> resultados_doping = orq.analisar_impacto_doping(cold_vs_warm)")
    print(">>> resultados_suavidade = orq.analisar_suavidade_trajetorias(trajetorias)")
    print(">>> orq.plotar_convergencia_comparativa(historicos_erro)")


if __name__ == '__main__':
    main()
