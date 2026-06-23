#!/usr/bin/env python3
"""
=============================================================================
  ORQUESTRADOR PRINCIPAL - CINEMÁTICA INVERSA DO ROBÔ BIMO (6-DOF)
  
  Este script coordena a execução dos experimentos importando as lógicas
  dos módulos especializados dentro do pacote 'bimo'.
=============================================================================
"""

import sys
import json
from pathlib import Path
import numpy as np
from scipy import stats

# Garante que o Python enxergue a pasta local 'bimo' corretamente
sys.path.append(str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS DO SEU PACOTE MODULAR (Alinhado exatamente com a sua árvore de arquivos)
# ─────────────────────────────────────────────────────────────────────────────
try:
    # Configurações globais e caminhos
    from bimo.config import (
        DIRS, TRAJETORIA, SEED_BASE, N_RUNS, N_PASSOS, 
        PSO_POP, PSO_ITER, AG_POP, AG_GENS
    )
    
    # Seus algoritmos isolados
    from bimo.algorithms.pso import rodar_pso
    from bimo.algorithms.pso_dfl import rodar_pso_dfl
    from bimo.algorithms.ga import rodar_ag
    
    # Seus utilitários de cálculo e visualização
    from bimo.utils.metricas import calcular_metricas
    from bimo.utils.warm_cold import teste_warm_vs_cold
    from bimo.utils.plots_extras import (
        gerar_graficos_convergencia,
        gerar_graficos_diversidade,
        gerar_graficos_suavidade,
        gerar_grafico_trajetoria  # Nova função 3D na pasta 'trajetorias'
    )

except ImportError as e:
    print(f"\n[ERRO DE IMPORTAÇÃO] Erro ao carregar os módulos do pacote 'bimo'.")
    print(f"Verifique se as variáveis globais e caminhos estão expostos no seu 'bimo/config.py'.")
    print(f"Detalhe do erro: {e}\n")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE ORQUESTRAÇÃO E RELATÓRIO DO EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────

def rodar_experimento_completo():
    """Executa as N rodadas de cada algoritmo importado do bloco algorithms."""
    print("=" * 75)
    print("EXPERIMENTO COMPARATIVO MODULAR: PSO vs PSO-DFL vs AG")
    print(f"Rodadas (N_RUNS)={N_RUNS} | Pontos na Trajetória (N_PASSOS)={N_PASSOS}")
    print("=" * 75)
    
    resultados_pso = []
    resultados_pso_dfl = []
    resultados_ag = []
    
    for run in range(N_RUNS):
        seed = SEED_BASE + run
        print(f"\n[Rodada {run+1}/{N_RUNS}] (Seed: {seed})")
        
        # Execução PSO
        print("  [Módulo PSO] executando...", end=" ", flush=True)
        res_pso = rodar_pso(seed=seed, rastrear_diversidade=True)
        metricas_pso = calcular_metricas(res_pso)
        resultados_pso.append(metricas_pso)
        print(f"RMSE={metricas_pso['rmse_pos_mm']:.3f}mm | t={metricas_pso['tempo_s']:.1f}s")
        
        # Execução PSO-DFL
        print("  [Módulo PSO-DFL] executando...", end=" ", flush=True)
        res_pso_dfl = rodar_pso_dfl(seed=seed, rastrear_diversidade=True)
        metricas_pso_dfl = calcular_metricas(res_pso_dfl)
        resultados_pso_dfl.append(metricas_pso_dfl)
        print(f"RMSE={metricas_pso_dfl['rmse_pos_mm']:.3f}mm | t={metricas_pso_dfl['tempo_s']:.1f}s")
        
        # Execução AG
        print("  [Módulo AG] executando...", end=" ", flush=True)
        res_ag = rodar_ag(seed=seed, rastrear_diversidade=True)
        metricas_ag = calcular_metricas(res_ag)
        resultados_ag.append(metricas_ag)
        print(f"RMSE={metricas_ag['rmse_pos_mm']:.3f}mm | t={metricas_ag['tempo_s']:.1f}s")
        
    return resultados_pso, resultados_pso_dfl, resultados_ag


def imprimir_tabela_formatada(resultados_pso, resultados_pso_dfl, resultados_ag):
    """Gera o relatório analítico no console e salva a tabela em formato JSON."""
    print("\n" + "=" * 105)
    print(f"{'MÉTRICA EXTRAÍDA':<30} {'PSO (média +/- std)':>24}  {'PSO-DFL (média +/- std)':>24}  {'AG (média +/- std)':>24}")
    print("=" * 105)
    
    metricas_alvo = [
        ("rmse_pos_mm", "RMSE Posição (mm)"),
        ("erro_medio_mm", "Erro Médio (mm)"),
        ("erro_max_mm", "Erro Máximo (mm)"),
        ("suavidade_rad", "Suavidade Total (rad)"),
        ("jerk_articular", "Jerk Articular (rad^2)"),
        ("indice_suavidade", "Índice Suavidade"),
        ("taxa_convergencia", "Taxa de Convergência"),
        ("fitness_medio", "Fitness Médio"),
        ("tempo_s", "Tempo de Execução (s)"),
    ]
    
    tabela_exportacao = []
    
    for chave, nome in metricas_alvo:
        vals_pso = np.array([r[chave] for r in resultados_pso])
        vals_dfl = np.array([r[chave] for r in resultados_pso_dfl])
        vals_ag = np.array([r[chave] for r in resultados_ag])
        
        print(f"{nome:<30} {vals_pso.mean():8.4f} +/- {vals_pso.std():6.4f}  "
              f"{vals_dfl.mean():8.4f} +/- {vals_dfl.std():6.4f}  "
              f"{vals_ag.mean():8.4f} +/- {vals_ag.std():6.4f}")
        
        tabela_exportacao.append({
            "Metrica": nome,
            "PSO_mu": float(vals_pso.mean()), "PSO_sigma": float(vals_pso.std()),
            "PSO_DFL_mu": float(vals_dfl.mean()), "PSO_DFL_sigma": float(vals_dfl.std()),
            "AG_mu": float(vals_ag.mean()), "AG_sigma": float(vals_ag.std())
        })
    print("=" * 105)
    
    # Exporta para a pasta correta mapeada pelo config
    caminho_json = DIRS["dados_brutos"] / "tabela_comparativa.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(tabela_exportacao, f, indent=2)
    print(f"[OK] Dados brutos consolidados em: {caminho_json}")


def rodar_validacao_estatistica(resultados_pso, resultados_pso_dfl, resultados_ag):
    """Executa testes de hipótese ANOVA e t-Student salvando relatório em arquivo txt."""
    print("\n[PIPELINE] Processando testes estatísticos de significância...")
    caminho_txt = DIRS["estatistica"] / "analise_estatistica.txt"
    
    metricas_estat = [
        ("rmse_pos_mm", "RMSE Posição (mm)"),
        ("suavidade_rad", "Suavidade (rad)"),
        ("tempo_s", "Tempo de Execução (s)"),
        ("jerk_articular", "Jerk Articular (rad^2)")
    ]
    
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ANÁLISE ESTATÍSTICA AVANÇADA - VALIDÇÃO DO ROBÔ BIMO\n")
        f.write("=" * 80 + "\n\n")
        
        for chave, nome in metricas_estat:
            vals_pso = np.array([r[chave] for r in resultados_pso])
            vals_dfl = np.array([r[chave] for r in resultados_pso_dfl])
            vals_ag = np.array([r[chave] for r in resultados_ag])
            
            f.write(f"\nMÉTRICA: {nome}\n" + "-"*40 + "\n")
            f_stat, p_anova = stats.f_oneway(vals_pso, vals_dfl, vals_ag)
            f.write(f"ANOVA -> F-statistic: {f_stat:.5f} | p-valor: {p_anova:.6f}\n")
            f.write(f"Significativo (p < 0.05): {'SIM' if p_anova < 0.05 else 'NÃO'}\n")
            
            # Pairwise t-Student
            _, p_pso_dfl = stats.ttest_ind(vals_pso, vals_dfl)
            _, p_pso_ag = stats.ttest_ind(vals_pso, vals_ag)
            _, p_dfl_ag = stats.ttest_ind(vals_dfl, vals_ag)
            
            f.write(f"  t-Test (PSO vs PSO-DFL): p = {p_pso_dfl:.6f}\n")
            f.write(f"  t-Test (PSO vs AG):      p = {p_pso_ag:.6f}\n")
            f.write(f"  t-Test (PSO-DFL vs AG):  p = {p_dfl_ag:.6f}\n")
            
    print(f"  [OK] Relatório estatístico salvo em: {caminho_txt}")


# ─────────────────────────────────────────────────────────────────────────────
# FLUXO DE EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "🚀 " * 20)
    print("      INICIANDO ENGINE MODULAR BIMO - CINEMÁTICA INVERSA")
    print("🚀 " * 20 + "\n")
    
    # 1. Core dos Experimentos
    res_pso, res_pso_dfl, res_ag = rodar_experimento_completo()
    
    # 2. Métricas Analíticas e Tabelas
    imprimir_tabela_formatada(res_pso, res_pso_dfl, res_ag)
    
    # 3. Módulo Visual (Geração de Gráficos do plots_extras.py)
    print("\n[PIPELINE] Acionando pipeline gráfico (bimo/utils/plots_extras.py)...")
    gerar_graficos_convergencia(res_pso, res_pso_dfl, res_ag)
    gerar_graficos_diversidade(res_pso, res_pso_dfl, res_ag)
    gerar_graficos_suavidade(res_pso, res_pso_dfl, res_ag)
    gerar_grafico_trajetoria(res_pso, res_pso_dfl, res_ag)  # Integração do seu gráfico 3D!
    
    # 4. Validação Hipotética/Estatística
    rodar_validacao_estatistica(res_pso, res_pso_dfl, res_ag)
    
    # 5. Teste Warm-Start vs Cold-Start (bimo/utils/warm_cold.py)
    print("\n[PIPELINE] Iniciando simulação de Doping (Warm vs Cold)...")
    teste_warm_vs_cold()
    
    print("\n" + "=" * 75)
    print("  ✓ SUCESSO: O pipeline modular foi finalizado sem nenhuma inconsistência!")
    print(f"  Todos os artefatos foram divididos e salvos na pasta: /{DIRS['dados_brutos'].parent}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()