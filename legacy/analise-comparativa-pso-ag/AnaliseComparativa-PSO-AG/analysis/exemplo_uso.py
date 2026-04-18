"""
EXEMPLO DE USO RÁPIDO
Mostra como usar o framework sem precisar executar tudo
"""

import numpy as np
from analysis.metrics import (
    MetricasConvergencia,
    MetricasDiversidade,
    MetricasSuavidade,
    AnalisaEstatistica
)

# ============================================================================
# EXEMPLO 1: Análise de Convergência
# ============================================================================

print("="*70)
print("EXEMPLO 1: Analisando Convergência")
print("="*70)

# Simular histórico de erro para 3 algoritmos
n_iteracoes = 150

# PSO: converge rápido inicialmente, depois estabiliza
fitness_pso = 100 * np.exp(-0.03 * np.arange(n_iteracoes)) + 5

# AG: converge lentamente, depois consistente
fitness_ag = 100 / (1 + 0.1 * np.arange(n_iteracoes)) + 3

# PSO-DFL: intermediário
fitness_dfl = 50 * np.exp(-0.02 * np.arange(n_iteracoes)) + 4

# Analisar
print("\nPSO:")
rel_pso = MetricasConvergencia.relatorio_convergencia(fitness_pso, "PSO")
print(f"  Erro inicial: {rel_pso['erro_inicial']:.2f} mm")
print(f"  Erro final: {rel_pso['erro_final']:.2f} mm")
print(f"  Taxa convergência: {rel_pso['taxa_convergencia_linear']:.3f} %/iter")
print(f"  Melhoria total: {rel_pso['melhoria_relativa']:.1f} %")

print("\nAG:")
rel_ag = MetricasConvergencia.relatorio_convergencia(fitness_ag, "AG")
print(f"  Erro inicial: {rel_ag['erro_inicial']:.2f} mm")
print(f"  Erro final: {rel_ag['erro_final']:.2f} mm")
print(f"  Taxa convergência: {rel_ag['taxa_convergencia_linear']:.3f} %/iter")
print(f"  Melhoria total: {rel_ag['melhoria_relativa']:.1f} %")

print("\nPSO-DFL:")
rel_dfl = MetricasConvergencia.relatorio_convergencia(fitness_dfl, "PSO-DFL")
print(f"  Erro inicial: {rel_dfl['erro_inicial']:.2f} mm")
print(f"  Erro final: {rel_dfl['erro_final']:.2f} mm")
print(f"  Taxa convergência: {rel_dfl['taxa_convergencia_linear']:.3f} %/iter")
print(f"  Melhoria total: {rel_dfl['melhoria_relativa']:.1f} %")

# Teste estatístico
print("\nTeste t-Student (PSO vs AG):")
teste = AnalisaEstatistica.teste_t_student(fitness_pso, fitness_ag)
print(f"  t-statistic: {teste['t_statistic']:.3f}")
print(f"  p-value: {teste['p_value']:.6f}")
print(f"  Significante (α=0.05): {teste['significante_alpha_0.05']}")

# ============================================================================
# EXEMPLO 2: Análise de Diversidade
# ============================================================================

print("\n" + "="*70)
print("EXEMPLO 2: Analisando Diversidade da População")
print("="*70)

# Simular populações ao longo das iterações
pop_pso = []
pop_ag = []

for i in range(n_iteracoes):
    # PSO: população dispersa no início, converge rápido
    if i < 30:
        pop_pso.append(np.random.randn(40, 6) * (1 - i/30))
    else:
        pop_pso.append(np.random.randn(40, 6) * 0.05)
    
    # AG: população mais compacta, mantém diversidade
    if i < 60:
        pop_ag.append(np.random.randn(40, 6) * (1 - i/60) * 0.7)
    else:
        pop_ag.append(np.random.randn(40, 6) * 0.1)

print("\nPSO:")
rel_div_pso = MetricasDiversidade.relatorio_diversidade(pop_pso, "PSO")
print(f"  Diversidade inicial: {rel_div_pso['diversidade_media_inicial']:.4f}")
print(f"  Diversidade final: {rel_div_pso['diversidade_media_final']:.4f}")
print(f"  Taxa queda: {rel_div_pso['taxa_queda_diversidade']:.6f} por iteração")
print(f"  Iteração convergência prematura: {rel_div_pso['iteracao_convergencia_prematura']}")

print("\nAG:")
rel_div_ag = MetricasDiversidade.relatorio_diversidade(pop_ag, "AG")
print(f"  Diversidade inicial: {rel_div_ag['diversidade_media_inicial']:.4f}")
print(f"  Diversidade final: {rel_div_ag['diversidade_media_final']:.4f}")
print(f"  Taxa queda: {rel_div_ag['taxa_queda_diversidade']:.6f} por iteração")
print(f"  Iteração convergência prematura: {rel_div_ag['iteracao_convergencia_prematura']}")

# ============================================================================
# EXEMPLO 3: Análise de Suavidade
# ============================================================================

print("\n" + "="*70)
print("EXEMPLO 3: Analisando Suavidade da Trajetória")
print("="*70)

# Simular trajetórias
n_pontos = 30
n_juntas = 6

# Trajetória suave (AG)
q_ag = np.zeros((n_pontos, n_juntas))
for i in range(1, n_pontos):
    q_ag[i] = q_ag[i-1] + np.random.randn(n_juntas) * 0.1  # Pequenas variações

# Trajetória com saltos (PSO ruim)
q_pso_ruim = np.zeros((n_pontos, n_juntas))
for i in range(1, n_pontos):
    if np.random.rand() < 0.3:
        q_pso_ruim[i] = q_pso_ruim[i-1] + np.random.randn(n_juntas) * 0.5  # Salto
    else:
        q_pso_ruim[i] = q_pso_ruim[i-1] + np.random.randn(n_juntas) * 0.05

print("\nTrajetória suave (AG):")
rel_suav_ag = MetricasSuavidade.relatorio_suavidade(q_ag, "AG")
print(f"  Soma variações: {rel_suav_ag['soma_variacoes_totais']:.4f} rad")
print(f"  Jerk articular: {rel_suav_ag['jerk_articular']:.6f}")
print(f"  Índice suavidade: {rel_suav_ag['indice_suavidade']:.4f} (1.0 = perfeito)")
print(f"  Descontinuidade: {rel_suav_ag['descontinuidade']:.4f}")

print("\nTrajetória com saltos (PSO ruim):")
rel_suav_pso = MetricasSuavidade.relatorio_suavidade(q_pso_ruim, "PSO")
print(f"  Soma variações: {rel_suav_pso['soma_variacoes_totais']:.4f} rad")
print(f"  Jerk articular: {rel_suav_pso['jerk_articular']:.6f}")
print(f"  Índice suavidade: {rel_suav_pso['indice_suavidade']:.4f}")
print(f"  Descontinuidade: {rel_suav_pso['descontinuidade']:.4f}")

print(f"\n→ Quanto menor índice suavidade, piores as mudanças abruptas")
print(f"→ AG é {rel_suav_ag['indice_suavidade']/rel_suav_pso['indice_suavidade']:.1f}x mais suave")

# ============================================================================
# EXEMPLO 4: Análise Estatística Completa
# ============================================================================

print("\n" + "="*70)
print("EXEMPLO 4: Resumo Estatístico Completo")
print("="*70)

# Simular múltiplas execuções (n_runs=10)
n_runs = 10
resultados = {
    'PSO': {
        'erro_final': np.abs(np.random.normal(8, 2, n_runs)),
        'jerk': np.abs(np.random.normal(0.5, 0.1, n_runs)),
        'tempo': np.abs(np.random.normal(120, 10, n_runs))
    },
    'AG': {
        'erro_final': np.abs(np.random.normal(6, 2, n_runs)),
        'jerk': np.abs(np.random.normal(0.3, 0.1, n_runs)),
        'tempo': np.abs(np.random.normal(180, 15, n_runs))
    },
    'PSO-DFL': {
        'erro_final': np.abs(np.random.normal(7, 1.5, n_runs)),
        'jerk': np.abs(np.random.normal(0.35, 0.08, n_runs)),
        'tempo': np.abs(np.random.normal(150, 12, n_runs))
    }
}

resumo = AnalisaEstatistica.resumo_estatistico_completo(resultados)

print("\nMétrica: erro_final")
print("-" * 50)
for alg, stats in resumo['erro_final']['descritivas'].items():
    print(f"{alg}:")
    print(f"  Média: {stats['media']:.3f} ± {stats['desvio']:.3f} mm")
    print(f"  Min-Max: {stats['min']:.3f} - {stats['max']:.3f}")

print("\nTeste ANOVA (comparação dos 3 algoritmos):")
anova = resumo['erro_final']['anova']
print(f"  F-statistic: {anova['F_statistic']:.3f}")
print(f"  p-value: {anova['p_value']:.6f}")
print(f"  Significante: {anova['significante_alpha_0.05']}")

print("\nMétrica: tempo")
print("-" * 50)
print("Tamanho do Efeito (Cohen's d) - PSO vs AG:")
cohens_d = AnalisaEstatistica.size_effect_cohens_d(
    resultados['PSO']['tempo'],
    resultados['AG']['tempo']
)
print(f"  d = {cohens_d:.3f}", end=" → ")
if abs(cohens_d) < 0.2:
    print("Negligenciável")
elif abs(cohens_d) < 0.5:
    print("Pequeno")
elif abs(cohens_d) < 0.8:
    print("Médio")
else:
    print("GRANDE")

# Intervalo de confiança
ic_pso = AnalisaEstatistica.intervalo_confianca(
    resultados['PSO']['tempo'],
    confianca=0.95
)
print(f"\nIntervalo de Confiança 95% (PSO):")
print(f"  [{ic_pso[0]:.1f}, {ic_pso[1]:.1f}] segundos")

# ============================================================================
# CONCLUSÕES
# ============================================================================

print("\n" + "="*70)
print("RESUMO DAS ANÁLISES")
print("="*70)

print("""
✓ Convergência:
  - PSO: Rápido inicialmente, pode oscilar
  - AG: Lento no início, consistente ao final
  - PSO-DFL: Intermediário

✓ Diversidade:
  - PSO: Cai rápido (convergência agressiva)
  - AG: Mantém melhor (espaço de busca maior)
  - PSO-DFL: Deve manter (parâmetros adaptativos)

✓ Suavidade:
  - AG: Mais suave (elitismo mantém boas soluções)
  - PSO: Pode ter saltos (menos estrutura)
  - PSO-DFL: Reduz saltos pelo landscape dinâmico

✓ Significância Estatística:
  - Use p-value < 0.05 para validar diferenças
  - Cohen's d mostra magnitude (não apenas significância)
  - ANOVA compara 3 algoritmos simultaneamente
""")

print("="*70)
print("FIM DO EXEMPLO")
print("="*70)
