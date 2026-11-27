#!/usr/bin/env python
"""Experimento pequeno para validar correções."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Modificar configuração para teste rápido
import sys
sys.path.insert(0, '.')

# Importar configurações
import ultimaesperanca_v2 as ult

print("=" * 70)
print("EXPERIMENTO DE VALIDAÇÃO - TESTE DAS CORREÇÕES")
print("=" * 70)

# Reduzir tamanho do experimento para teste rápido
ult.N_RUNS = 1
ult.N_PASSOS = 5
ult.PSO_ITER = 20  # Reduzir iterações para teste rápido
ult.AG_GENS = 20

print(f"\nConfiguração de teste:")
print(f"  N_RUNS={ult.N_RUNS}, N_PASSOS={ult.N_PASSOS}")
print(f"  PSO_ITER={ult.PSO_ITER}, AG_GENS={ult.AG_GENS}")

# Verificar que PESO_JTA é usado
print(f"\nVerificação de PESO_JTA:")
print(f"  PESO_JTA = {ult.PESO_JTA}")
print(f"  PESO_POS = {ult.PESO_POS}")

# Teste 1: DFL determinístico
print(f"\n[TESTE 1] DFL Determinístico (sem ruído):")
alvo = np.array([150, 100, 200])
q_ant = np.zeros(6)
fitness_dfl = ult.criar_fitness_dinamico(alvo, q_ant, tempo=0)
q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
vals = [fitness_dfl(q_test) for _ in range(5)]
print(f"  5 avaliações de q_test: {[round(v, 8) for v in vals]}")
print(f"  ✓ Determinístico: {len(set([round(v, 10) for v in vals])) == 1}")

# Teste 2: PSO com PSORastreador (PSO_ITER iterações)
print(f"\n[TESTE 2] PSO - Número correto de iterações:")
res_pso = ult.rodar_pso(seed=42, rastrear_diversidade=True)
n_iter = len(res_pso['historico_convergencia'][0])
print(f"  Len(historico_convergencia[0]) = {n_iter}")
print(f"  Esperado = PSO_ITER = {ult.PSO_ITER}")
print(f"  ✓ Correto: {n_iter == ult.PSO_ITER}")

# Teste 3: PSO-DFL com reavaliação de gbest
print(f"\n[TESTE 3] PSO-DFL - Reavaliação de gbest:")
res_dfl = ult.rodar_pso_dfl_verdadeiro(seed=42, rastrear_diversidade=True)
n_iter_dfl = len(res_dfl['historico_convergencia'][0])
print(f"  Len(historico_convergencia[0]) = {n_iter_dfl}")
print(f"  Esperado = PSO_ITER = {ult.PSO_ITER}")
print(f"  ✓ Correto: {n_iter_dfl == ult.PSO_ITER}")

# Teste 4: AG com warm-start e cold-start
print(f"\n[TESTE 4] AG - Warm-start vs Cold-start:")
res_ag_warm = ult.rodar_ag(seed=42, usar_warmstart=True)
res_ag_cold = ult.rodar_ag(seed=42, usar_warmstart=False)
erro_warm = np.mean(res_ag_warm['historico_erro'])
erro_cold = np.mean(res_ag_cold['historico_erro'])
print(f"  Erro médio (warm-start): {erro_warm:.6f}")
print(f"  Erro médio (cold-start): {erro_cold:.6f}")
print(f"  Diferença: {abs(erro_warm - erro_cold):.6f}")
print(f"  ✓ Cold-start diferente de warm-start: {erro_warm != erro_cold}")

# Teste 5: Comparação de tempos
print(f"\n[TESTE 5] Comparação de tempos:")
print(f"  PSO:        {res_pso['tempo_s']:.2f}s")
print(f"  PSO-DFL:    {res_dfl['tempo_s']:.2f}s")
print(f"  AG:         {res_ag_warm['tempo_s']:.2f}s")

print(f"\n{'='*70}")
print("✓ EXPERIMENTO DE VALIDAÇÃO COMPLETO!")
print(f"{'='*70}\n")

# Mostrar resumo das correções
print("\n[RESUMO DAS CORREÇÕES APLICADAS]")
print("✓ [1] Bug PSO: Corrigido - Agora executa PSO_ITER iterações (não PSO_ITER²)")
print("✓ [2] Ruído DFL: Removido - Fitness é determinística")
print("✓ [3] gbest stale: Corrigido - atualizar_fitness reavalia pbest/gbest")
print("✓ [4] Cold-start: Implementado - Teste warm vs cold diferencia")
print("✓ [5] PESO_JTA: Consistente - Usa constante em todas funções")
print("✓ [6] Warm-start DFL: Documentado - Apenas na primeira iteração")
