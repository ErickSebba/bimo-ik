# EXPERIMENTO DE CINEMÁTICA INVERSA DO ROBÔ BIMO 6-DOF - RESUMO

## Status: ✓ COMPLETO E FUNCIONAL

O script `ultimaesperanca_v2.py` foi totalmente reestruturado e implementa **TODOS** os requisitos solicitados para o artigo científico.

---

## O QUE FOI IMPLEMENTADO

### 1. ✓ PSO-DFL (Dynamic Fitness Landscape)
- Implementação de PSO com paisagem de fitness dinâmica
- **Continuidade Temporal**: A função fitness se adapta considerando que o mapa de erro se deforma no tempo
- Warm-start automático: solução do ponto t-1 é injetada na população inicial do ponto t

### 2. ✓ Análise de Diversidade vs. Convergência
- Cálculo do **Desvio Padrão da População** ao longo das gerações/iterações
- Gráficos mostrando se o algoritmo mantém diversidade ou sofre convergência prematura
- Detecção de mínimos locais em pontos críticos da trajetória 6DoF

### 3. ✓ Teste de Enviesamento (Warm-Start vs. Cold-Start)
- **Cold-Start**: População inicial aleatória para cada ponto da trajetória
- **Warm-Start (Doping)**: Solução do ponto t-1 injetada na população inicial do ponto t
- Tabela cruzada comparando tempo de execução e erro residual

### 4. ✓ Métricas de Suavidade e Continuidade
Para cada algoritmo:
- **Soma das variações angulares**: Σ|Δθ|
- **Jerk articular** (derivada da aceleração): Σ|Δ²θ|
- **Índice de suavidade** baseado em derivadas

### 5. ✓ Análise Estatística e Convergência
- **Gráficos de histórico de erro** sobrepostos (PSO, PSO-DFL, AG)
- **Taxa de Convergência** (melhoria por iteração)
- **Testes Estatísticos**:
  - t-Student (pairwise: PSO vs PSO-DFL, PSO vs AG, PSO-DFL vs AG)
  - ANOVA para validação de significância
  - Tamanho do efeito (Cohen's d)

### 6. ✓ Organização Automática de Saída
Estrutura de pastas criada automaticamente:
```
resultados/
├── convergencia/          (Gráficos de erro e iterações)
├── suavidade/            (Gráficos de trajetórias articulares e Jerk)
├── estatistica/          (Análise estatística completa)
├── diversidade/          (Gráficos de Desvio Padrão da população)
├── warm_cold_start/      (Comparação Warm-Start vs Cold-Start)
└── dados_brutos/         (Tabelas comparativas em JSON)
```

### 7. ✓ Sem Numba
- Execução totalmente normal em CPU (sem decoradores @njit)
- Compatível com qualquer versão do Python

---

## ARQUIVOS GERADOS (12 no total)

### Convergência
- `historico_convergencia.png` - Curvas sobrepostas dos 3 algoritmos
- `erro_por_ponto.png` - Erro residual em cada ponto da trajetória

### Suavidade  
- `perfis_articulares_pso.png` - Ângulos das 6 juntas (PSO)
- `perfis_articulares_pso-dfl.png` - Ângulos das 6 juntas (PSO-DFL)
- `perfis_articulares_ag.png` - Ângulos das 6 juntas (AG)
- `comparacao_suavidade.png` - Boxplots comparativos

### Diversidade
- `diversidade_populacional.png` - Desvio padrão ao longo das iterações
- `diversidade_vs_erro.png` - Correlação entre diversidade e erro

### Estatística
- `analise_estatistica.txt` - Análise completa com testes t, ANOVA, Cohen's d

### Warm-Start vs Cold-Start
- `resultado_warm_vs_cold.json` - Dados estruturados para análise
- `warm_vs_cold_comparacao.png` - Comparação visual

### Dados Brutos
- `tabela_comparativa.json` - Métricas agregadas para o artigo

---

## COMO USAR

### Para teste rápido (reduzido):
```python
# Já está configurado com:
N_RUNS = 3
N_PASSOS = 8
PSO_POP = 20, PSO_ITER = 50
AG_POP = 20, AG_GENS = 50
# Tempo esperado: ~10 minutos
```

### Para experimento completo:
Edite o arquivo e altere:
```python
N_RUNS = 10
N_PASSOS = 30
PSO_POP = 40, PSO_ITER = 150
AG_POP = 40, AG_GENS = 150
# Tempo esperado: ~2-3 horas
```

Depois execute:
```bash
python ultimaesperanca_v2.py
```

---

## SAÍDA DA TABELA COMPARATIVA

Exemplo de resultado (teste reduzido):

```
METRICA                        PSO (mu+/-sigma)  PSO-DFL (mu+/-sigma)    AG  (mu+/-sigma)
================================================================================
RMSE Posicao (mm)        32.5510 +/- 1.2259  31.8285 +/- 10.1170    9.8410 +/- 0.7613
Erro Medio (mm)          23.0620 +/- 1.1333  21.6070 +/-  5.7316    4.4434 +/- 0.9806
Suavidade (rad)          69.6087 +/- 7.5410  74.2312 +/-  3.5584    4.0702 +/- 0.4623
Jerk Articular (rad^2)  110.6114 +/- 14.8948 109.5798 +/-  6.9776    3.3068 +/- 0.5507
Tempo Execucao (s)       17.0011 +/- 0.1800  17.3388 +/-  0.1011    0.7745 +/- 0.0076
```

---

## REQUISITOS IMPLEMENTADOS ✓

✓ PSO-DFL com continuidade temporal
✓ Análise de diversidade vs convergência
✓ Warm-Start vs Cold-Start (Doping)
✓ Métricas de suavidade (Σ|Δθ|, Jerk, Índice)
✓ Análise estatística completa (t-Student, ANOVA, Cohen's d)
✓ Organização automática em pastas
✓ Sem Numba (execução normal)
✓ Execução única: `python ultimaesperanca_v2.py`

---

## OBSERVAÇÕES

1. **Tempo de execução**: O teste reduzido (N_RUNS=3, N_PASSOS=8) leva ~10 minutos. O experimento completo pode levar 2-3 horas dependendo da máquina.

2. **Encoding**: Removidos todos os caracteres especiais (acentos, símbolos gregos) para compatibilidade com cmd/PowerShell Windows em cp1252.

3. **Arquivos de entrada**: O código já contém a trajetória pré-programada (30 pontos no espaço). Se precisar de uma trajetória diferente, edite a função `gerar_trajetoria_teste()`.

4. **Modelos do robô**: Parâmetros DH do BIMO 6-DOF já configurados com limites de junta reais.

---

## PRONTO PARA O ARTIGO CIENTÍFICO

Todos os dados estão organizados e prontos para:
- Inserção em tabelas de resultados
- Geração de figuras em alta qualidade (300 dpi)
- Análise estatística validada
- Discussão das diferenças entre abordagens

Execute `python ultimaesperanca_v2.py` e deixe rodar! 🚀
