# Framework de Análises Comparativas: PSO vs AG vs PSO-DFL

## 📋 Visão Geral

Este framework foi criado para conduzir uma análise científica rigorosa comparando:
- **PSO** (Particle Swarm Optimization)
- **AG** (Algoritmo Genético)
- **PSO-DFL** (PSO com Dynamic Fitness Landscape)

Para resolver o problema de **Cinemática Inversa de um Robô BIMO 6-DoF**.

---

## 🗂️ Estrutura de Diretórios

```
analysis/
├── algorithms/              # Implementações dos algoritmos
│   ├── __init__.py
│   ├── pso_base.py         # PSO padrão e PSO-DFL
│   ├── ag_base.py          # Algoritmo Genético
│   └── experimental_setups.py  # Configurador de testes
│
├── metrics/                 # Módulo de métricas
│   ├── __init__.py
│   ├── convergence_metrics.py       # Velocidade e qualidade
│   ├── diversity_metrics.py         # Preservação de diversidade
│   ├── smoothness_metrics.py        # Suavidade de trajetória
│   └── statistics.py                # Testes estatísticos
│
├── results/                 # Saídas dos testes
│   ├── plots/              # Gráficos (PNG, PDF)
│   ├── statistics/         # Resumos estatísticos (JSON)
│   └── data/               # Dados brutos (NPZ)
│
├── run_analysis.py         # Orquestrador de análises
└── test_integrated_analysis.py  # Script de execução
```

---

## 🚀 Como Usar

### 1. Execução Básica

```bash
cd analysis
python test_integrated_analysis.py
```

Isso vai:
1. Executar PSO, AG e PSO-DFL múltiplas vezes
2. Coletar históricos de convergência e diversidade
3. Gerar análises estatísticas
4. Criar visualizações para artigo

### 2. Usar Diretamente seus Dados

```python
from analysis.run_analysis import OrquestradorAnalises
from analysis.metrics import (
    MetricasConvergencia,
    MetricasDiversidade,
    MetricasSuavidade,
    AnalisaEstatistica
)

# Seu orquestrador
orq = OrquestradorAnalises(path_base="analysis/results", n_runs=10)

# Após executar seus algoritmos, passar dados para análise
relatorio = orq.analisar_preservacao_diversidade(historico_pop)
resultados = orq.analisar_impacto_doping(cold_vs_warm)
suavidade = orq.analisar_suavidade_trajetorias(trajetorias)

# Gerar visualizações
orq.plotar_convergencia_comparativa(historico_erro)
orq.plotar_impacto_doping(resultados_doping)
```

---

## 📊 Métricas Implementadas

### 1. **Preservação de Diversidade vs Convergência Prematura**

**Classe:** `MetricasDiversidade`

Calcula:
- **Desvio Padrão da População**: Monitora dispersão das soluções
- **Taxa de Perda de Diversidade**: Indica se convergência é muito rápida
- **Ponto de Convergência Prematura**: Detecta quando população "morre"
- **Entropia de Shannon**: Medida informação-teórica de dispersão

**Interpretação:**
- PSO geralmente mantém maior diversidade
- AG tende a convergir mais rapidamente (pode ser prematura)
- PSO-DFL deve manter diversidade mesmo com landscape dinâmico

---

### 2. **Impacto do Enviesamento (Warm-Start vs Cold-Start)**

**Classe:** `MetricasConvergencia`

Compara:
- **Tempo de Execução**: Cold vs Warm
- **Qualidade Final**: Fitness alcançado em cada caso
- **Taxa de Melhoria por Ponto**: Quanto doping ajuda

**Resultado Esperado:**
```
PSO Cold-Start:  2.0 seg/ponto
PSO Warm-Start:  0.5 seg/ponto  (75% mais rápido)

AG Cold-Start:   1.5 seg/ponto
AG Warm-Start:   1.2 seg/ponto  (20% mais rápido)

→ PSO tem sinergia maior com doping!
```

---

### 3. **Métricas de Suavidade e Continuidade**

**Classe:** `MetricasSuavidade`

Calcula:
- **Soma de Variações Angulares**: Σ|Δθ| entre pontos
- **Jerk Articular**: Variação da aceleração (3ª derivada)
- **Índice de Suavidade**: Score [0,1] normali zado
- **Descontinuidade**: Detecta "saltos" na trajetória
- **Energia Mecânica**: Esforço total da trajetória

**Importância:**
- AG geralmente gera trajetórias mais suaves (elitismo)
- PSO pode produzir saltos abruptos
- PSO-DFL deve melhorar suavidade pelo landscape dinâmico

---

### 4. **Métricas de Convergência**

**Classe:** `MetricasConvergencia`

Mede:
- **Taxa de Convergência**: % melhoria por iteração
- **Iterações para Tolerância**: Quantas iterações para atingir 5% do melhor
- **Melhoria Relativa**: Quanto melhorou do início
- **Suavidade de Convergência**: Quanto oscila

**Padrão Esperado:**
```
PSO:        Convergência RÁPIDA inicial, pode oscilar
AG:         Convergência LENTA inicial, estável ao final
PSO-DFL:    Convergência EQUILIBRADA (melhor do dois mundos?)
```

---

### 5. **Análises Estatísticas**

**Classe:** `AnalisaEstatistica`

Aplicações:
- **Teste t-Student**: Diferenças significativas entre PSO e AG
- **ANOVA**: Comparação de 3+ grupos (PSO, AG, PSO-DFL)
- **Teste Wilcoxon**: Não-paramétrico, mais robusto
- **d de Cohen**: Tamanho do efeito (negligenciável vs grande)
- **Intervalo de Confiança**: Margem de erro (95%)
- **Teste Shapiro-Wilk**: Verificar normalidade dos dados

**Validação Científica:**
```json
{
  "metrica": "tempo_execucao",
  "t_statistic": -3.45,
  "p_value": 0.002,
  "significante": true,
  "cohens_d": 1.2,  // Efeito "grande"
  "interpretacao": "PSO é significativamente mais rápido (p < 0.05)"
}
```

---

## 📈 Visualizações Geradas

### 1. **Convergência Comparativa** (`01_convergencia_comparativa.png`)
- Gráficos de erro ao longo das iterações
- Escala linear E logarítmica para melhor visualização
- Média com ±1 desvio padrão

### 2. **Preservação de Diversidade** (`02_diversidade_preservacao.png`)
- Histórico de desvio padrão da população
- Detecta convergência prematura

### 3. **Impacto de Doping** (`03_impacto_doping.png`)
- Comparação lado-a-lado de Cold-Start vs Warm-Start
- Tempo e fitness para cada algoritmo

### 4. **Boxplots de Métricas** (`04_boxplots_metricas.png`)
- Distribuição de erro_final, jerk, suavidade, etc.
- Mostra outliers e quartis

---

## 📝 Arquivos de Saída

### Relatórios JSON

1. **`statistics/diversidade_preservacao.json`**
   - Desvio padrão inicial/final
   - Taxa de queda de diversidade
   - Iteração de convergência prematura

2. **`statistics/impacto_doping.json`**
   - Tempo cold vs warm
   - Fitness cold vs warm
   - Significância estatística

3. **`statistics/suavidade_trajetorias.json`**
   - Soma de variações
   - Jerk, suavidade, descontinuidade
   - Por algoritmo e com desvios

4. **`statistics/convergencia.json`**
   - Taxa de convergência (linear e exponencial)
   - Iterações para tolerância
   - Melhoria relativa

5. **`statistics/resumo_estatistico.json`**
   - Testes t-Student, ANOVA, Wilcoxon
   - Tamanho do efeito (Cohen's d)
   - Intervalos de confiança

---

## 🔧 Personalização

### Ajustar Número de Execuções

```python
integrador = IntegradorAnalises(
    n_runs=10  # Mais runs = mais robusto mas mais lento
)
```

### Mudar Parâmetros do PSO

```python
pso = PSO_Padrao(
    fitness_func=sua_funcao,
    pop=50,        # Aumentar população
    max_iter=200,  # Mais iterações
    seed=42
)
```

### Usar PSO-DFL com Modo Adaptativo Diferente

```python
pso_dfl = PSO_DFL(
    fitness_func=sua_funcao,
    modo_adaptacao='exponencial'  # ou 'linear', 'sigmoid'
)
```

### Incluir Mais Métricas

```python
from analysis.metrics import MetricasDiversidade

# Calcular entropia personalizada
entropia = MetricasDiversidade.entropia_populacao(
    populacoes_historico,
    n_bins=20  # Mais bins = mais detalhado
)
```

---

## 📚 Referências Implementadas

1. **Duymazlar & Gönenli (Cap. 17)**
   - Comparação PSO vs AG em Delta robot
   - Métrica de preservação de diversidade

2. **Farnese et al.**
   - Crossover aritmético (AG4)
   - Seleção por roleta
   - Configuração justa para comparação

3. **Luo et al. (2022) - MOFOPSO**
   - Parâmetros de PSO adaptativos
   - Fronteira de Pareto para multi-objetivo

4. **IROS / IEEE Standards**
   - Índice de suavidade baseado em derivadas
   - Métricas de jerk articular

---

## ⚡ Performance

Tempo estimado (10 runs, 30 pontos trajetória):

| Algoritmo | Tempo |
|-----------|-------|
| PSO       | ~2 min |
| AG        | ~3 min |
| PSO-DFL   | ~3.5 min |
| **Total** | **~8 min** |

---

## 🐛 Troubleshooting

### ImportError: numpy not found
```bash
pip install numpy scipy matplotlib seaborn scikit-opt
```

### Dados vazios nas análises
Certifique-se de:
1. Algoritmos estão gerando históricos (checar `.historico_melhor_fitness`)
2. Dados estão no formato correto (arrays numpy)
3. Passar dados com formato certo para `orq.analisar_*()`

### Gráficos não aparecem
Usar: `plt.savefig()` em vez de `plt.show()`
Resultados salvos em `analysis/results/plots/`

---

## 💡 Dicas para Artigo

### Seção de Metodologia
Cite:
- "Utilizamos n=10 execuções independentes com sementes aleatórias controladas"
- "Cada algoritmo foi avaliado com população=40, iterações=150"
- "Doping implementado conforme Farnese et al."

### Seção de Resultados
Use:
```
Tabela 1: Resumo de Métricas
Figura 1: Convergência Comparativa
Figura 2: Impacto do Doping
Figura 3: Boxplots de Distribuição
```

### Seção de Discussão
"O teste t-Student (p=0.002, d=1.2) mostra diferença significativa..."

---

## 📄 Licença e Autoria

Desenvolvido para suportar artigo científico sobre Cinemática Inversa de Robô BIMO.

---

## 📞 Suporte

Para dúvidas ou melhorias:
1. Verificar exemplos em `test_integrated_analysis.py`
2. Consultar docstrings das classes
3. Executar `python -m pydoc analysis.metrics` para documentação inline
