# Análise Comparativa: PSO vs AG vs PSO-DFL

Pacote completo para execução de análise comparativa entre algoritmos de otimização (PSO, Algoritmo Genético e PSO com Dinâmica de Fitness Adaptativa).

## Descrição

Este projeto implementa uma bateria completa de testes comparativos entre três algoritmos de otimização aplicados ao problema de **cinemática inversa de um robô 6-DoF (BIMO)**:

1. **PSO (Particle Swarm Optimization)** - Tradicional
2. **AG (Algoritmo Genético)** - Com crossover aritmético e mutação gaussiana
3. **PSO-DFL** - PSO com Dynamic Fitness Landscape para continuidade

### Características

✅ **Múltiplas Execuções**: Garante rigor estatístico com N_RUNS configurável  
✅ **Warm-Start vs Cold-Start**: Testa impacto do "doping" (solução anterior como seed)  
✅ **Métricas Abrangentes**: Convergência, diversidade, suavidade, tempo de execução  
✅ **Análises Estatísticas**: t-Student, ANOVA, Wilcoxon  
✅ **Visualizações Prontas**: Gráficos em PNG de qualidade para publicação  
✅ **Relatórios JSON**: Dados brutos para análise posterior  

---

## Estrutura de Pastas

```
AnaliseComparativa-PSO-AG/
├── test_integrated_analysis.py    # Script principal
├── erick_dan.py                   # Base de cinemática e trajetória
├── README.md                      # Este arquivo
├── pyproject.toml                 # Dependências Python
├── setup.py                       # Configuração de instalação
│
├── analysis/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── pso_base.py            # Implementação do PSO
│   │   ├── ag_base.py             # Implementação do AG
│   │   ├── experimental_setups.py # Configurações
│   │   └── __init__.py
│   │
│   ├── metrics/
│   │   ├── convergence_metrics.py # Análise de convergência
│   │   ├── diversity_metrics.py   # Análise de diversidade
│   │   ├── smoothness_metrics.py  # Análise de suavidade
│   │   ├── statistics.py          # Testes estatísticos
│   │   └── __init__.py
│   │
│   ├── run_analysis.py            # Orquestrador de análises
│   └── results/                   # Pasta de saída (criada automaticamente)
│       ├── data/
│       ├── plots/
│       └── statistics/
```

---

## Como Instalar

### 1. Pré-requisitos

- Python 3.8+
- pip ou conda

### 2. Instalação de Dependências

**Opção A: Usando pip**
```bash
pip install -r requirements.txt
```

**Opção B: Usando pyproject.toml**
```bash
pip install .
```

**Opção C: Manualmente**
```bash
pip install numpy scipy matplotlib seaborn scikit-opt
```

### Dependências Principais

| Pacote | Versão | Uso |
|--------|--------|-----|
| numpy | >=1.20 | Computação numérica |
| scipy | >=1.7 | Testes estatísticos |
| matplotlib | >=3.3 | Visualizações |
| seaborn | >=0.11 | Gráficos estatísticos |
| scikit-opt | >=0.5 | Algoritmo PSO |

---

## Como Executar

### Execução Rápida (Padrão)

```bash
python test_integrated_analysis.py
```

**Parâmetros Padrão**:
- `n_runs = 20` (20 execuções independentes)
- `N_PASSOS = 30` (30 pontos na trajetória)
- `PSO_ITER = 150` (150 iterações por PSO)
- `AG_GENS = 150` (150 gerações por AG)

### Customização

Edite as primeiras linhas de `test_integrated_analysis.py`:

```python
# Para teste rápido (1-2 min)
# Descomente no main():
N_RUNS = 3
PSO_ITER = 50
AG_GENS = 50

# Para máxima precisão (recomendado para artigo: 30-60 min)
N_RUNS = 50
PSO_ITER = 200
AG_GENS = 200
```

---

## Saídas Geradas

### 📊 Pasta `analysis/results/`

#### `data/`
- `dados_brutos.json` - Todos os históricos de erro e diversidade em JSON

#### `plots/`
- `fig1_trajetoria_3d.png` - Trajetória 3D desejada vs obtida
- `fig2_erro_rastreamento.png` - Erro por ponto (média ± σ)
- `fig3_convergencia.png` - Convergência dos 3 algoritmos
- `fig4_boxplot_metricas.png` - Boxplots comparativos
- `fig5_warm_vs_cold.png` - Impacto do warm-start

#### `statistics/`
- `relatorio_completo.json` - Estatísticas resumidas
- `testes_estatisticos.json` - Resultados t-Student, ANOVA, Wilcoxon
- `metricas_suavidade.json` - Jerk e variações articulares
- `resumo_analises.txt` - Resumo em texto legível

---

## Métricas Coletadas

### 1. **Convergência**
- Erro final médio (mm)
- Erro máximo observado (mm)
- Erro mínimo obtido (mm)
- Desvio padrão do erro

### 2. **Diversidade**
- Desvio padrão da população ao longo das iterações
- Detecta convergência prematura
- Indica capacidade de exploração

### 3. **Suavidade**
- Soma de variações angulares: Σ|Δθ|
- Jerk articular (derivada da aceleração)
- Índice de suavidade normalizado

### 4. **Tempo de Execução**
- Tempo total por algoritmo (segundos)
- Tempo médio por ponto da trajetória

### 5. **Análises Estatísticas**
- **t-Student**: PSO vs AG, PSO vs PSO-DFL
- **ANOVA**: Comparação dos 3 algoritmos
- **p-values**: Para validação científica (α = 0.05)

---

## Interpretação dos Resultados

### ✅ Bom Desempenho

| Métrica | Intervalo | Interpretação |
|---------|-----------|---------------|
| RMSE Posição | < 5 mm | Erro aceitável |
| Suavidade | Baixa | Trajetória suave |
| Diversidade | Alta (início) → Baixa (fim) | Exploração depois convergência |
| p-value | < 0.05 | Diferença significativa |

### ⚠️ Sinais de Atenção

- RMSE > 20 mm: Ajustar limites articulares ou trajetória
- Diversidade sempre alta: Possível falta de convergência
- p-value > 0.05: Diferença não significativa entre algoritmos
- Tempo > 5 min por ponto: Reduzir PSO_ITER ou AG_GENS

---

## Configuração Avançada

### Modificar Trajetória

Em `erick_dan.py`, altere:

```python
N_PASSOS = 50  # Aumentar pontos

# Nova trajetória (círculo 3D)
theta = np.linspace(0, 2*np.pi, N_PASSOS)
x_traj = 200 + 100 * np.cos(theta)
y_traj = 100 * np.sin(theta)
z_traj = 200 + 50 * np.sin(2*theta)
TRAJETORIA = np.column_stack([x_traj, y_traj, z_traj])
```

### Ajustar Pesos da Fitness

```python
PESO_POS = 1.0    # Erro de posição (peso maior = mais importante)
PESO_JTA = 0.05   # Penalidade articular (suavidade)
```

### Modificar Limites Articulares

```python
LIMITES_INF = np.array([-np.pi, -np.pi*110/180, ...])
LIMITES_SUP = np.array([ np.pi,  np.pi*155/180, ...])
```

---

## Para Publicação em Artigo

### Recomendações

1. **Rigor Estatístico**: Use `n_runs >= 30`
2. **Iterações Suficientes**: `PSO_ITER >= 200`, `AG_GENS >= 200`
3. **Múltiplas Trajetórias**: Teste 3-5 trajetórias diferentes
4. **Descrição**: Documente todos os parâmetros na seção de Métodos

### Citação de Resultados

Exemplo de texto para o artigo:

> *"Executamos 30 execuções independentes com 200 iterações por algoritmo. O PSO com warm-start (RMSE = 2.34 ± 0.12 mm) superou o AG (RMSE = 3.21 ± 0.45 mm) com significância estatística (p = 0.003, teste t-Student)."*

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'sko'"

```bash
pip install scikit-opt
```

### ❌ "Encoding UnicodeDecodeError"

O script foi otimizado para Windows. Se usar Linux/Mac, remova caracteres especiais dos prints.

### ❌ Script muito lento

- Reduza `n_runs` para 5-10 (teste)
- Reduza `PSO_ITER` e `AG_GENS` para 50-75
- Reduza `N_PASSOS` para 10

### ❌ Gráficos não salvam

Verifique permissões na pasta `analysis/results/`. Se necessário, crie manualmente:

```bash
mkdir -p analysis/results/{data,plots,statistics}
```

---

## Referências Bibliográficas

1. **Luo et al. (2022)** – "MOFOPSO: Multi-Objective Fitness Optimization Particle Swarm Optimization", Front. Neurorobot. 16:791796

2. **Farnese et al.** – "Comparative Analysis of GA vs PSO for 6-DoF Robot IK"

3. **Duymazlar & Gönenli** – "Genetic Algorithm vs PSO for Delta Robot IK", Cap. 17

4. **Kennedy & Eberhart (1995)** – Particle Swarm Optimization original

---

## Autor & Licença

Desenvolvido para fins acadêmicos e de pesquisa.

**Robô**: BIMO (6-DoF Educational Robot)  
**Aplicação**: Cinemática Inversa Comparativa  
**Status**: Pronto para Publicação  

---

## Perguntas Frequentes

**P: Posso usar esta análise para outros robôs?**  
R: Sim! Adapte os parâmetros DH em `erick_dan.py` e os limites articulares.

**P: Quanto tempo leva para executar tudo?**  
R: ~30-60 minutos com `n_runs=30` (recomendado para artigo).

**P: Os gráficos são de qualidade para publicação?**  
R: Sim, todos salvos em 300 DPI (PNG). Pronto para LaTeX e Word.

**P: Como combino resultados de várias execuções?**  
R: Todos os dados brutos estão em `dados_brutos.json`. Use Python/MATLAB para análise posterior.

---

**Versão**: 1.0  
**Data**: 2026-04-18  
**Status**: ✅ Pronto para uso
