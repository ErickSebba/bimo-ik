# Mapeamento do Código Atual - Robô BIMO (6-DOF)

Este documento descreve a organização modular do código de cinemática inversa do braço robótico BIMO, migrado de um script único para uma arquitetura profissional de pacotes.

## 1. Módulo Core: `bimo/kinematics.py`
Responsável pela modelagem matemática do braço robótico sem algoritmos de otimização.
- **Parâmetros DH / Geometria:** Definição das dimensões dos elos do BIMO.
- **`fkine(q)`**: Função de Cinemática Direta. Recebe o vetor de ângulos das juntas `q` e retorna a pose cartesiana do efetuador.
- **Funções de Transformação Homogênea:** Matrizes de rotação e translação por junta.

## 2. Módulo de Algoritmos: `bimo/algorithms/`
Contém as heurísticas de busca para resolução da cinemática inversa ao longo de trajetórias.
- **`pso.py`**: Implementação do PSO padrão. Resolve a cinemática inversa reinicializando a população a cada ponto da trajetória (Cold-Start).
- **`pso_dfl.py`**: Implementação do PSO-DFL (Dynamic Fitness Landscape). Incorpora a continuidade temporal reaproveitando o estado final do passo anterior (Warm-Start / Doping) para suavização do movimento.
- **`ga.py`**: Implementação do Algoritmo Genético (AG/GA) adaptado para a busca de configurações articulares.

## 3. Módulo de Utilidades e Métricas: `bimo/utils/`
Funções de suporte, tratamento de dados e cálculo de critérios de qualidade para artigos.
- **`metrics.py`**:
  - `rmse(...)`: Erro quadrático médio da posição do efetuador.
  - `smoothness(...)` / `jerk(...)`: Cálculo do jerk articular e suavidade da trajetória ($\sum |\Delta \theta|$).
  - `cohen_d(...)`: Cálculo do tamanho do efeito para validação estatística.
- **`paths.py`**: Gerenciamento e criação automática da estrutura de pastas de resultados (`resultados/dados_brutos/`, `resultados/graficos/`, etc.).

## 4. Script Principal: `main.py`
Ponto de entrada do sistema. Coordena a execução dos experimentos, gera os relatórios, monta tabelas comparativas e plota os gráficos estatísticos e de convergência.