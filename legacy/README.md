# BIMO: cinemática inversa, versões anteriores

Versões antigas do código de cinemática inversa do BIMO (6-DOF) que compara PSO, PSO-DFL e Algoritmo Genético: `v1-ik-roboticstoolbox`, `v3-auditoria-e-correcoes`, `modular`, `analise-comparativa-pso-ag`, os scripts soltos `ultimaesperanca*.py`, `main.py` e `plot.py`, e os arquivos resgatados dos `.rar` antigos.

## Contexto
Destino previsto: pasta `legacy/` dentro do repositório ativo `bimo-ik` (hoje em `Área de Trabalho\Bimo\Artigo2\bimo-ik`, que não foi tocado por ter alterações sem commit).

## Estrutura
- `analise-comparativa-pso-ag/` (31 arquivos)
- `modular/` (46 arquivos)
- `resgatados-dos-rar/` (126 arquivos)
- `scripts-e-notas-soltos/` (4 arquivos)
- `scripts-soltos/` (7 arquivos)
- `v1-ik-roboticstoolbox/` (22 arquivos)
- `v3-auditoria-e-correcoes/` (37 arquivos)

## Conteúdo
- 268 arquivos (64.3 MB no total).
- Tipos mais comuns: .png ×122, .py ×73, .txt ×21, .json ×20, .gif ×11, .md ×7, .toml ×4, .log ×3.
- Datas dos arquivos: 2026-04-14 a 2026-11-24.

## Como executar
- Python: `pip install -r requirements.txt` (na pasta que contém o arquivo).
- Python (pyproject): `pip install .` ou `poetry install`.
- Jupyter: `jupyter lab` e abrir os `.ipynb`.

## Estado e observações
Arquivado. **A linha do tempo das versões está em `LINHAGEM.md`.** Os nomes antigos das pastas (Desgraça, DesgraçaFinal) eram enganosos: o conteúdo é a IK do BIMO. Em 25/11, `coisa.py`, `teo.py` e `videoteste.py` ganharam nomes pelo conteúdo. Os arquivos de `resgatados-dos-rar` mantêm os nomes originais, por serem cópia fiel dos `.rar`.

## Origem dos arquivos
Reorganização feita entre 24 e 25/11/2026. De onde veio cada parte:
- `modular` ← `Área de Trabalho\Projetos-faculdade-codigo\Faculdade-Robotica\modular`
- `v1-ik-roboticstoolbox` ← `Área de Trabalho\Projetos-faculdade-codigo\Faculdade-Violao-Desgraca\Desgraça`
- `v3-auditoria-e-correcoes` ← `Área de Trabalho\Projetos-faculdade-codigo\Faculdade-Violao-Desgraca\DesgraçaFinal`
- `scripts-e-notas-soltos\2026-04-14_notas-do-artigo-ag-e-rnas-6dof.txt` ← `Área de Trabalho\Projetos-faculdade-codigo\Codigos-soltos\2026-04-14_notas-do-artigo-ag-e-rnas-6dof.txt`
- `scripts-e-notas-soltos\2026-04-30_ultimaesperanca-v2.py` ← `Área de Trabalho\Projetos-faculdade-codigo\Codigos-soltos\2026-04-30_ultimaesperanca-v2.py`
- `scripts-e-notas-soltos\2026-05-01_ultimaesperanca-v5.py` ← `Área de Trabalho\Projetos-faculdade-codigo\Codigos-soltos\2026-05-01_ultimaesperanca-v5.py`
- `scripts-e-notas-soltos\2026-09-28_bimo-base-relatorio-final-ic.md` ← `Área de Trabalho\Projetos-faculdade-codigo\Codigos-soltos\2026-09-28_bimo-base-relatorio-final-ic.md`
- `analise-comparativa-pso-ag` ← `Área de Trabalho\AnaliseComparativa-PSO-AG`
- `scripts-soltos\2026-04-16_ik-pso-vs-ag-bimo.py` ← `Área de Trabalho\ik_pso_vs_ag_bimo.py`
- `scripts-soltos\2026-04-30_main.py` ← `Área de Trabalho\main.py`
- `scripts-soltos\2026-04-30_plot.py` ← `Área de Trabalho\plot.py`
- `scripts-soltos\2026-04-18_ultimaesperanca.py` ← `Área de Trabalho\ultimaesperanca.py`
- `scripts-soltos\2026-04-18_ultimaesperanca-v2.py` ← `Área de Trabalho\ultimaesperanca_v2.py`
- `scripts-soltos\2026-04-30_ultimaesperanca-v4.py` ← `Área de Trabalho\ultimaesperanca_v4.py`
- `scripts-soltos\2026-05-01_ultimaesperanca-v5.py` ← `Área de Trabalho\ultimaesperanca_v5.py`
- `v1-ik-roboticstoolbox\Desgraça\desgraca_projeto\bimo_dh_animacao_rtb.py` ← `C:\Users\erick\Projetos\bimo-ik-legacy\desgraca-v1-ik\Desgraça\desgraca_projeto\coisa.py`
- `v1-ik-roboticstoolbox\Desgraça\desgraca_projeto\ik_pso_warmstart_vs_ag.py` ← `C:\Users\erick\Projetos\bimo-ik-legacy\desgraca-v1-ik\Desgraça\desgraca_projeto\teo.py`
- `v1-ik-roboticstoolbox\Desgraça\desgraca_projeto\teste_animacao_3d.py` ← `C:\Users\erick\Projetos\bimo-ik-legacy\desgraca-v1-ik\Desgraça\desgraca_projeto\videoteste.py`
- `v1-ik-roboticstoolbox` ← `C:\Users\erick\Projetos\bimo-ik-legacy\desgraca-v1-ik`
- `v3-auditoria-e-correcoes` ← `C:\Users\erick\Projetos\bimo-ik-legacy\desgraca-final-auditoria-v3`
- `resgatados-dos-rar/` ← arquivos próprios extraídos de `Desgraça.rar`, `DESFRAÇA_FINAL_PORFAVOR.rar`, `Desgraça2.rar` e `resultados.rar` (os `.rar` originais seguem em `_triagem`)

_Detalhes completos em `log_movimentos_fase1.csv` a `log_movimentos_fase5.csv` (pasta `testesClaudeCode`)._