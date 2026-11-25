# Linha do tempo das versões da IK do BIMO

Montada em 25/11/2026 comparando o **hash** (conteúdo idêntico) e o cabeçalho de cada script. As datas são de modificação dos arquivos. Todos fazem cinemática inversa do BIMO (6 GDL) comparando metaheurísticas.

| Data | Versão (arquivo) | Tamanho | O que muda | Onde está (cópias idênticas) |
|---|---|---:|---|---|
| 15/04 | `bimo_dh_animacao_rtb.py` (antes `coisa.py`) | 4,9 KB | modelo DH do BIMO no roboticstoolbox e animação | `v1-ik-roboticstoolbox\...\desgraca_projeto` (+ versão quase igual em `resgatados-dos-rar\desgraca\...\Codigos_velhos\coisa.py`) |
| 16/04 | `ik_pso_vs_ag_bimo.py` | 47,6 KB | primeiro comparativo PSO com warm-start x AG | `scripts-soltos` = `resgatados-dos-rar\desgraca\...` |
| 17/04 | `ik_pso_warmstart_vs_ag.py` (antes `teo.py`) | 38,3 KB | variante enxuta do mesmo comparativo | `v1-ik-roboticstoolbox\...\desgraca_projeto` |
| 17-18/04 | `erick_dan.py` + `run_analysis.py` + `test_integrated_analysis.py` | 51 / 27 / 48 KB | pacote de análise comparativa (PSO, AG, PSO-DFL) com relatórios | `analise-comparativa-pso-ag` |
| 18/04 | `ultimaesperanca.py` | 66,4 KB | comparativo PSO warm-start x AG ampliado | `scripts-soltos` = `resgatados-dos-rar\desgraca2` |
| 18/04 | `ultimaesperanca_v2.py` | 51,0 KB | "análise comparativa exaustiva": PSO, PSO-DFL e AG | `scripts-soltos` = `modular` = `resgatados-dos-rar\desgraca2` |
| 30/04 | `ultimaesperanca_v2.py` corrigido | 62,0 KB | 3 correções da auditoria | `v3-auditoria-e-correcoes\Correção1` |
| 30/04 | `2026-04-30_ultimaesperanca-v2.py` | 55,6 KB | outra variante da v2 corrigida (hash diferente) | `scripts-e-notas-soltos` |
| 30/04 | `ultimaesperanca_v3.py` | 67,8 KB | "v3.0": as 3 correções aplicadas | `v3-auditoria-e-correcoes\Correção2` |
| 30/04 | `ultimaesperanca_v4.py` | 74,0 KB | "v4.0": comparativo justo PSO \| PSO-DFL \| AG | `scripts-soltos` = `resgatados-dos-rar\desfraca-final-porfavor` |
| 30/04 | `main.py` + `plot.py` | 33,4 + 8,1 KB | v4.0 separada em execução (`main`) e gráficos (`plot`); `main.py` faz `import plot` | `scripts-soltos` = `resgatados-dos-rar\...\Desgraça_Final_3` |
| 01/05 | `ultimaesperanca_v5.py` | 80,1 KB | evolução da v4.0 (o cabeçalho ainda diz v4.0) | `scripts-soltos` = `scripts-e-notas-soltos` = `resgatados-dos-rar\...\DesgraçaFinal2` |
| 20/06 | `modular\` (`main.py`, `bimo\config.py`, `bimo\algorithms\{pso,ga}.py`, `kinematics.py`) | — | refatoração em pacote com orquestrador | `modular` |
| out/2026 | repositório `bimo-ik` (pacote `src/bimo_ik`) | — | versão atual, usada no artigo do LARS 2026 | `Área de Trabalho\Bimo\Artigo2\bimo-ik` (não movido) |

## Leitura recomendada
- **Para reproduzir resultados antigos:** `scripts-soltos\main.py` + `plot.py` (v4.0) ou `ultimaesperanca_v5.py` (a mais completa em arquivo único).
- **Para a história do método:** v2 → auditoria (`v3-auditoria-e-correcoes\README.md`) → v3 → v4.
- **Duplicatas exatas** (mesmo hash) estão marcadas com `=` na tabela. Ao mover este material para `bimo-ik/legacy/`, basta manter uma cópia de cada.

## Ainda em aberto
- `teo2.py` está vazio (0 bytes).
- `teste_animacao_3d.py` (antes `videoteste.py`) é um trecho que depende de funções definidas em outro arquivo: não roda sozinho.
