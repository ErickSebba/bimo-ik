# legacy/: versões anteriores da cinemática inversa do BIMO

Código das versões que vieram antes do pacote atual (`src/bimo_ik`), de abril a junho de 2026. Todas fazem cinemática inversa do BIMO (6 GDL) comparando metaheurísticas (PSO, PSO-DFL e algoritmo genético). Está aqui como histórico do método; não é mantido nem testado.

A linha do tempo de cada versão, com o que mudou de uma para outra, está em [`LINHAGEM.md`](LINHAGEM.md). As cópias idênticas que não entraram estão listadas em [`OMITIDOS.md`](OMITIDOS.md).

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `v1-ik-roboticstoolbox/` | primeira versão: modelo DH no Robotics Toolbox e comparativo PSO com *warm start* × AG |
| `analise-comparativa-pso-ag/` | pacote de análise comparativa (PSO, AG e PSO-DFL) com relatórios |
| `v3-auditoria-e-correcoes/` | revisão da v2 (`correcao-1/`) e a v3 com as correções aplicadas (`correcao-2/`) |
| `modular/` | refatoração em pacote com orquestrador (junho) |
| `resgatados-dos-rar/` | versões recuperadas de arquivos `.rar` antigos: `v1-ik-pso-ag/`, `v2-comparativo/`, `v4-comparativo-justo/` (com as variantes 2 e 3) e `resultados-avulsos/` |
| `scripts-e-notas-soltos/` | uma variante avulsa da v2 |

Os scripts `ik_comparativo_v1.py` a `ik_comparativo_v5.py` são as versões em arquivo único do comparativo; os números seguem a ordem da `LINHAGEM.md`.
