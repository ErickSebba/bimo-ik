import time
import os
from pathlib import Path

monitor_dir = Path("resultados")
print("Monitorando geração de resultados...\n")

arquivos_esperados = {
    "convergencia": ["historico_convergencia.png", "erro_por_ponto.png"],
    "diversidade": ["diversidade_populacional.png", "diversidade_vs_erro.png"],
    "suavidade": ["perfis_articulares_pso.png", "perfis_articulares_pso_dfl.png", 
                  "perfis_articulares_ag.png", "comparacao_suavidade.png"],
    "estatistica": ["analise_estatistica.txt"],
    "warm_cold_start": ["resultado_warm_vs_cold.json", "warm_vs_cold_comparacao.png"],
    "dados_brutos": ["tabela_comparativa.json"]
}

for pasta, arquivos in arquivos_esperados.items():
    pasta_path = monitor_dir / pasta
    print(f"\n[{pasta}]")
    for arq in arquivos:
        arq_path = pasta_path / arq
        if arq_path.exists():
            size = arq_path.stat().st_size
            print(f"  ✓ {arq} ({size} bytes)")
        else:
            print(f"  ✗ {arq} (aguardando...)")

# Contar total de arquivos gerados
total = sum(len(list(Path(monitor_dir / pasta).glob("*"))) 
            for pasta in arquivos_esperados.keys())
print(f"\n{'='*50}")
print(f"Total de arquivos gerados: {total}")
print(f"{'='*50}")
