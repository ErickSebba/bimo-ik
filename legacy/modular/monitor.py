import os
from pathlib import Path

def verificar_estrutura():
    base_dir = Path("resultados")
    
    # Mapeamento do que precisa existir
    estrutura_esperada = {
        "Dados Brutos": [
            base_dir / "dados_brutos" / "tabela_comparativa.json"
        ],
        "Gráficos de Convergência": [
            base_dir / "graficos" / "convergencia" / "historico_convergencia.png",
            base_dir / "graficos" / "convergencia" / "erro_por_ponto.png"
        ],
        "Análise Estatística": [
            base_dir / "graficos" / "estatistica" / "analise_estatistica.txt"
        ],
        "Métricas de Suavidade": [
            base_dir / "graficos" / "suavidade" / "perfis_articulares_pso.png",
            base_dir / "graficos" / "suavidade" / "perfis_articulares_pso-dfl.png",
            base_dir / "graficos" / "suavidade" / "perfis_articulares_ag.png",
            base_dir / "graficos" / "suavidade" / "comparacao_suavidade.png"
        ],
        "Métricas de Diversidade": [
            base_dir / "graficos" / "diversidade" / "diversidade_populacional.png",
            base_dir / "graficos" / "diversidade" / "diversidade_vs_erro.png"
        ],
        "Warm-Start vs Cold-Start": [
            base_dir / "graficos" / "warm_cold" / "resultado_warm_vs_cold.json",
            base_dir / "graficos" / "warm_cold" / "warm_vs_cold_comparacao.png"
        ]
    }
    
    print("\n" + "="*60)
    print("         MONITOR DE VERIFICAÇÃO DE RESULTADOS")
    print("="*60)
    
    tudo_ok = True
    for bloco, caminhos in estrutura_esperada.items():
        print(f"\n[ {bloco} ]")
        for caminho in caminhos:
            if caminho.exists():
                print(f"  ✓ {caminho.name:<35} | Encontrado")
            else:
                print(f"  ✗ {caminho.name:<35} | FALTANDO")
                tudo_ok = False
                
    print("\n" + "="*60)
    if tudo_ok:
        print("  ✓ SUCESSO: Todos os arquivos do experimento estão prontos!")
    else:
        print("  ✗ ATENÇÃO: Alguns arquivos ainda estão faltando.")
    print("="*60 + "\n")

if __name__ == "__main__":
    verificar_estrutura()