#!/usr/bin/env python3
"""
=============================================================================
  MÓDULO DE PLOTS EXTRAS E ANÁLISE VISUAL - ROBÔ BIMO (6-DOF)
  
  Responsável pela geração de gráficos comparativos de convergência,
  curvas de diversidade populacional e boxplots de suavidade/jerk.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from bimo.config import DIRS

def gerar_graficos_convergencia(res_pso, res_pso_dfl, res_ag):
    """Gera e salva as curvas de histórico de convergência média das runs."""
    plt.figure(figsize=(10, 6))
    
    algoritmos = [
        (res_pso, "PSO", "#1f77b4"), 
        (res_pso_dfl, "PSO-DFL", "#2ca02c"), 
        (res_ag, "AG", "#d62728")
    ]
    
    for res, nome, cor in algoritmos:
        if isinstance(res, list) and len(res) > 0:
            # Calcula a média da evolução de fitness entre todas as runs executadas
            historicos = [r.get("historico_convergencia", []) for r in res if "historico_convergencia" in r]
            if historicos:
                min_len = min(len(h) for h in historicos)
                curva_media = np.mean([h[:min_len] for h in historicos], axis=0)
                plt.plot(curva_media, label=nome, color=cor, linewidth=2.5)
        elif isinstance(res, dict) and "historico_convergencia" in res:
            plt.plot(res["historico_convergencia"], label=nome, color=cor, linewidth=2.5)
            
    plt.title("Histórico de Convergência Médio (Função Fitness)", fontsize=12, fontweight='bold')
    plt.xlabel("Iterações / Gerações", fontsize=10)
    plt.ylabel("Melhor Custo Encontrado (Escala Log)", fontsize=10)
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    path_salvar = DIRS["convergencia"] / "comparativo_convergencia.png"
    plt.savefig(path_salvar, dpi=300)
    plt.close()
    print(f"  [OK] Gráfico de convergência salvo em: {path_salvar}")


def gerar_graficos_diversidade(res_pso, res_pso_dfl, res_ag):
    """Gera e salva o comportamento do desvio padrão da população (Exploração vs Explotação)."""
    plt.figure(figsize=(10, 6))
    
    algoritmos = [
        (res_pso, "PSO", "#1f77b4"), 
        (res_pso_dfl, "PSO-DFL", "#2ca02c"), 
        (res_ag, "AG", "#d62728")
    ]
    
    for res, nome, cor in algoritmos:
        if isinstance(res, list) and len(res) > 0:
            historicos = [r.get("historico_diversidade", []) for r in res if "historico_diversidade" in r]
            if historicos and any(len(h) > 0 for h in historicos):
                min_len = min(len(h) for h in historicos)
                curva_media = np.mean([h[:min_len] for h in historicos], axis=0)
                plt.plot(curva_media, label=nome, color=cor, linewidth=2.5)
        elif isinstance(res, dict) and "historico_diversidade" in res:
            plt.plot(res["historico_diversidade"], label=nome, color=cor, linewidth=2.5)
            
    plt.title("Evolução da Diversidade Populacional (Desvio Padrão Interno)", fontsize=12, fontweight='bold')
    plt.xlabel("Iterações / Gerações", fontsize=10)
    plt.ylabel("Métrica de Diversidade da População", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    path_salvar = DIRS["diversidade"] / "comparativo_diversidade.png"
    plt.savefig(path_salvar, dpi=300)
    plt.close()
    print(f"  [OK] Gráfico de diversidade salvo em: {path_salvar}")


def gerar_graficos_suavidade(res_pso, res_pso_dfl, res_ag):
    """Gera análises de Boxplot comparando o Jerk Articular acumulado."""
    plt.figure(figsize=(8, 5))
    
    dados_jerk = []
    nomes = []
    
    algoritmos = [
        (res_pso, "PSO"), 
        (res_pso_dfl, "PSO-DFL"), 
        (res_ag, "AG")
    ]
    
    for res, nome in algoritmos:
        if isinstance(res, list):
            jerks = [r.get("jerk_articular", 0) for r in res if "jerk_articular" in r]
            if jerks:
                dados_jerk.append(jerks)
                nomes.append(nome)
                
    if dados_jerk:
        # Correção aqui: removendo o 'labels' de dentro do boxplot
        bp = plt.boxplot(dados_jerk, patch_artist=True,
                         boxprops=dict(facecolor="#e6f2ff", color="#1f77b4"),
                         medianprops=dict(color="#d62728", linewidth=2))
        
        # Colocando os labels corretamente no eixo X
        plt.xticks(range(1, len(nomes) + 1), nomes)
        
        plt.title("Distribuição do Jerk Articular Acumulado", fontsize=11, fontweight='bold')
        plt.ylabel("Jerk Total (Σ|Δ²θ|)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.3, axis='y')
        
        # O try-except vai segurar aquele aviso chato de margens do tight_layout
        try:
            plt.tight_layout()
        except UserWarning:
            pass
            
        path_salvar = DIRS["suavidade"] / "boxplot_jerk_articular.png"
        plt.savefig(path_salvar, dpi=300)
        plt.close()
        print(f"  [OK] Gráfico de suavidade mecânica salvo em: {path_salvar}")
        
def gerar_grafico_trajetoria(res_pso, res_pso_dfl, res_ag):
    """
    Gera o gráfico 3D comparativo mostrando o rastro do efetuador de cada 
    algoritmo contra a trajetória alvo de referência.
    """
    from bimo.config import TRAJETORIA
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plota a trajetória de referência (Alvo)
    ax.plot(TRAJETORIA[:, 0], TRAJETORIA[:, 1], TRAJETORIA[:, 2], 
            'k--', linewidth=2, label="Trajetória Alvo")
    
    algoritmos = [
        (res_pso, "PSO", "#1f77b4", "o"), 
        (res_pso_dfl, "PSO-DFL", "#2ca02c", "^"), 
        (res_ag, "AG", "#d62728", "s")
    ]
    
    for res, nome, cor, marcador in algoritmos:
        if isinstance(res, list) and len(res) > 0:
            # Pega o histórico de posições da primeira run válida para ilustrar no rastro 3D
            for r in res:
                if "historico_pos" in r and len(r["historico_pos"]) > 0:
                    pos = np.array(r["historico_pos"])
                    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                            color=cor, marker=marcador, markersize=4, 
                            linestyle="-", alpha=0.8, label=f"Rastro {nome}")
                    break
        elif isinstance(res, dict) and "historico_pos" in res:
            pos = np.array(res["historico_pos"])
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                    color=cor, marker=marcador, markersize=4, label=f"Rastro {nome}")

    ax.set_title("Rastreamento Espacial da Trajetória 3D - BIMO", fontsize=12, fontweight='bold')
    ax.set_xlabel("Eixo X (mm)")
    ax.set_ylabel("Eixo Y (mm)")
    ax.set_zlabel("Eixo Z (mm)")
    
    # Limites para dar foco na ação, ajustados conforme o config.py
    ax.set_xlim(100, 350)
    ax.set_ylim(-150, 150)
    ax.set_zlim(100, 300)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path_salvar = DIRS["trajetorias"] / "rastreamento_trajetoria_3d.png"
    plt.savefig(path_salvar, dpi=300)
    plt.close()
    print(f"  [OK] Gráfico de trajetória 3D salvo em: {path_salvar}")