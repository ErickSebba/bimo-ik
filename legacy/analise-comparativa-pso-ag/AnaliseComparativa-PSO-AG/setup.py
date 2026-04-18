#!/usr/bin/env python3
"""
SETUP.PY - Configuração e Verificação do Framework
Verifique se tudo está instalado e pronto para usar
"""

import sys
import os

def verificar_instalacao():
    """Verifica se as dependências estão instaladas."""
    
    dependencias = {
        'numpy': 'NumPy - Computação numérica',
        'scipy': 'SciPy - Algoritmos científicos',
        'matplotlib': 'Matplotlib - Visualização',
        'seaborn': 'Seaborn - Gráficos estatísticos',
        'sko': 'scikit-opt - Otimização'
    }
    
    faltantes = []
    
    print("=" * 70)
    print("VERIFICANDO DEPENDÊNCIAS")
    print("=" * 70)
    
    for modulo, descricao in dependencias.items():
        try:
            __import__(modulo)
            print(f"✓ {modulo:20} {descricao}")
        except ImportError:
            print(f"✗ {modulo:20} {descricao}")
            faltantes.append(modulo)
    
    if faltantes:
        print("\n" + "=" * 70)
        print("INSTALANDO DEPENDÊNCIAS FALTANTES")
        print("=" * 70)
        os.system(f"pip install {' '.join(faltantes)}")
        return False
    
    return True


def verificar_estrutura():
    """Verifica se a estrutura de pastas foi criada."""
    
    pastas_esperadas = [
        'analysis',
        'analysis/algorithms',
        'analysis/metrics',
        'analysis/results',
        'analysis/results/plots',
        'analysis/results/statistics',
        'analysis/results/data'
    ]
    
    print("\n" + "=" * 70)
    print("VERIFICANDO ESTRUTURA DE PASTAS")
    print("=" * 70)
    
    tudo_ok = True
    for pasta in pastas_esperadas:
        if os.path.exists(pasta):
            print(f"✓ {pasta}")
        else:
            print(f"✗ {pasta} (criando...)")
            os.makedirs(pasta, exist_ok=True)
            tudo_ok = False
    
    return tudo_ok


def verificar_arquivos():
    """Verifica se os arquivos principais existem."""
    
    arquivos_esperados = [
        'test_integrated_analysis.py',
        'analysis/run_analysis.py',
        'analysis/README.md',
        'README_ANALISES.md'
    ]
    
    print("\n" + "=" * 70)
    print("VERIFICANDO ARQUIVOS PRINCIPAIS")
    print("=" * 70)
    
    tudo_ok = True
    for arquivo in arquivos_esperados:
        if os.path.exists(arquivo):
            tamanho_kb = os.path.getsize(arquivo) / 1024
            print(f"✓ {arquivo:40} ({tamanho_kb:.1f} KB)")
        else:
            print(f"✗ {arquivo}")
            tudo_ok = False
    
    return tudo_ok


def main():
    """Função principal."""
    
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                    SETUP DO FRAMEWORK                         ║")
    print("║          PSO vs AG vs PSO-DFL para Cinemática Inversa         ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Verificar dependências
    deps_ok = verificar_instalacao()
    
    # Verificar estrutura
    struct_ok = verificar_estrutura()
    
    # Verificar arquivos
    arquivos_ok = verificar_arquivos()
    
    # Resultado final
    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    
    if deps_ok and struct_ok and arquivos_ok:
        print("""
✓ TUDO ESTÁ PRONTO!

Próximo passo:
    python test_integrated_analysis.py

Aguarde 10-20 minutos para análises completas.
        """)
        return 0
    else:
        if not deps_ok:
            print("⚠ Instale as dependências faltantes")
        if not struct_ok:
            print("✓ Estrutura de pastas criada automaticamente")
        if not arquivos_ok:
            print("✗ Alguns arquivos estão faltando")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
