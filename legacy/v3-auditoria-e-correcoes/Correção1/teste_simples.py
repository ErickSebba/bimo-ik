#!/usr/bin/env python
"""Teste simples para validar as correções principais."""

print("Verificando código Python para erros de sintaxe...")

try:
    with open("ultimaesperanca_v2.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    # Compilar para verificar erros de sintaxe
    compile(code, "ultimaesperanca_v2.py", "exec")
    print("✓ Nenhum erro de sintaxe encontrado")
    
    # Verificações de texto (sem importar tudo)
    checks = [
        ("def criar_fitness(alvo, q_anterior, peso_trans=PESO_POS", "Fitness usa PESO_POS"),
        ("def criar_fitness_dinamico(alvo, q_anterior, tempo=0, peso_trans=PESO_POS", "DFL usa PESO_POS"),
        ("peso_junta=PESO_JTA", "Ambas usam PESO_JTA"),
        ("custo = (peso_trans * erro_cartesiano) + \\n                (peso_junta * erro_continuo * 10.0)", "DFL sem ruído"),
        ("for i in range(self.pop):\n            self.y_best[i] = self.func(self.X_best[i])", "atualizar_fitness reavalia pbest"),
        ("rastreador = PSORastreador(juiz, dim=6, pop=PSO_POP", "rodar_pso usa PSORastreador"),
        ("usar_warmstart=True", "Funções têm parâmetro usar_warmstart"),
    ]
    
    for check_text, desc in checks:
        if check_text in code:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc}")
    
    print("\n✓ CÓDIGO COMPILADO COM SUCESSO!")
    print("→ Pronto para executar experimento completo")
    
except SyntaxError as e:
    print(f"✗ ERRO DE SINTAXE: {e}")
except Exception as e:
    print(f"✗ ERRO: {e}")
