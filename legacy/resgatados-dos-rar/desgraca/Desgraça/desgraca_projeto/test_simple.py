#!/usr/bin/env python3
"""
Teste simples para verificar se o código funcionacom Python 3.11 e RoboticsToolbox
"""

import numpy as np
import scipy
if not hasattr(scipy, "randn"):
    scipy.randn = np.random.randn

import roboticstoolbox as rtb
import spatialmath as sm
import time

print("=" * 70)
print("TESTE DE COMPATIBILIDADE: Python 3.11 + RoboticsToolbox")
print("=" * 70)

# 1. Testar RoboticsToolbox
print("\n✓ Testando RoboticsToolbox...")
try:
    bimo = rtb.DHRobot([
        rtb.RevoluteDH(d=184, a=0,   alpha=np.pi/2),
        rtb.RevoluteDH(d=0,   a=222, alpha=0),
        rtb.RevoluteDH(d=0,   a=205, alpha=0),
        rtb.RevoluteDH(d=0,   a=0,   alpha=-np.pi/2),
        rtb.RevoluteDH(d=0,   a=0,   alpha=np.pi/2),
        rtb.RevoluteDH(d=75,  a=0,   alpha=0)
    ], name="BIMO")
    print("  ✓ Robô BIMO definido com sucesso")
except Exception as e:
    print(f"  ✗ Erro ao definir robô: {e}")
    exit(1)

# 2. Testar cinemática direta
print("\n✓ Testando cinemática direta...")
try:
    q_test = np.zeros(6)
    T = bimo.fkine(q_test)
    print(f"  ✓ Cinemática direta calculada: pos = {T.t}")
except Exception as e:
    print(f"  ✗ Erro na cinemática direta: {e}")
    exit(1)

# 3. Testar múltiplas chamadas à cinemática (stress test)
print("\n✓ Teste de estresse (100 chamadas a fkine)...")
try:
    t0 = time.time()
    for i in range(100):
        q = np.random.uniform(-np.pi, np.pi, 6)
        T = bimo.fkine(q)
        pos = T.t
    t1 = time.time()
    print(f"  ✓ 100 cinemáticas calculadas em {t1-t0:.2f}s")
except Exception as e:
    print(f"  ✗ Erro durante stress test: {e}")
    exit(1)

# 4. Testar PSO simples
print("\n✓ Testando scikit-opt PSO...")
try:
    from sko.PSO import PSO
    
    def sphere(X):
        """Simples função esfera para teste de PSO."""
        if len(X.shape) == 1:
            return np.sum(X**2)
        else:
            return np.sum(X**2, axis=1)
    
    pso = PSO(func=sphere, n_dim=6, pop=10, max_iter=20,
              lb=-5*np.ones(6), ub=5*np.ones(6))
    best_x, best_y = pso.run()
    print(f"  ✓ PSO convergiu: best_y = {best_y[0]:.6f}")
except Exception as e:
    print(f"  ✗ Erro no PSO: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. Testar fitness computation
print("\n✓ Testando fitness com cinemática...")
try:
    alvo = np.array([100, 50, 300])
    q_anterior = np.zeros(6)
    
    def fitness_fn(q):
        try:
            T = bimo.fkine(q)
            pos_atual = T.t
            erro = np.linalg.norm(pos_atual - alvo)
            return erro
        except:
            return 1000  # penalidade para erros
    
    # Testar com alguns pontos
    for i in range(10):
        q = np.random.uniform(-np.pi, np.pi, 6)
        erro = fitness_fn(q)
    
    print(f"  ✓ Fitness computada com sucesso (último erro: {erro:.2f}mm)")
except Exception as e:
    print(f"  ✗ Erro na fitness: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✓ TODOS OS TESTES PASSARAM!")
print("=" * 70)
print("\nO ambiente está pronto para rodar o experimento completo.")
print("Executar: poetry run python ik_pso_vs_ag_bimo.py")
