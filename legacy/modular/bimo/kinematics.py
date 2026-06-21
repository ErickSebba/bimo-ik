import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# MODELAGEM DO ROBÔ BIMO (parâmetros DH)
# ─────────────────────────────────────────────────────────────────────────────
DH_params = [
    [0, 184, 0,   np.pi/2],   # Junta 1
    [0, 0,   222, 0],         # Junta 2
    [0, 0,   205, 0],         # Junta 3
    [0, 0,   0,   -np.pi/2],  # Junta 4
    [0, 0,   0,   np.pi/2],   # Junta 5
    [0, 75,  0,   0]          # Junta 6
]

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE CINEMÁTICA DIRETA
# ─────────────────────────────────────────────────────────────────────────────
def dh_matrix(theta, d, a, alpha):
    """Matriz de transformacao DH homogênea."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def cinematica_direta(q):
    """Retorna posicao XYZ (mm) do efetuador."""
    T = np.eye(4)
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T = T @ dh_matrix(q[i] + theta, d, a, alpha)
    return T[:3, 3]

def obter_matriz_se3(q):
    """Retorna a matriz homogênea 4x4 completa."""
    T = np.eye(4)
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T = T @ dh_matrix(q[i] + theta, d, a, alpha)
    return T

def posicoes_dos_elos(q):
    """Retorna as posicoes de cada elo do robô."""
    T = np.eye(4)
    pontos = [T[:3, 3].copy()]
    for i in range(6):
        theta, d, a, alpha = DH_params[i]
        T_i = dh_matrix(q[i] + theta, d, a, alpha)
        T = T @ T_i
        pontos.append(T[:3, 3].copy())
    return np.array(pontos)

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE FITNESS
# ─────────────────────────────────────────────────────────────────────────────
def criar_fitness(alvo, q_anterior, peso_trans=1.0, peso_rot=0.5, peso_junta=0.005):
    """Cria funcao fitness para otimizacao."""
    alvo = np.asarray(alvo)
    if alvo.ndim == 1:
        posicao_alvo = alvo[:3]
        rotacao_alvo = None
    else:
        posicao_alvo = alvo[:3, 3]
        rotacao_alvo = alvo[:3, :3]
    
    def fitness_computada(q):
        posicao_atual = cinematica_direta(q)
        erro_cartesiano = np.linalg.norm(posicao_atual - posicao_alvo)
        
        if rotacao_alvo is not None:
            matriz_atual = obter_matriz_se3(q)
            rotacao_atual = matriz_atual[:3, :3]
            matriz_diferenca = np.dot(rotacao_alvo.T, rotacao_atual)
            traco = np.clip((np.trace(matriz_diferenca) - 1.0) / 2.0, -1.0, 1.0)
            erro_angular = np.arccos(traco)
        else:
            erro_angular = 0.0
        
        erro_continuo = np.linalg.norm(q - q_anterior)
        custo = (peso_trans * erro_cartesiano) + \
                (peso_rot * erro_angular * 100.0) + \
                (peso_junta * erro_continuo * 10.0)
        return custo
    
    return fitness_computada

def criar_fitness_dinamico(alvo, q_anterior, tempo=0, peso_trans=1.0, 
                          peso_rot=0.5, peso_junta=0.005, ruido_dinamico=0.1):
    """Cria funcao fitness com landscape dinâmico (PSO-DFL)."""
    alvo = np.asarray(alvo)
    if alvo.ndim == 1:
        posicao_alvo = alvo[:3]
    else:
        posicao_alvo = alvo[:3, 3]
    
    def fitness_dinamico(q):
        posicao_atual = cinematica_direta(q)
        erro_cartesiano = np.linalg.norm(posicao_atual - posicao_alvo)
        erro_continuo = np.linalg.norm(q - q_anterior)
        
        # Ruido temporal: cria landscape dinâmico
        ruido = ruido_dinamico * np.sin(tempo * 0.1) * np.random.randn()
        
        custo = (peso_trans * erro_cartesiano) + \
                (peso_junta * erro_continuo * 10.0) + ruido
        return custo
    
    return fitness_dinamico