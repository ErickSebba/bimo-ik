import numpy as np
import roboticstoolbox as rtb

# =====================================================================
# 1. MODELAGEM DO ROBÔ BIMO (Cinemática)
# =====================================================================
# Aqui construímos o robô usando os parâmetros de Denavit-Hartenberg.
# SUBSTITUA os valores de 'd' (deslocamento) e 'a' (tamanho do elo)
# pelas medidas reais do BIMO (escolha trabalhar tudo em mm ou tudo em metros).

bimo = rtb.DHRobot([
    # rtb.RevoluteDH(d=100, a=0,   alpha=np.pi/2),  # Junta 1 (Base)
    # rtb.RevoluteDH(d=0,   a=150, alpha=0),        # Junta 2
    # rtb.RevoluteDH(d=0,   a=150, alpha=0),        # Junta 3
    # rtb.RevoluteDH(d=120, a=0,   alpha=-np.pi/2), # Junta 4
    # rtb.RevoluteDH(d=0,   a=0,   alpha=np.pi/2),  # Junta 5
    # rtb.RevoluteDH(d=50,  a=0,   alpha=0)         # Junta 6 (Efetuador/Garra)
    # d: deslocamento no eixo Z | a: comprimento do elo no eixo X | alpha: torção do eixo
    rtb.RevoluteDH(d=184, a=0,   alpha=np.pi/2),   # Junta 1 (Base - Rotação Horizontal)
    rtb.RevoluteDH(d=0,   a=222, alpha=0),         # Junta 2 (Ombro - Rotação Vertical)
    rtb.RevoluteDH(d=0,   a=205, alpha=0),         # Junta 3 (Cotovelo - Rotação Vertical)
    rtb.RevoluteDH(d=0,   a=0,   alpha=-np.pi/2),  # Junta 4 (Punho 1 - Elevação)
    rtb.RevoluteDH(d=0,   a=0,   alpha=np.pi/2),   # Junta 5 (Punho 2 - Torção lateral)
    rtb.RevoluteDH(d=75,  a=0,   alpha=0)          # Junta 6 (Flange/Garra - Giro da ferramenta)
    
], name="Braço Robótico BIMO")


# =====================================================================
# 2. FUNÇÃO DE CINEMÁTICA DIRETA (Forward Kinematics)
# =====================================================================
def cinematica_direta(q):
    """ 
    Recebe um array com 6 ângulos (em radianos) e retorna 
    a posição cartesiana [x, y, z] do efetuador final.
    """
    # fkine calcula a matriz de transformação homogênea (4x4)
    matriz_transformacao = bimo.fkine(q)

    # '.t' extrai apenas o vetor de translação (x, y, z), ignorando a rotação
    posicao_xyz = matriz_transformacao.t
    return posicao_xyz


# =====================================================================
# 3. A FUNÇÃO DE AVALIAÇÃO / FITNESS FUNCTION (O "Juiz")
# =====================================================================
def funcao_fitness(q, alvo_xyz):
    """
    Esta é a função que o AG, PSO e DE vão tentar minimizar (chegar a zero).
    Ela calcula a Distância Euclidiana entre onde a garra está agora e onde queremos que ela vá.
    """
    posicao_atual = cinematica_direta(q)

    # Distância Euclidiana: sqrt((x_alvo - x_atual)^2 + (y_alvo - y_atual)^2 + (z_alvo - z_atual)^2)
    erro_distancia = np.linalg.norm(posicao_atual - alvo_xyz)

    return erro_distancia


# =====================================================================
# TESTE DO DIA 1
# =====================================================================
if __name__ == "__main__":
    # 1. Mostra a tabela D-H do robô no terminal
    print("--- Configuração do BIMO ---")
    print(bimo)

    # 2. Testa a cinemática direta com todos os motores no ângulo Zero
    angulos_teste = [0, 0, 0, 0, 0, 0] # radianos
    posicao_garra = cinematica_direta(angulos_teste)
    print(f"\nPosição da garra com ângulos zerados (x, y, z): {posicao_garra}")

    # 3. Define um ponto alvo fictício no espaço para testar o "Juiz"
    alvo_desejado = np.array([200.0, 100.0, 150.0])

    # 4. Calcula o quão longe a garra está do alvo
    erro = funcao_fitness(angulos_teste, alvo_desejado)
    print(f"Erro atual (Distância até o alvo): {erro:.2f} unidades")