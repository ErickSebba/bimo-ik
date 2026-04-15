poetry installimport numpy as np

import roboticstoolbox as rtb
import spatialmath as sm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def criar_bimo():
    return rtb.DHRobot([
        rtb.RevoluteDH(d=100, a=0, alpha=np.pi / 2),
        rtb.RevoluteDH(d=0, a=150, alpha=0),
        rtb.RevoluteDH(d=0, a=150, alpha=0),
        rtb.RevoluteDH(d=120, a=0, alpha=-np.pi / 2),
        rtb.RevoluteDH(d=0, a=0, alpha=np.pi / 2),
        rtb.RevoluteDH(d=50, a=0, alpha=0),
    ], name="Braço Robótico BIMO")


def cinematica_direta(robot, q):
    matriz_transformacao = robot.fkine(q)
    return matriz_transformacao.t


def funcao_fitness(robot, q, alvo_xyz):
    posicao_atual = cinematica_direta(robot, q)
    return np.linalg.norm(posicao_atual - alvo_xyz)


def gerar_trajetoria_simples(passos=20):
    qs = []
    for t in np.linspace(0.0, 2.0 * np.pi, passos):
        qs.append([
            0.35 * np.sin(t),
            0.25 * np.cos(t),
            -0.2 * np.sin(t),
            0.15 * np.cos(t),
            0.1 * np.sin(t),
            0.0,
        ])
    return np.array(qs)


def posicoes_dos_elos(robot, q):
    transformacao = sm.SE3()
    pontos = [transformacao.t]

    for link, angulo in zip(robot.links, q):
        transformacao = transformacao * link.A(angulo)
        pontos.append(transformacao.t)

    return np.array(pontos)


def salvar_animacao_gif(robot, trajetoria, nome_arquivo="movimento_bimo.gif", fps=20):
    caminhos = [posicoes_dos_elos(robot, q) for q in trajetoria]
    pontas = np.array([caminho[-1] for caminho in caminhos])
    pontos = np.concatenate(caminhos, axis=0)
    alvo_desejado = np.array([200.0, 100.0, 150.0])
    erros = np.linalg.norm(pontas - alvo_desejado, axis=1)

    limite_xy = np.max(np.abs(pontos[:, :2])) + 50
    limite_z = max(200, np.max(pontos[:, 2]) + 50)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    linha, = ax.plot([], [], [], "o-", lw=2, color="tab:blue")
    rastro, = ax.plot([], [], [], "-", lw=1.5, color="tab:orange", alpha=0.8)
    alvo, = ax.plot([], [], [], "x", markersize=10, color="tab:red")
    ponta_atual, = ax.plot([], [], [], "o", markersize=8, color="tab:green")

    ax.set_title("Movimento do BIMO")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-limite_xy, limite_xy)
    ax.set_ylim(-limite_xy, limite_xy)
    ax.set_zlim(0, limite_z)

    def atualizar(frame_index):
        pontos_frame = caminhos[frame_index]
        linha.set_data(pontos_frame[:, 0], pontos_frame[:, 1])
        linha.set_3d_properties(pontos_frame[:, 2])

        rastro_pontos = pontas[: frame_index + 1]
        rastro.set_data(rastro_pontos[:, 0], rastro_pontos[:, 1])
        rastro.set_3d_properties(rastro_pontos[:, 2])

        ponta_atual.set_data([rastro_pontos[-1, 0]], [rastro_pontos[-1, 1]])
        ponta_atual.set_3d_properties([rastro_pontos[-1, 2]])

        ax.set_title(
            f"Movimento do BIMO | Frame {frame_index + 1}/{len(caminhos)} | "
            f"Erro: {erros[frame_index]:.2f}"
        )

        return linha, rastro, alvo, ponta_atual

    alvo.set_data([alvo_desejado[0]], [alvo_desejado[1]])
    alvo.set_3d_properties([alvo_desejado[2]])

    animacao = FuncAnimation(fig, atualizar, frames=len(caminhos), interval=1000 / fps, blit=False)
    animacao.save(nome_arquivo, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return nome_arquivo


if __name__ == "__main__":
    bimo = criar_bimo()

    print("--- Configuração do BIMO ---")
    print(bimo)

    q_zero = np.zeros(6)
    pos_zero = cinematica_direta(bimo, q_zero)
    print(f"\nPosição da garra com ângulos zerados (x, y, z): {pos_zero}")

    alvo_desejado = np.array([200.0, 100.0, 150.0])
    erro_zero = funcao_fitness(bimo, q_zero, alvo_desejado)
    print(f"Erro atual (Distância até o alvo): {erro_zero:.2f} unidades")

    trajetoria = gerar_trajetoria_simples(passos=12)
    print("\nPrimeiros pontos da trajetória de teste:")
    for indice, q in enumerate(trajetoria[:5], start=1):
        pos = cinematica_direta(bimo, q)
        print(f"  {indice:02d}. q={np.round(q, 3)} -> pos={np.round(pos, 2)}")

    melhor_indice = min(range(len(trajetoria)), key=lambda i: funcao_fitness(bimo, trajetoria[i], alvo_desejado))
    melhor_q = trajetoria[melhor_indice]
    melhor_pos = cinematica_direta(bimo, melhor_q)
    melhor_erro = funcao_fitness(bimo, melhor_q, alvo_desejado)

    print("\nMelhor amostra da trajetória:")
    print(f"  q = {np.round(melhor_q, 3)}")
    print(f"  pos = {np.round(melhor_pos, 2)}")
    print(f"  erro = {melhor_erro:.2f} unidades")

    caminho_gif = salvar_animacao_gif(bimo, trajetoria, nome_arquivo="movimento_bimo.gif", fps=10)
    print(f"\nGIF salvo em: {caminho_gif}")