# Desgraca Projeto

Projeto de cinemática e animação de um braço robótico usando `roboticstoolbox`.

## Requisitos

- Python 3.11
- Poetry
- VS Code (opcional, mas recomendado)

## Como rodar do zero

1. Instale o Python 3.11.
2. Instale o Poetry.
3. Abra esta pasta no VS Code.
4. No terminal, entre na pasta do projeto.
5. Instale as dependências:

	```bash
	poetry install
	```

6. Rode os scripts:

	```bash
	poetry run python coisa.py
	poetry run python cinematica_bimo.py
	```

## O que cada arquivo faz

- `coisa.py`: modela o BIMO, calcula cinemática e gera o GIF `movimento_bimo.gif`.
- `cinematica_bimo.py`: mostra a cinemática direta e a função de erro do robô.

## Saída esperada

- `coisa.py` imprime dados do robô no terminal e salva um GIF do movimento.
- `cinematica_bimo.py` imprime a tabela D-H, a posição inicial da garra e o erro até um alvo fixo.

## Observações

- O projeto foi preparado para Python 3.11.
- Se o Poetry reclamar de versão do Python, confirme que o interpretador 3.11 está instalado e ativo.
- Se o GIF não aparecer, verifique se o `matplotlib` e o `Pillow` foram instalados pelo `poetry install`.
