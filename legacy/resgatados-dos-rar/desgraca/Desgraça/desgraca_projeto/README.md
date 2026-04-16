# Desgraca Projeto

Projeto de cinemática e animação de um braço robótico usando `roboticstoolbox`.

## Requisitos

- Python 3.11
- Poetry
- VS Code (opcional, mas recomendado)

## Como instalar Python 3.11 e Poetry (Windows e Linux)

### Windows

1. Baixe o Python 3.11 no site oficial: https://www.python.org/downloads/
2. Execute o instalador do Python.
3. Marque a opção **Add Python to PATH**.
4. Clique em **Install Now** e finalize a instalação.
5. Feche e abra o terminal (PowerShell) novamente.
6. Verifique se o Python foi instalado:

	```powershell
	python --version
	```

7. Instale o Poetry (com o instalador oficial):

	```powershell
	(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
	```

8. Feche e abra o terminal novamente.
9. Verifique se o Poetry foi instalado:

	```powershell
	poetry --version
	```

### Linux

1. Atualize os pacotes do sistema (exemplo para Debian/Ubuntu):

	```bash
	sudo apt update
	sudo apt install -y python3.11 python3.11-venv python3-pip curl
	```

2. Verifique se o Python 3.11 foi instalado:

	```bash
	python3.11 --version
	```

3. Instale o Poetry (com o instalador oficial):

	```bash
	curl -sSL https://install.python-poetry.org | python3.11 -
	```

4. Adicione o Poetry ao PATH (caso necessário):

	```bash
	export PATH="$HOME/.local/bin:$PATH"
	```

5. Verifique se o Poetry foi instalado:

	```bash
	poetry --version
	```

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
