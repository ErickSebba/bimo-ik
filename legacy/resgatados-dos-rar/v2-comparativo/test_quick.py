# Teste rápido sem Wait
import subprocess
import sys

print("Iniciando teste rápido... (pode levar ~2-3 minutos)")
print("=" * 70)

# Executar em background
proc = subprocess.Popen(
    [sys.executable, "ultimaesperanca_v2.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Mostrar output conforme aparece
for linha in proc.stdout:
    print(linha, end='', flush=True)

proc.wait()
print("\n" + "=" * 70)
print(f"Processo finalizado com código: {proc.returncode}")
