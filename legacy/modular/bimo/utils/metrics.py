import numpy as np

def converter_para_float(valor):
    """Converte tipos do numpy para float nativo do Python."""
    try:
        if isinstance(valor, np.ndarray):
            return valor.tolist()
        return float(valor)
    except (TypeError, ValueError):
        return valor

def calcular_suavidade(historico_q):
    """
    Calcula a suavidade total da trajetória.
    Métrica: Soma de todas as variações angulares absolutas entre passos consecutivos.
    """
    q = np.array(historico_q)
    
    if len(q) < 2:
        return 0.0
    
    delta_q = np.diff(q, axis=0)
    suavidade = np.sum(np.abs(delta_q))
    
    return float(suavidade)

def calcular_jerk(historico_q, dt=1.0):
    """
    Calcula o Jerk articular médio (derivada da aceleração).
    """
    q = np.array(historico_q)
    
    if len(q) < 4:
        return 0.0
        
    v = np.diff(q, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    jerk = np.diff(a, axis=0) / dt
    
    jerk_medio = np.mean(np.abs(jerk))
    
    return float(jerk_medio)