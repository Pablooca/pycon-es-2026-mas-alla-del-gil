import numpy as np
import pandas as pd
import os
from pathlib import Path

# Las matrices de la primera ejecución ya están en esta carpeta (matrices_bibit / matrices_bcca)
# y este script NO las sobrescribe: solo genera las que falten. La semilla fija hace que, si
# hubiera que generarlas de cero, salgan siempre iguales (las originales no tenían semilla).
# La semilla depende solo del tamaño y la variante, así que da igual el orden o si se
# generan en varias sesiones: cada matriz sale siempre idéntica.
CARPETA = os.path.dirname(os.path.abspath(__file__))

def generar_y_guardar_matriz_bibit(n_filas, n_columnas, p_unos, carpeta_destino):
    """
    Genera una matriz binaria (0 y 1) para el algoritmo BiBit y la guarda en CSV.
    """
    Path(carpeta_destino).mkdir(parents=True, exist_ok=True)
    
    porcentaje_entero = int(p_unos * 100)
    nombre_archivo = f"{n_filas}x{n_columnas}_{porcentaje_entero}.csv"
    ruta_completa = os.path.join(carpeta_destino, nombre_archivo)
    
    if os.path.exists(ruta_completa):
        print(f"Saltando: {nombre_archivo} ya existe.")
        return

    print(f"Generando {nombre_archivo}...")
    
    # Generación eficiente binaria en int8
    np.random.seed((n_filas * 100003 + n_columnas * 101 + porcentaje_entero) % 2**32)
    matriz = (np.random.rand(n_filas, n_columnas) < p_unos).astype(np.int8)
    
    df = pd.DataFrame(matriz)
    df.to_csv(ruta_completa, sep=',', index=False, header=False)
    
    del matriz  
    del df

# --- Configuración BiBit ---
probabilidades = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

cuadradas = [
    (1500, 1500),
    (3000, 3000),
    (5000, 5000),
    (10000, 10000)
]

no_cuadradas = [
    (5000, 1000),
    (15000, 1000),
    (30000, 800),
    (1000, 5000),
    (1000, 15000),
    (800, 30000)
]

print("Iniciando generación de datasets para BiBit...")

# Ambas se guardan dentro del directorio local 'matrices_bibit'
for f, c in cuadradas:
    for p in probabilidades:
        generar_y_guardar_matriz_bibit(f, c, p, os.path.join(CARPETA, "matrices_bibit"))

for f, c in no_cuadradas:
    for p in probabilidades:
        generar_y_guardar_matriz_bibit(f, c, p, os.path.join(CARPETA, "matrices_bibit"))

print("\n¡Proceso completado! Todos los datasets de BiBit están en 'matrices_bibit'.")