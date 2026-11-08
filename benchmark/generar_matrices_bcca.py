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

def generar_y_guardar_matriz_bcca(n_filas, n_columnas, tipo_ruido, carpeta_destino):
    """
    Genera una matriz de datos continuos (float32) para el algoritmo BCCA.
    Modula los datos según un identificador de ruido para obtener 6 variantes por tamaño.
    """
    Path(carpeta_destino).mkdir(parents=True, exist_ok=True)
    
    nombre_archivo = f"{n_filas}x{n_columnas}_var{tipo_ruido}.csv"
    ruta_completa = os.path.join(carpeta_destino, nombre_archivo)
    
    if os.path.exists(ruta_completa):
        print(f"Saltando: {nombre_archivo} ya existe.")
        return

    print(f"Generando {nombre_archivo} para BCCA...")
    
    # Generamos datos continuos distribuidos en float32 (óptimo para GPU G5)
    # Variamos sutilmente la generación matemática para tener 6 datasets con propiedades distintas
    np.random.seed((n_filas * 100003 + n_columnas * 101 + tipo_ruido) % 2**32)
    if tipo_ruido == 1:
        matriz = np.random.rand(n_filas, n_columnas).astype(np.float32)
    elif tipo_ruido == 2:
        matriz = np.random.randn(n_filas, n_columnas).astype(np.float32) # Distribución normal
    else:
        matriz = (np.random.rand(n_filas, n_columnas) * (tipo_ruido * 0.5)).astype(np.float32)
        
    df = pd.DataFrame(matriz)
    df.to_csv(ruta_completa, sep=',', index=False, header=False)
    
    del matriz  
    del df

# --- Configuración BCCA Ajustada ---
# Mantenemos las 6 variantes numéricas por tamaño para conservar consistencia experimental
variantes_ruido = [1, 2, 3, 4, 5, 6]

cuadradas = [
    (50, 50),
    (75, 75),
    (100, 100),
    (200, 200),
    (400, 400),
    (800, 800),
    (1000, 1000)
]

no_cuadradas = [
    (80, 120),
    (100, 400),
    (200, 800),
    (120, 80),
    (400, 100),
    (800, 200)
]

print("Iniciando generación de datasets continuos optimizados para BCCA...")

for f, c in cuadradas:
    for v in variantes_ruido:
        generar_y_guardar_matriz_bcca(f, c, v, os.path.join(CARPETA, "matrices_bcca"))

for f, c in no_cuadradas:
    for v in variantes_ruido:
        generar_y_guardar_matriz_bcca(f, c, v, os.path.join(CARPETA, "matrices_bcca"))

total_datasets = (len(cuadradas) + len(no_cuadradas)) * len(variantes_ruido)
print(f"\n¡Proceso completado! Se han generado {total_datasets} datasets de BCCA en 'matrices_bcca'.")