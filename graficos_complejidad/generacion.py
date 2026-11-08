import numpy as np
import matplotlib.pyplot as plt
import os

# Directorio de salida
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "presentacion", "figuras", "complejidad")
os.makedirs(output_dir, exist_ok=True)

# Configuración visual minimalista
plt.style.use('seaborn-v0_8-whitegrid')
t_op = 1e-9 # Base teórica: 1 nanosegundo por operación

def redondear_a_cero(num):
    """Fuerza el redondeo a la decena más cercana para terminar siempre en 0"""
    return int(round(num, -1))

def generar_grafico(sizes, t_mejor, t_promedio, t_peor, titulo, filename):
    plt.figure(figsize=(6, 4.5))
    x = np.arange(len(sizes))
    
    # Curvas de complejidad de alta densidad (10.000 puntos)
    plt.plot(x, t_mejor, label='Mejor Caso', color='#10B981', linewidth=3)
    plt.plot(x, t_promedio, label='Caso Promedio', color='#F59E0B', linewidth=3)
    plt.plot(x, t_peor, label='Peor Caso', color='#EF4444', linewidth=3)
    
    plt.yscale('log')
    plt.title(titulo, fontsize=15, fontweight='bold', pad=12)
    plt.xlabel('Tamaño Dataset (Genes, Condiciones)', fontsize=12)
    plt.ylabel('Tiempo Estimado (segundos)', fontsize=12)
    
    # Extraer 5 etiquetas y forzar su redondeo a 0
    ticks_idx = np.linspace(0, len(sizes) - 1, 5, dtype=int)
    labels = [f"({redondear_a_cero(sizes[i][0])}, {redondear_a_cero(sizes[i][1])})" for i in ticks_idx]
    
    plt.xticks(ticks_idx, labels, rotation=25, ha='right')
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

# 1. BiBit (10.000 tuplas simultáneas)
n_bibit = np.linspace(1000, 100000, 10000)
m_bibit = np.linspace(50, 2000, 10000)
s_bibit = list(zip(n_bibit, m_bibit))

generar_grafico(s_bibit, 
                [n * m * t_op for n, m in s_bibit], 
                [n**2 * t_op for n, m in s_bibit], 
                [n**2 * m * t_op for n, m in s_bibit], 
                'BiBit: Coste Teórico', 'bibit_complejidad.png')

# 2. BCCA (10.000 tuplas simultáneas)
n_bcca = np.linspace(100, 2500, 10000)
m_bcca = np.linspace(10, 500, 10000)
s_bcca = list(zip(n_bcca, m_bcca))

generar_grafico(s_bcca, 
                [n**3 * t_op for n, m in s_bcca], 
                [n**4 * t_op for n, m in s_bcca], 
                [(n**4) * (m/10) * t_op for n, m in s_bcca], 
                'BCCA: Coste Teórico', 'bcca_complejidad.png')

# 3. Algoritmos NP-Hard (10.000 tuplas simultáneas)
# Rango ajustado para que el redondeo visual encaje a la perfección
n_np = np.linspace(10, 60, 10000)
m_np = np.linspace(10, 50, 10000) 
s_np = list(zip(n_np, m_np))

generar_grafico(s_np, 
                [n * m * t_op for n, m in s_np], 
                [n**3 * t_op for n, m in s_np], 
                [(2**n) * t_op for n, m in s_np], 
                'Algoritmos NP-Hard (General)', 'nphard_complejidad.png')