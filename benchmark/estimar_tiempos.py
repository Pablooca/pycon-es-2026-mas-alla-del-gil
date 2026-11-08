"""
Estimación de los tiempos que no se han podido medir (por límite de horas o
por no haberse ejecutado).
"""
import os
import re
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
MATRICES_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    "bibit": ("bibit_results.csv", "matrices_bibit", "bibit_estimaciones.csv", r"(\d+)x(\d+)_(\d+)\.csv"),
    "bcca": ("bcca_results.csv", "matrices_bcca", "bcca_estimaciones.csv", r"(\d+)x(\d+)_var(\d+)\.csv"),
}

def parse(name, pattern):
    m = re.fullmatch(pattern, os.path.basename(name))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

def features(rows, cols, param):
    return np.column_stack([np.ones(len(rows)), np.log(rows), np.log(cols), np.asarray(param, dtype=float)])

def main(alg):
    csv_in, matrices, csv_out, pattern = CONFIG[alg]
    df = pd.read_csv(os.path.join(BASE_DIR, csv_in))
    df["t"] = pd.to_numeric(df["Tiempo_Ejecucion"], errors="coerce")
    limit_h = df["Tiempo_Ejecucion"].astype(str).str.extract(r"(?:TIMEOUT|OMITIDO) \(>(\d+(?:\.\d+)?)h\)")[0].astype(float)
    df["limite_s"] = limit_h * 3600

    datasets = sorted(f for f in os.listdir(os.path.join(MATRICES_DIR, matrices)) if f.endswith(".csv"))
    rows_out = []
    
    for mode in df["Modo"].unique():
        real = df[(df["Modo"] == mode) & df["t"].notna() & (df["t"] > 0)]
        if len(real) < 4:
            continue
            
        dims = np.array([parse(n, pattern) for n in real["Nombre_Dataset"]])
        X = features(dims[:, 0], dims[:, 1], dims[:, 2])
        coef, *_ = np.linalg.lstsq(X, np.log(real["t"].to_numpy()), rcond=None)

        measured = set(real["Nombre_Dataset"])
        for name in datasets:
            if name in measured:
                continue
            r, c, p = parse(name, pattern)
            est = float(np.exp(features([r], [c], [p]) @ coef)[0])
            lim = df[(df["Modo"] == mode) & (df["Nombre_Dataset"] == name)]["limite_s"].max()
            
            if pd.notna(lim) and alg == "bcca":
                est = max(est, lim)
                
            rows_out.append({"Nombre_Dataset": name, "Modo": mode, "Tiempo_Ejecucion_Estimado": est})

    pd.DataFrame(rows_out).to_csv(os.path.join(BASE_DIR, csv_out), index=False)
    print(f"{len(rows_out)} estimaciones guardadas en {csv_out}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bibit")