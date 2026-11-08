"""
Benchmark v2 de BiBit y BCCA (6 modos) con el código optimizado.

No toca nada de ../benchmark. Lee las matrices de ./matrices_bibit y ./matrices_bcca (las mismas
de la primera ejecución) y escribe en ./resultados.

Uso:
    python run_benchmark.py bibit                 # ejecución completa
    python run_benchmark.py bcca --limite-horas 2
    python run_benchmark.py bibit --modos 2,3,5,6 # solo algunos modos
    python run_benchmark.py bcca --smoke          # prueba rápida (Colab / arranque en AWS)

Cambios respecto a ../benchmark/run_*.py:
  - Cada ejecución (matriz, modo) va en un proceso nuevo con límite de tiempo,
    también los modos paralelos y GPU. Un fallo de CUDA o de memoria no tumba el resto.
  - Antes de cronometrar se hace un calentamiento con una matriz diminuta en el
    mismo proceso: la compilación JIT de Numba y la creación del contexto CUDA
    no cuentan en el tiempo (antes sí contaban en la primera matriz de cada modo).
  - Si un proceso muere sin devolver nada se guarda ERROR, no un tiempo de 0 s.
  - BCCA no recurre en silencio a bs.bcca(dataset) (modo secuencial) si falla la llamada.
  - Si un modo supera el límite en una matriz, se omiten para ese modo las matrices
    con igual o más filas y columnas (serían aún más lentas); se estiman después.
  - Se avisa si los modos no devuelven el mismo número de biclústeres.
  - Se guarda el entorno (CPU, GPU, versiones) en resultados/entorno_<alg>.json.
  - El CSV tiene las mismas columnas que antes, así que estudio_estadistico.py lo lee igual.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import queue as queue_mod
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALGORITMOS = {
    "bibit": {
        "matrices": os.path.join(BASE_DIR, "matrices_bibit"),
        "patron": r"(\d+)x(\d+)_(\d+)\.csv",
        "modos": {1: "Python Secuencial", 2: "Numba Paralelo CPU", 3: "Numba GPU",
                  4: "Cython Secuencial", 5: "Cython OpenMP", 6: "Cython GPU"},
    },
    "bcca": {
        "matrices": os.path.join(BASE_DIR, "matrices_bcca"),
        "patron": r"(\d+)x(\d+)_var(\d+)\.csv",
        "modos": {1: "CPU Secuencial", 2: "Numba Paralelo CPU", 3: "Numba GPU",
                  4: "Cython Secuencial", 5: "Cython OpenMP", 6: "Cython GPU"},
    },
}

# Mismos parámetros que el benchmark original
BIBIT_MIN_ROWS = 2
BIBIT_MIN_COLS = 160
BIBIT_DEBUG = True
BCCA_THRESHOLD = 0.7
BCCA_MIN_COLS = 3

CAMPOS = ["Nombre_Dataset", "Modo", "Numero_Resultados", "Tiempo_Ejecucion", "Fecha"]


# ---------------------------------------------------------------- trabajo en el proceso hijo
def _ejecutar(alg, dataset, mode, bibit_min_cols):
    import bioscience as bs
    if alg == "bibit":
        modelos = bs.bibit(dataset, cMnr=BIBIT_MIN_ROWS, cMnc=bibit_min_cols, mode=mode, debug=BIBIT_DEBUG)
    else:
        modelos = bs.bcca(dataset, correlationThreshold=BCCA_THRESHOLD, minCols=BCCA_MIN_COLS, mode=mode, debug=False)
    return len(next(iter(modelos)).results) if modelos else 0


def _calentar(alg, mode):
    """Compila Numba y crea el contexto CUDA con una matriz diminuta (no se cronometra)."""
    import numpy as np
    import bioscience as bs
    rng = np.random.default_rng(0)
    if alg == "bibit":
        data = (rng.random((16, 200)) < 0.5).astype(np.int64)
        bs.bibit(bs.Dataset(data), cMnr=2, cMnc=2, mode=mode, debug=BIBIT_DEBUG)
    else:
        data = rng.random((12, 10))
        bs.bcca(bs.Dataset(data), correlationThreshold=BCCA_THRESHOLD, minCols=BCCA_MIN_COLS, mode=mode, debug=False)


def _silenciar_stdout():
    # La librería imprime un resumen por ejecución; se descarta para no llenar el log
    sys.stdout.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)


def worker(alg, path, mode, bibit_min_cols, cola):
    try:
        import warnings
        import bioscience as bs
        # Aviso de "GPU infrautilizada" en matrices pequeñas: no es un error
        warnings.filterwarnings("ignore", message=".*GPU under-utilization")
        _silenciar_stdout()
        _calentar(alg, mode)
        dataset = bs.load(db=path, separator=",", index_gene=-1, head=None, naFilter=False)
        t0 = time.perf_counter()
        n = _ejecutar(alg, dataset, mode, bibit_min_cols)
        t = time.perf_counter() - t0
        cola.put(("ok", n, t))
    except BaseException as e:  # incluye MemoryError
        msg = " ".join(f"{type(e).__name__}: {e}".split())
        cola.put(("error", msg[:300], None))


# ---------------------------------------------------------------- utilidades del proceso principal
def leer_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def guardar(path, fila):
    nuevo = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def log(path, msg):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def comando(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def guardar_entorno(path):
    info = {"fecha": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version, "plataforma": platform.platform(),
            "cpus_logicas": os.cpu_count(),
            "cpu": comando(["bash", "-c", "lscpu | grep 'Model name' || sysctl -n machdep.cpu.brand_string"]),
            "memoria": comando(["bash", "-c", "free -h | head -2"]),
            "gpu": comando(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            "nvcc": comando(["bash", "-c", "nvcc --version | tail -1"])}
    for mod in ("numpy", "numba", "Cython", "pandas", "bioscience"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
            if mod == "bioscience":
                info["bioscience_ruta"] = os.path.dirname(m.__file__)
        except Exception as e:
            info[mod] = f"no disponible ({e})"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def comprobar_version_optimizada():
    """Aborta si se importa la librería original en lugar de la optimizada."""
    from bioscience.dataMining.biclustering import BiBit, Bcca
    if not hasattr(BiBit, "_SequentialEngine") or not hasattr(Bcca, "_bccaNumbaGpu"):
        sys.exit("ERROR: se está importando bioscience ORIGINAL. Instala bioscience-0.1.x (./rebuild.sh).")
    faltan = [m for m, mod in (("bibit_cython", BiBit.bibit_cython), ("BiBit_gpu_cython", BiBit.bibit_gpu_cython),
                               ("bcca_cython", Bcca.bcca_cython), ("Bcca_gpu_cython", Bcca.bcca_gpu_cython))
              if mod is None]
    return faltan


def ejecutar_uno(alg, path, mode, limite_s, bibit_min_cols):
    """Devuelve (estado, n, tiempo): estado = ok | timeout | error."""
    ctx = mp.get_context("spawn")
    cola = ctx.Queue()
    p = ctx.Process(target=worker, args=(alg, path, mode, bibit_min_cols, cola))
    p.start()
    p.join(timeout=limite_s)
    if p.is_alive():
        p.terminate()
        p.join(10)
        if p.is_alive():
            p.kill()
            p.join()
        return "timeout", 0, None
    try:
        estado, n, t = cola.get(timeout=5)
    except queue_mod.Empty:
        return "error", 0, f"proceso terminado sin resultado (código {p.exitcode}; posible falta de RAM)"
    if estado == "error":
        return "error", 0, n
    return "ok", n, t


# ---------------------------------------------------------------- matrices de la prueba rápida
def generar_smoke(alg, carpeta):
    import numpy as np
    os.makedirs(carpeta, exist_ok=True)
    rng = np.random.default_rng(1)
    if alg == "bibit":
        tamanos = [(300, 400, 30), (400, 300, 60), (500, 700, 40)]
        for r, c, p in tamanos:
            m = (rng.random((r, c)) < p / 100).astype(np.int8)
            np.savetxt(os.path.join(carpeta, f"{r}x{c}_{p}.csv"), m, delimiter=",", fmt="%d")
    else:
        for r, c, v in [(40, 30, 1), (60, 50, 2), (80, 120, 3)]:
            m = rng.random((r, c)) if v != 2 else rng.standard_normal((r, c))
            np.savetxt(os.path.join(carpeta, f"{r}x{c}_var{v}.csv"), m.astype(np.float32), delimiter=",", fmt="%.7g")
    return carpeta


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("algoritmo", choices=ALGORITMOS)
    ap.add_argument("--limite-horas", type=float, default=2.0, help="límite por ejecución (todos los modos)")
    ap.add_argument("--modos", default="1,2,3,4,5,6")
    ap.add_argument("--matrices", default=None, help="carpeta de matrices (por defecto ./matrices_<alg>)")
    ap.add_argument("--filtro", default=None, help="regex sobre el nombre de la matriz, p. ej. '^1500x1500'")
    ap.add_argument("--salida", default=None,
                    help="carpeta de resultados (por defecto ./resultados); en Colab, una carpeta de Google Drive")
    ap.add_argument("--smoke", action="store_true", help="prueba rápida con matrices pequeñas generadas al vuelo")
    args = ap.parse_args()

    alg = args.algoritmo
    cfg = ALGORITMOS[alg]
    modos = [int(m) for m in args.modos.split(",")]
    limite_s = args.limite_horas * 3600
    etiqueta_limite = f"{args.limite_horas:g}h"
    # En la prueba rápida las matrices son pequeñas: con 160 columnas no saldría ningún biclúster
    bibit_min_cols = 40 if args.smoke else BIBIT_MIN_COLS

    salida = os.path.join(BASE_DIR, "resultados_smoke") if args.smoke else (args.salida or os.path.join(BASE_DIR, "resultados"))
    os.makedirs(salida, exist_ok=True)
    csv_path = os.path.join(salida, f"{alg}_results.csv")
    log_path = os.path.join(salida, f"{alg}.log")

    faltan = comprobar_version_optimizada()
    entorno = guardar_entorno(os.path.join(salida, f"entorno_{alg}.json"))
    log(log_path, f"=== {alg.upper()} | límite {etiqueta_limite} | modos {modos} | GPU: {entorno['gpu'] or 'ninguna'} ===")
    if faltan:
        log(log_path, f"AVISO: módulos compilados no disponibles: {', '.join(faltan)} (esos modos darán ERROR)")

    if args.smoke:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        carpeta = generar_smoke(alg, os.path.join(salida, "matrices"))
    else:
        carpeta = os.path.abspath(args.matrices or cfg["matrices"])

    patron = re.compile(cfg["patron"])
    archivos = []
    for f in os.listdir(carpeta):
        m = patron.fullmatch(f)
        if m and (not args.filtro or re.search(args.filtro, f)):
            r, c, p = map(int, m.groups())
            archivos.append((r * c, r, c, p, f))
    archivos.sort()
    if not archivos:
        sys.exit(f"No hay matrices en {carpeta}")

    # Checkpoint: lo ya guardado (ok, timeout, error u omitido) no se repite
    previas = leer_csv(csv_path)
    hechos = {(f["Nombre_Dataset"], f["Modo"]) for f in previas}
    # Matrices en las que cada modo superó el límite (para omitir las mayores)
    techo = {m: [] for m in modos}
    for f in previas:
        for m in modos:
            if f["Modo"] == cfg["modos"][m] and f["Tiempo_Ejecucion"].startswith("TIMEOUT"):
                r, c, _ = map(int, patron.fullmatch(f["Nombre_Dataset"]).groups())
                techo[m].append((r, c))

    pendientes = sum((f, cfg["modos"][m]) not in hechos for *_, f in archivos for m in modos)
    log(log_path, f"{len(archivos)} matrices en {carpeta}; {pendientes} ejecuciones pendientes")
    inicio = time.time()

    for idx, (_, r, c, _, nombre) in enumerate(archivos, 1):
        log(log_path, f"[{idx}/{len(archivos)}] {nombre}")
        for m in modos:
            modo = cfg["modos"][m]
            if (nombre, modo) in hechos:
                continue
            fila = {"Nombre_Dataset": nombre, "Modo": modo, "Numero_Resultados": 0,
                    "Fecha": datetime.now().isoformat(timespec="seconds")}
            if any(r >= tr and c >= tc for tr, tc in techo[m]):
                fila["Tiempo_Ejecucion"] = f"OMITIDO (>{etiqueta_limite})"
                guardar(csv_path, fila)
                log(log_path, f"   {modo:<20} omitido: superó el límite en una matriz menor")
                continue

            estado, n, t = ejecutar_uno(alg, os.path.join(carpeta, nombre), m, limite_s, bibit_min_cols)
            if estado == "ok":
                fila["Numero_Resultados"], fila["Tiempo_Ejecucion"] = n, f"{t:.8f}"
                log(log_path, f"   {modo:<20} {t:12.3f} s   {n} biclústeres")
            elif estado == "timeout":
                fila["Tiempo_Ejecucion"] = f"TIMEOUT (>{etiqueta_limite})"
                techo[m].append((r, c))
                log(log_path, f"   {modo:<20} TIMEOUT (>{etiqueta_limite})")
            else:
                fila["Tiempo_Ejecucion"] = "ERROR"
                log(log_path, f"   {modo:<20} ERROR: {t}")
            guardar(csv_path, fila)
            hechos.add((nombre, modo))

        # Todos los modos deberían devolver el mismo número de biclústeres
        filas = [f for f in leer_csv(csv_path) if f["Nombre_Dataset"] == nombre
                 and re.fullmatch(r"[\d.]+", f["Tiempo_Ejecucion"])]
        cuentas = {f["Modo"]: f["Numero_Resultados"] for f in filas}
        if len(set(cuentas.values())) > 1:
            log(log_path, f"   AVISO: número de biclústeres distinto entre modos: {cuentas}")

    log(log_path, f"=== FIN {alg.upper()} en {timedelta(seconds=int(time.time() - inicio))} ===")


if __name__ == "__main__":
    main()
