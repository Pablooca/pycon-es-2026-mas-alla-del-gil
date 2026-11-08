"""
Gráficos para la ponencia a partir del benchmark v2 (código optimizado, AWS g5.8xlarge).

Solo usa medidas reales. Lo que no terminó en el límite de 30 min se muestra como tal:
- BCCA: el secuencial de 800x800 y 1000x1000 se dibuja con la estimación de
  estimar_tiempos.py (marcadores huecos y línea discontinua, rotulado "estimado").
- BiBit: no se usa bibit_estimaciones.csv (no respeta el límite de 30 min y su modelo no
  recoge el efecto de la densidad). Se estima aquí con estimar_bibit(), un modelo del
  trabajo del algoritmo (parejas de genes y biclústeres), y se dibuja igual que en BCCA.

Uso:
    python graficos_ponencia.py
    BENCHMARK_DIR=/ruta/a/resultados python graficos_ponencia.py

Salida: ../presentacion/figuras/resultados/*.png y resumen_ponencia.md
"""
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
from scipy.optimize import nnls
from scipy.stats import binom

BASE_DIR = Path(__file__).resolve().parent
RESULTADOS = Path(os.environ.get("BENCHMARK_DIR", BASE_DIR.parent / "benchmark" / "resultados"))
SALIDA = BASE_DIR.parent / "presentacion" / "figuras" / "resultados"
LIMITE_S = 1800  # límite por ejecución del benchmark (0,5 h)

# --- Estilo ----------------------------------------------------------------------
SUPERFICIE = "#fafafa"  # fondo del tema metropolis
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_SUAVE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"

# Paleta categórica validada (CVD y visión normal) sobre #fafafa; gris para la referencia
MODOS = [  # (clave interna, nombre en las gráficas, color, marcador)
    ("seq", "Python secuencial", "#52514e", "o"),
    ("numba_cpu", "Numba CPU", "#2a78d6", "s"),
    ("numba_gpu", "Numba GPU", "#eb6834", "^"),
    ("cy_seq", "Cython secuencial", "#1baf7a", "D"),
    ("cy_omp", "Cython OpenMP", "#eda100", "v"),
    ("cy_gpu", "Cython GPU", "#e87ba4", "P"),
]
NOMBRE = {k: n for k, n, _, _ in MODOS}
COLOR = {k: c for k, _, c, _ in MODOS}
MARCA = {k: m for k, _, _, m in MODOS}
ACELERADOS = [k for k, *_ in MODOS if k != "seq"]

MODO_CSV = {
    "Python Secuencial": "seq", "CPU Secuencial": "seq",
    "Numba Paralelo CPU": "numba_cpu", "Numba GPU": "numba_gpu",
    "Cython Secuencial": "cy_seq", "Cython OpenMP": "cy_omp", "Cython GPU": "cy_gpu",
}
PATRON = {"bibit": r"(\d+)x(\d+)_(\d+)\.csv", "bcca": r"(\d+)x(\d+)_var(\d+)\.csv"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 17,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": EJE,
    "axes.labelcolor": TINTA_2,
    "xtick.color": TINTA_2,
    "ytick.color": TINTA_2,
    "text.color": TINTA,
    "axes.grid": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 2.2,
    "lines.markersize": 8,
})
TAMANO_FIG = (11, 5.6)  # apaisado para diapositivas 16:9


def coma(x, dec=1):
    """Número con coma decimal (estilo español)."""
    return f"{x:.{dec}f}".replace(".", ",")


def fmt_speedup(x):
    return f"{coma(x)}×" if x < 10 else f"{x:.0f}×"


def eje_tiempo(ax, eje="y"):
    """Eje logarítmico de tiempo con marcas legibles (ms, s, min, h)."""
    marcas = [(0.001, "1 ms"), (0.01, "10 ms"), (0.1, "100 ms"), (1, "1 s"), (10, "10 s"),
              (60, "1 min"), (600, "10 min"), (3600, "1 h"), (3 * 3600, "3 h"), (10 * 3600, "10 h"),
              (86400, "1 día"), (3 * 86400, "3 días")]
    lim = ax.get_ylim() if eje == "y" else ax.get_xlim()
    usar = [(v, t) for v, t in marcas if lim[0] <= v <= lim[1]]
    axis = ax.yaxis if eje == "y" else ax.xaxis
    axis.set_major_locator(FixedLocator([v for v, _ in usar]))
    axis.set_ticklabels([t for _, t in usar])
    axis.set_minor_locator(NullLocator())


def nota(fig, texto):
    fig.text(0.01, 0.012, texto, fontsize=13, color=TINTA_SUAVE, ha="left", va="bottom")


def guardar(fig, nombre):
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / nombre, dpi=200)
    plt.close(fig)
    print("  ", SALIDA / nombre)


# --- Datos -----------------------------------------------------------------------
def cargar(alg):
    df = pd.read_csv(RESULTADOS / f"{alg}_results.csv")
    df["modo"] = df["Modo"].map(MODO_CSV)
    df["t"] = pd.to_numeric(df["Tiempo_Ejecucion"], errors="coerce")
    dims = df["Nombre_Dataset"].str.extract(PATRON[alg]).astype(int)
    df["filas"], df["cols"], df["param"] = dims[0], dims[1], dims[2]
    df["geometria"] = df["filas"].astype(str) + "×" + df["cols"].astype(str)
    df["celdas"] = df["filas"] * df["cols"]
    return df


def speedups(df):
    """Speedup de cada modo frente al secuencial, solo en matrices con los dos tiempos medidos."""
    piv = df.pivot(index="Nombre_Dataset", columns="modo", values="t")
    piv = piv[piv["seq"].notna() & (piv["seq"] > 0)]
    sp = pd.DataFrame({m: piv["seq"] / piv[m] for m in ACELERADOS if m in piv})
    return sp


def media_geom(s):
    s = s.dropna()
    s = s[s > 0]
    return float(np.exp(np.log(s).mean())) if len(s) else np.nan


# --- Gráficos --------------------------------------------------------------------
def barras_speedup(ax, sp, titulo=None, xlim=None):
    valores = {m: media_geom(sp[m]) for m in ACELERADOS}
    orden = sorted(ACELERADOS, key=lambda m: valores[m])
    y = np.arange(len(orden))
    ax.barh(y, [valores[m] for m in orden], height=0.62, color=[COLOR[m] for m in orden],
            edgecolor=SUPERFICIE, linewidth=2)
    ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    for yi, m in zip(y, orden):
        ax.text(valores[m] * 1.08, yi, fmt_speedup(valores[m]), va="center", ha="left",
                fontsize=15, fontweight="bold", color=TINTA)
    ax.set_yticks(y, [NOMBRE[m] for m in orden])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.axvline(1, color=EJE, linewidth=1.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    if titulo:
        ax.set_title(titulo, loc="left", fontweight="bold", color=TINTA)
    return valores


def fig_speedup(alg, sp, n):
    fig, ax = plt.subplots(figsize=TAMANO_FIG)
    vals = barras_speedup(ax, sp, xlim=(1, max(sp.max().max(), 10) * 3))
    ax.set_xlabel("Speedup frente a Python secuencial (escala log; mayor es mejor)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    nota(fig, f"Media geométrica sobre {n} matrices con todos los modos medidos · "
              "AWS g5.8xlarge (AMD EPYC 7R32, 32 vCPU · NVIDIA A10G)")
    guardar(fig, f"{alg}_speedup.png")
    return vals


def fig_comparativa(sp_bibit, sp_bcca, n_bibit, n_bcca):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True)
    tope = max(sp_bibit.max().max(), sp_bcca.max().max()) * 3
    for ax, sp, tit in ((axes[0], sp_bibit, "BiBit (bits)"), (axes[1], sp_bcca, "BCCA (coma flotante)")):
        barras_speedup(ax, sp, titulo=tit, xlim=(1, tope))
        ax.set_xlabel("Speedup (escala log)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    nota(fig, f"Media geométrica · BiBit: {n_bibit} matrices, BCCA: {n_bcca} matrices · "
              "mismo hardware y mismos seis modos")
    guardar(fig, "comparativa_speedup.png")


def leyenda_modos(ax, modos, extra=(), ncol=3):
    h = [Line2D([], [], color=COLOR[m], marker=MARCA[m], markeredgecolor=SUPERFICIE,
                linewidth=2.2, label=NOMBRE[m]) for m in modos]
    ax.figure.legend(handles=h + list(extra), loc="upper left", ncol=ncol, bbox_to_anchor=(0.01, 0.995),
                     handlelength=1.8, columnspacing=1.4)


def fig_bcca_tiempo(df, est):
    med = (df.dropna(subset=["t"]).groupby(["geometria", "celdas", "filas", "modo"])["t"]
           .median().reset_index())
    geos = (df[["geometria", "celdas", "filas"]].drop_duplicates()
            .sort_values(["celdas", "filas"]).reset_index(drop=True))
    x_de = {g: i for i, g in enumerate(geos["geometria"])}
    fig, ax = plt.subplots(figsize=TAMANO_FIG)
    for m, *_ in MODOS:
        d = med[med["modo"] == m].copy()
        d["x"] = d["geometria"].map(x_de)
        d = d.sort_values("x")
        ax.plot(d["x"], d["t"], color=COLOR[m], marker=MARCA[m], markeredgecolor=SUPERFICIE,
                markeredgewidth=1.2, zorder=3)
    # Secuencial estimado donde no terminó en 30 min
    extra = []
    if est is not None and len(est):
        e = est.groupby("geometria")["t_est"].median().reset_index()
        e["x"] = e["geometria"].map(x_de)
        e = e.sort_values("x")
        ult = med[med["modo"] == "seq"].assign(x=lambda d: d["geometria"].map(x_de)).sort_values("x").iloc[-1]
        ax.plot([ult["x"], *e["x"]], [ult["t"], *e["t_est"]], color=COLOR["seq"], linestyle=(0, (4, 3)),
                linewidth=1.8, zorder=2)
        ax.plot(e["x"], e["t_est"], linestyle="none", marker="o", markerfacecolor=SUPERFICIE,
                markeredgecolor=COLOR["seq"], markeredgewidth=2, zorder=3)
        extra = [Line2D([], [], color=COLOR["seq"], linestyle=(0, (4, 3)), marker="o",
                        markerfacecolor=SUPERFICIE, markeredgewidth=2, label="Secuencial estimado")]
    ax.axhline(LIMITE_S, color=TINTA_SUAVE, linewidth=1)
    ax.text(0.05, LIMITE_S * 1.12, "límite por ejecución: 30 min", color=TINTA_SUAVE, fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 4 * 3600)
    eje_tiempo(ax)
    ax.set_xticks(range(len(geos)), geos["geometria"], rotation=45, ha="right")
    ax.set_xlim(-0.4, len(geos) - 0.6)
    ax.set_xlabel("Matriz (genes × condiciones), ordenadas por tamaño")
    ax.set_ylabel("Tiempo (mediana)")
    leyenda_modos(ax, [m for m, *_ in MODOS], extra, ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    nota(fig, "BCCA · mediana de 6 variantes por tamaño · escala logarítmica · AWS g5.8xlarge")
    guardar(fig, "bcca_tiempo_tamano.png")


def fig_bcca_speedup_tamano(df, est):
    tiempos = df.pivot(index="Nombre_Dataset", columns="modo", values="t")
    base = tiempos["seq"].copy()
    estimado = pd.Series(False, index=base.index)
    if est is not None and len(est):
        faltan = base.isna() & base.index.isin(est.index)
        base[faltan] = est.loc[base.index[faltan], "t_est"]
        estimado[faltan] = True
    info = df.drop_duplicates("Nombre_Dataset").set_index("Nombre_Dataset")[["geometria", "celdas", "filas"]]
    geos = info.drop_duplicates("geometria").sort_values(["celdas", "filas"])["geometria"].tolist()
    x_de = {g: i for i, g in enumerate(geos)}
    fig, ax = plt.subplots(figsize=TAMANO_FIG)
    for m in ACELERADOS:
        sp = (base / tiempos[m]).rename("sp").to_frame().join(info).assign(est=estimado)
        g = sp.groupby("geometria").agg(sp=("sp", "median"), est=("est", "any")).reset_index()
        g["x"] = g["geometria"].map(x_de)
        g = g.sort_values("x")
        medido, est_g = g[~g["est"]], g[g["est"]]
        ax.plot(medido["x"], medido["sp"], color=COLOR[m], marker=MARCA[m], markeredgecolor=SUPERFICIE,
                markeredgewidth=1.2, zorder=3)
        if len(est_g):
            puente = pd.concat([medido.tail(1), est_g])
            ax.plot(puente["x"], puente["sp"], color=COLOR[m], linestyle=(0, (4, 3)), linewidth=1.8, zorder=2)
            ax.plot(est_g["x"], est_g["sp"], linestyle="none", marker=MARCA[m], markerfacecolor=SUPERFICIE,
                    markeredgecolor=COLOR[m], markeredgewidth=2, zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 500]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xticks(range(len(geos)), geos, rotation=45, ha="right")
    ax.set_xlim(-0.4, len(geos) - 0.6)
    ax.set_xlabel("Matriz (genes × condiciones), ordenadas por tamaño")
    ax.set_ylabel("Speedup frente a Python secuencial")
    extra = [Line2D([], [], color=TINTA_2, linestyle=(0, (4, 3)), marker="o", markerfacecolor=SUPERFICIE,
                    markeredgewidth=2, label="Con secuencial estimado")]
    leyenda_modos(ax, ACELERADOS, extra, ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    nota(fig, "BCCA · mediana de 6 variantes por tamaño · huecos: el secuencial no terminó en 30 min y se estima")
    guardar(fig, "bcca_speedup_tamano.png")


def _geos_bibit(df):
    """Geometrías ordenadas por nº de genes: BiBit compara cada pareja de genes (coste ~ genes²)."""
    return (df[["geometria", "filas", "cols"]].drop_duplicates()
            .sort_values(["filas", "cols"])["geometria"].tolist())


def estimar_bibit(df, cmnc=160):
    """Completa los tiempos de BiBit que no se midieron con un modelo del trabajo que hace el algoritmo.

    - Biclústeres esperados: parejas de genes × P(Binomial(columnas, densidad²) >= cMnc).
      Reproduce los recuentos medidos con menos de un 7 % de error.
    - Tiempo de cada modo (mínimos cuadrados no negativos, error relativo):
      t = a·parejas·palabras + b·biclústeres + c·biclústeres·palabras + d·genes·columnas + e
          + f·biclústeres·genes
      (palabras = columnas/64; el último término es comparar cada patrón con todos los genes).
    - Si la ejecución no terminó (TIMEOUT, u OMITIDO con al menos el trabajo de un TIMEOUT),
      la estimación nunca baja de 30 min.
    Devuelve df con 't_plot' (medido o estimado) y 'estimado' (bool).
    """
    d = df.copy()
    d["pares"] = d["filas"] * (d["filas"] - 1) / 2
    d["palabras"] = np.ceil(d["cols"] / 64)
    d["nb"] = d["pares"] * binom.sf(cmnc - 1, d["cols"], (d["param"] / 100) ** 2)

    def X(g):
        return np.column_stack([g["pares"] * g["palabras"], g["nb"], g["nb"] * g["palabras"],
                                g["filas"] * g["cols"], np.ones(len(g)), g["nb"] * g["filas"]])

    d["t_plot"], d["estimado"] = d["t"], False
    estado = d["Tiempo_Ejecucion"].astype(str)
    for m, g in d.groupby("modo"):
        ok = g[g["t"].notna() & (g["t"] > 0)]
        coef, _ = nnls(X(ok) / ok["t"].to_numpy()[:, None], np.ones(len(ok)))
        falta = g[g["t"].isna()]
        if falta.empty:
            continue
        est = pd.Series(X(falta) @ coef, index=falta.index)
        techos = g[estado[g.index].str.startswith("TIMEOUT")][["filas", "cols", "nb"]].to_numpy()
        for i, r in falta.iterrows():
            no_termino = estado[i].startswith("TIMEOUT") or any(
                r["filas"] >= f and r["cols"] >= c and r["nb"] >= nb * 0.99 for f, c, nb in techos)
            if no_termino:
                est[i] = max(est[i], LIMITE_S)
        d.loc[falta.index, "t_plot"] = est
        d.loc[falta.index, "estimado"] = True
    return d


def _serie_medida_estimada(ax, x, y, est, m):
    """Línea continua entre puntos medidos; discontinua en los tramos que tocan un estimado."""
    for i in range(len(x) - 1):
        if np.isnan(y[i]) or np.isnan(y[i + 1]):
            continue
        tramo_est = est[i] or est[i + 1]
        ax.plot(x[i:i + 2], y[i:i + 2], color=COLOR[m], zorder=2,
                linestyle=(0, (4, 3)) if tramo_est else "-", linewidth=1.8 if tramo_est else 2.2)
    med = ~est & ~np.isnan(y)
    ax.plot(x[med], y[med], linestyle="none", marker=MARCA[m], color=COLOR[m],
            markeredgecolor=SUPERFICIE, markeredgewidth=1.2, zorder=3)
    ax.plot(x[est], y[est], linestyle="none", marker=MARCA[m], markerfacecolor=SUPERFICIE,
            markeredgecolor=COLOR[m], markeredgewidth=2, zorder=3)


NOTA_BIBIT = ("huecos y discontinua: estimado con un modelo de parejas de genes y biclústeres; "
              "si no terminó, ≥ 30 min")


def fig_bibit_tiempo_geometria(d, densidad=50):
    d = d[d["param"] == densidad]
    geos = _geos_bibit(d)
    x = np.arange(len(geos))
    fig, ax = plt.subplots(figsize=TAMANO_FIG)
    for m, *_ in MODOS:
        dm = d[d["modo"] == m].set_index("geometria").reindex(geos)
        _serie_medida_estimada(ax, x, dm["t_plot"].to_numpy(float), dm["estimado"].fillna(False).to_numpy(bool), m)
    ax.axhline(LIMITE_S, color=TINTA_SUAVE, linewidth=1)
    ax.text(0.05, LIMITE_S * 1.12, "límite por ejecución: 30 min", color=TINTA_SUAVE, fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(4 * 3600, d["t_plot"].max() * 2))
    eje_tiempo(ax)
    ax.set_xticks(x, geos, rotation=45, ha="right")
    ax.set_xlim(-0.4, len(geos) - 0.6)
    ax.set_xlabel("Matriz (genes × muestras), ordenadas por número de genes")
    ax.set_ylabel("Tiempo")
    extra = [Line2D([], [], color=TINTA_2, linestyle=(0, (4, 3)), marker="o", markerfacecolor=SUPERFICIE,
                    markeredgewidth=2, label="Estimado (no medido)")]
    leyenda_modos(ax, [m for m, *_ in MODOS], extra, ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    nota(fig, f"BiBit · densidad {densidad} % · " + NOTA_BIBIT)
    guardar(fig, "bibit_tiempo_tamano.png")


def fig_bibit_speedup_geometria(d, densidad=50):
    d = d[d["param"] == densidad].copy()
    # Si el modo acelerado no terminó, su tiempo solo es un mínimo: no hay speedup que calcular
    no_termino = d["Tiempo_Ejecucion"].astype(str).str.match("TIMEOUT|OMITIDO") & (d["modo"] != "seq")
    d.loc[no_termino, "t_plot"] = np.nan
    sin_datos = [g for g, x in d[d["modo"] != "seq"].groupby("geometria")["t_plot"] if x.isna().all()]
    geos = [g for g in _geos_bibit(d) if g not in sin_datos]
    x = np.arange(len(geos))
    t = d.pivot(index="geometria", columns="modo", values="t_plot").reindex(geos)
    e = d.pivot(index="geometria", columns="modo", values="estimado").reindex(geos).fillna(False).astype(bool)
    fig, ax = plt.subplots(figsize=TAMANO_FIG)
    for m in ACELERADOS:
        sp = (t["seq"] / t[m]).to_numpy(float)
        _serie_medida_estimada(ax, x, sp, (e["seq"] | e[m]).to_numpy(bool), m)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 500]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xticks(x, geos, rotation=45, ha="right")
    ax.set_xlim(-0.4, len(geos) - 0.6)
    ax.set_xlabel("Matriz (genes × muestras), ordenadas por número de genes")
    ax.set_ylabel("Speedup frente a Python secuencial")
    extra = [Line2D([], [], color=TINTA_2, linestyle=(0, (4, 3)), marker="o", markerfacecolor=SUPERFICIE,
                    markeredgewidth=2, label="Con algún tiempo estimado")]
    leyenda_modos(ax, ACELERADOS, extra, ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    nota(fig, f"BiBit · densidad {densidad} % · huecos y discontinua: Python secuencial estimado"
              + (f" · sin {' ni '.join(sin_datos)}: ningún modo terminó en 30 min" if sin_datos else ""))
    guardar(fig, "bibit_speedup_tamano.png")


# --- Resumen en texto (la "vista de tabla" de las gráficas) -------------------------
def resumen(datos, vals):
    lineas = ["# Resumen para la ponencia (benchmark v2)\n",
              f"Datos: `{RESULTADOS}`. Límite por ejecución: 30 min.\n"]
    for alg, df in datos.items():
        sp = speedups(df)
        estados = df["Tiempo_Ejecucion"].astype(str)
        lineas += [f"\n## {alg.upper()}\n",
                   f"- Ejecuciones: {len(df)} · medidas: {df['t'].notna().sum()} · "
                   f"TIMEOUT: {estados.str.startswith('TIMEOUT').sum()} · "
                   f"OMITIDO: {estados.str.startswith('OMITIDO').sum()}",
                   f"- Matrices con todos los modos medidos (base del speedup): {len(sp.dropna())}\n",
                   "| Modo | Speedup (media geom.) | Speedup (mediana) | Mín. | Máx. |",
                   "|---|---|---|---|---|"]
        for m in sorted(ACELERADOS, key=lambda m: -vals[alg][m]):
            s = sp[m].dropna()
            lineas.append(f"| {NOMBRE[m]} | {fmt_speedup(vals[alg][m])} | {fmt_speedup(s.median())} | "
                          f"{fmt_speedup(s.min())} | {fmt_speedup(s.max())} |")
        lineas.append("\nTiempo mediano por geometría (s):\n")
        piv = (df.dropna(subset=["t"]).groupby(["celdas", "filas", "geometria", "modo"])["t"].median()
               .unstack("modo").reset_index().sort_values(["celdas", "filas"]))
        cols = [m for m, *_ in MODOS if m in piv]
        lineas.append("| Matriz | " + " | ".join(NOMBRE[m] for m in cols) + " |")
        lineas.append("|---|" + "---|" * len(cols))
        for _, r in piv.iterrows():
            celdas = [("—" if pd.isna(r[m]) else coma(r[m], 2 if r[m] < 10 else 0)) for m in cols]
            lineas.append(f"| {r['geometria']} | " + " | ".join(celdas) + " |")
    (SALIDA / "resumen_ponencia.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("  ", SALIDA / "resumen_ponencia.md")


def main():
    print(f"Leyendo {RESULTADOS}")
    bibit, bcca = cargar("bibit"), cargar("bcca")

    est = None
    ruta_est = RESULTADOS / "bcca_estimaciones.csv"
    if ruta_est.exists():
        e = pd.read_csv(ruta_est)
        e = e[e["Modo"].map(MODO_CSV) == "seq"].set_index("Nombre_Dataset")
        e["t_est"] = e["Tiempo_Ejecucion_Estimado"]
        dims = e.index.to_series().str.extract(PATRON["bcca"]).astype(int)
        e["geometria"] = dims[0].astype(str) + "×" + dims[1].astype(str)
        est = e[["t_est", "geometria"]]

    sp_bibit, sp_bcca = speedups(bibit), speedups(bcca)
    n_bibit, n_bcca = len(sp_bibit.dropna()), len(sp_bcca.dropna())
    vals = {"bibit": fig_speedup("bibit", sp_bibit.dropna(), n_bibit),
            "bcca": fig_speedup("bcca", sp_bcca.dropna(), n_bcca)}
    fig_comparativa(sp_bibit.dropna(), sp_bcca.dropna(), n_bibit, n_bcca)
    fig_bcca_tiempo(bcca, est)
    fig_bcca_speedup_tamano(bcca, est)
    bibit_completo = estimar_bibit(bibit)
    fig_bibit_tiempo_geometria(bibit_completo)
    fig_bibit_speedup_geometria(bibit_completo)
    resumen({"bibit": bibit, "bcca": bcca}, vals)


if __name__ == "__main__":
    main()
