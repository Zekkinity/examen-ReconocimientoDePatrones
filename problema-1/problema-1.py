"""
Problema 1: Default of Credit Card Clients (UCI)
Grupos (a mod 3 = 1): EDUCATION -> posgrado (1) contra el resto (0).
Positivo = "incumple". Predecir 1 = RECHAZAR el crédito; predecir 0 = APROBAR.
  FPR = buen pagador rechazado      FNR = moroso aprobado
"""

# Para que python encuentre 'helper.py' en la carpeta superior
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # solo guarda las figuras, no bloquea con plt.show()
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import (GridSearchCV, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import helper

# ------------------------------------------------------------------
# Configuracion
# ------------------------------------------------------------------
G, L = 1.0, 2.5            # ganancia por buen cliente aprobado / perdida por moroso aprobado
FAIR = "FPR"               # definicion de equidad: "FPR" (buenos pagadores) o "FNR" (morosos)
TOL = 0.01                 # diferencia maxima tolerada en la metrica de equidad
N_BOOT = 1000
THS = np.round(np.arange(0.05, 0.601, 0.01), 2)   # rejilla de umbrales (sin problemas de float)
pd.set_option("display.width", 160)

# ------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
helper.get_file(URL, "dataset")
df = pd.read_excel("dataset/default of credit card clients.xls", header=1)

TARGET = "default payment next month"
df = df.drop(columns="ID")                       # el ID no es una variable predictiva

# Codigos que no aparecen en la descripcion:
#  EDUCATION 0, 5, 6 -> 4 (otros).   MARRIAGE 0 -> 3 (otros).
#  PAY_x: -2 y 0 no estan descritos; se dejan como estan (escala ordinal: -2,-1,0 = sin atraso).
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

group = (df["EDUCATION"] == 1).astype(int)       # 1 = posgrado, 0 = resto
y = df[TARGET]
X = df.drop(columns=[TARGET])                    # DECISION: el modelo SI usa EDUCATION (en (e) se quita)

# ------------------------------------------------------------------
# (a) Datos y modelo
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, group, test_size=0.3, random_state=helper.SEED, stratify=y)
yt, gt = y_test.values, g_test.values

lr_grid = GridSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    scoring="roc_auc", cv=5, n_jobs=-1)
lr_grid.fit(X_train, y_train)

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=helper.SEED),
    {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 4]},
    scoring="roc_auc", cv=5, n_jobs=-1)
gb_grid.fit(X_train, y_train)

lr_auc = roc_auc_score(y_test, lr_grid.predict_proba(X_test)[:, 1])
gb_auc = roc_auc_score(y_test, gb_grid.predict_proba(X_test)[:, 1])
print("=== (a) Modelos ===")
print(f"Regresion logistica: mejores params={lr_grid.best_params_}  AUC CV={lr_grid.best_score_:.4f}  AUC prueba={lr_auc:.4f}")
print(f"Gradient Boosting  : mejores params={gb_grid.best_params_}  AUC CV={gb_grid.best_score_:.4f}  AUC prueba={gb_auc:.4f}")

# El "mejor modelo" se elige con CV en entrenamiento (no con la prueba)
best_grid = gb_grid if gb_grid.best_score_ >= lr_grid.best_score_ else lr_grid
best_est = best_grid.best_estimator_
print("Mejor modelo (por AUC CV):", "Gradient Boosting" if best_grid is gb_grid else "Regresion logistica")

# ------------------------------------------------------------------
# (b) Calibracion
# ------------------------------------------------------------------
p_raw = best_est.predict_proba(X_test)[:, 1]

# Calibracion SIN fuga: el calibrador se ajusta con CV dentro del entrenamiento
cal_template = CalibratedClassifierCV(clone(best_est), method="sigmoid", cv=5)
cal = clone(cal_template).fit(X_train, y_train)
p_cal = cal.predict_proba(X_test)[:, 1]


def resumen_calibracion(nombre, p):
    filas = []
    for etiqueta, mask in [("Global", np.ones(len(yt), bool)), ("Posgrado", gt == 1), ("Resto", gt == 0)]:
        filas.append({"grupo": etiqueta, "n": mask.sum(),
                      "score_promedio": p[mask].mean(), "tasa_real": yt[mask].mean(),
                      "brier": brier_score_loss(yt[mask], p[mask])})
    print(f"\n--- {nombre} ---")
    print(pd.DataFrame(filas).round(4).to_string(index=False))


print("\n=== (b) Calibracion ===")
resumen_calibracion("Sin calibrar", p_raw)
resumen_calibracion("Calibrado (sigmoide, CV en entrenamiento)", p_cal)
print(f"AUC prueba: sin calibrar={roc_auc_score(yt, p_raw):.4f}  calibrado={roc_auc_score(yt, p_cal):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, p, titulo in [(axes[0], p_raw, "Sin calibrar"), (axes[1], p_cal, "Calibrado")]:
    for etiqueta, mask in [("Global", np.ones(len(yt), bool)), ("Posgrado", gt == 1), ("Resto", gt == 0)]:
        pt, pp = calibration_curve(yt[mask], p[mask], n_bins=10, strategy="quantile")
        ax.plot(pp, pt, marker="o", label=etiqueta)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibracion perfecta")
    ax.set_title(titulo)
    ax.set_xlabel("Probabilidad predicha")
    ax.legend()
axes[0].set_ylabel("Fraccion de incumplimientos reales")
fig.suptitle("Curvas de calibracion (prueba)")
fig.savefig("curvas_calibracion.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# De aqui en adelante se usan las probabilidades CALIBRADAS.
probs_test = p_cal

# ------------------------------------------------------------------
# Utilidades: metricas, rejilla de umbrales, bootstrap
# ------------------------------------------------------------------
def metricas(y_true, pred):
    y_true = np.asarray(y_true) == 1
    pred = np.asarray(pred).astype(bool)
    TP = np.sum(y_true & pred); FP = np.sum(~y_true & pred)
    TN = np.sum(~y_true & ~pred); FN = np.sum(y_true & ~pred)
    pg = y_true.mean()
    PPV = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    FNR = FN / (FN + TP) if (FN + TP) > 0 else np.nan
    return {"n": len(y_true), "pg": pg, "rechazo": pred.mean(), "PPV": PPV,
            "FPR": FPR, "FNR": FNR, "identidad": pg * (1 - FNR) + (1 - pg) * FPR}


def tabla(y_true, p, g, t1, t0):
    y_true, p, g = map(np.asarray, (y_true, p, g))
    m1 = metricas(y_true[g == 1], p[g == 1] >= t1)
    m0 = metricas(y_true[g == 0], p[g == 0] >= t0)
    return pd.DataFrame([m1, m0], index=["Posgrado (1)", "Resto (0)"])


def verifica_identidad(t):
    # rechazo = pg*(1-FNR) + (1-pg)*FPR
    for idx, row in t.iterrows():
        print(f"  Identidad [{idx}]: rechazo={row['rechazo']:.6f}  pg(1-FNR)+(1-pg)FPR={row['identidad']:.6f}  "
              f"dif={abs(row['rechazo'] - row['identidad']):.2e}")


def conteos(y_true, p):
    """TP, FP, TN, FN para cada umbral de THS."""
    pred = p[None, :] >= THS[:, None]
    pos = (y_true == 1)[None, :]
    return ((pred & pos).sum(1), (pred & ~pos).sum(1), (~pred & ~pos).sum(1), (~pred & pos).sum(1))


def tasas(c):
    TP, FP, TN, FN = [a.astype(float) for a in c]
    ppv = np.divide(TP, TP + FP, out=np.full_like(TP, np.nan), where=(TP + FP) > 0)
    return {"fpr": FP / (FP + TN), "fnr": FN / (FN + TP), "rech": (TP + FP) / (TP + FP + TN + FN),
            "ppv": ppv, "gan": TN * G - FN * L}   # se aprueba (pred=0): buenos +G, malos -L


def rejilla(y_true, p, g):
    """Ganancia y diferencias entre grupos para TODOS los pares (t_pos, t_resto)."""
    r1 = tasas(conteos(y_true[g == 1], p[g == 1]))
    r0 = tasas(conteos(y_true[g == 0], p[g == 0]))
    T1, T0 = np.meshgrid(THS, THS, indexing="ij")
    return pd.DataFrame({
        "t_pos": T1.ravel(), "t_resto": T0.ravel(),
        "ganancia": (r1["gan"][:, None] + r0["gan"][None, :]).ravel(),
        "dif_rechazo": (r1["rech"][:, None] - r0["rech"][None, :]).ravel(),
        "dif_FPR": (r1["fpr"][:, None] - r0["fpr"][None, :]).ravel(),
        "dif_FNR": (r1["fnr"][:, None] - r0["fnr"][None, :]).ravel(),
        "dif_PPV": (r1["ppv"][:, None] - r0["ppv"][None, :]).ravel()})


def fila(grid, t1, t0):
    return grid[np.isclose(grid["t_pos"], t1) & np.isclose(grid["t_resto"], t0)].iloc[0]


def bootstrap_dif(y_true, p, g, t1, t0, B=N_BOOT, seed=helper.SEED):
    rng = np.random.default_rng(seed)                      # reproducible
    pred = np.where(g == 1, p >= t1, p >= t0)
    n = len(y_true)
    d_fpr, d_fnr = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        yb, pb, gb = y_true[i], pred[i], g[i]
        m1, m0 = metricas(yb[gb == 1], pb[gb == 1]), metricas(yb[gb == 0], pb[gb == 0])
        d_fpr.append(m1["FPR"] - m0["FPR"]); d_fnr.append(m1["FNR"] - m0["FNR"])
    return np.array(d_fpr), np.array(d_fnr)


def reporta_ci(nombre, d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    veredicto = "CONTIENE 0 -> puede ser ruido" if lo <= 0 <= hi else "NO contiene 0 -> diferencia real"
    print(f"  IC 95% dif. {nombre}: [{lo:+.4f}, {hi:+.4f}]  {veredicto}")


# ------------------------------------------------------------------
# Umbral del banco (inciso (d)): se elige con predicciones fuera de muestra
# del ENTRENAMIENTO, para no usar la prueba al elegirlo.
# ------------------------------------------------------------------
p_oof = cross_val_predict(cal_template, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
gr_train = rejilla(y_train.values, p_oof, g_train.values)

t_teo = G / (G + L)   # aprobar conviene si (1-p)G > pL  <=>  p < G/(G+L)
diag = gr_train[np.isclose(gr_train["t_pos"], gr_train["t_resto"])]
t_star = diag.loc[diag["ganancia"].idxmax(), "t_pos"]
print("\n=== Umbral del banco ===")
print(f"Umbral teorico G/(G+L) = {t_teo:.4f}  |  umbral optimo empirico (CV en entrenamiento) = {t_star:.2f}")

# ------------------------------------------------------------------
# (c) Tabla por grupo con el umbral del banco
# ------------------------------------------------------------------
print(f"\n=== (c) Tabla por grupo, umbral t={t_star:.2f} (prueba) ===")
tabla_c = tabla(yt, probs_test, gt, t_star, t_star)
print(tabla_c.round(4))
verifica_identidad(tabla_c)

d_fpr, d_fnr = bootstrap_dif(yt, probs_test, gt, t_star, t_star)
print(f"Bootstrap ({N_BOOT} remuestras) de la prueba, Posgrado - Resto:")
reporta_ci("FPR", d_fpr)
reporta_ci("FNR", d_fnr)

# ------------------------------------------------------------------
# (d) Intervencion: umbral por grupo que iguala FPR (o FNR) con el menor costo
# ------------------------------------------------------------------
col = f"dif_{FAIR}"
elegibles = gr_train[gr_train[col].abs() <= TOL]
if elegibles.empty:   # por si ningun par cumple la tolerancia
    elegibles = gr_train[gr_train[col].abs() == gr_train[col].abs().min()]
mejor = elegibles.loc[elegibles["ganancia"].idxmax()]
t1_f, t0_f = mejor["t_pos"], mejor["t_resto"]

gr_test = rejilla(yt, probs_test, gt)
base, justo = fila(gr_test, t_star, t_star), fila(gr_test, t1_f, t0_f)

print(f"\n=== (d) Intervencion: igualar {FAIR} (tolerancia {TOL}) ===")
print(f"Umbrales elegidos en entrenamiento: t_posgrado={t1_f:.2f}, t_resto={t0_f:.2f}")
tabla_d = tabla(yt, probs_test, gt, t1_f, t0_f)
print(tabla_d.round(4))
verifica_identidad(tabla_d)

comp = pd.DataFrame({
    "Sin intervencion": [base["ganancia"], abs(base["dif_rechazo"]), abs(base["dif_FPR"]), abs(base["dif_FNR"]), abs(base["dif_PPV"])],
    "Con intervencion": [justo["ganancia"], abs(justo["dif_rechazo"]), abs(justo["dif_FPR"]), abs(justo["dif_FNR"]), abs(justo["dif_PPV"])]},
    index=["Ganancia total ($)", "|dif| tasa de rechazo (paridad demografica)", "|dif| FPR",
           "|dif| FNR", "|dif| PPV (paridad predictiva)"])
print("\nComparacion de definiciones de equidad (prueba):")
print(comp.round(4))
costo = base["ganancia"] - justo["ganancia"]
print(f"Costo de la equidad: ${costo:,.2f} ({costo / base['ganancia'] * 100:.2f}% de la ganancia)")

d_fpr_f, d_fnr_f = bootstrap_dif(yt, probs_test, gt, t1_f, t0_f)
print("Despues de intervenir:")
reporta_ci("FPR", d_fpr_f)
reporta_ci("FNR", d_fnr_f)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(gr_test[col].abs(), gr_test["ganancia"], c=gr_test["t_pos"], cmap="viridis", s=8, alpha=0.6)
ax.scatter(abs(base[col]), base["ganancia"], c="red", marker="*", s=200, label="Sin intervencion", zorder=3)
ax.scatter(abs(justo[col]), justo["ganancia"], c="black", marker="X", s=120, label="Equitativo elegido", zorder=3)
fig.colorbar(sc, label="Umbral Posgrado")
ax.set_xlabel(f"|{FAIR}_posgrado - {FAIR}_resto|")
ax.set_ylabel("Ganancia total del banco ($)")
ax.set_title(f"Ganancia vs disparidad de {FAIR} (prueba)")
ax.grid(True); ax.legend()
fig.savefig("tradeoff_ganancia.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# (e) Basta con no mirar? Reentrenar sin el atributo de grupo
# ------------------------------------------------------------------
print("\n=== (e) Reentrenamiento SIN EDUCATION ===")
X_train_ng, X_test_ng = X_train.drop(columns="EDUCATION"), X_test.drop(columns="EDUCATION")

# Mismos hiperparametros del mejor modelo (se podrian reajustar con CV; se reusan por comparabilidad)
cal_ng = clone(cal_template).fit(X_train_ng, y_train)
p_ng = cal_ng.predict_proba(X_test_ng)[:, 1]
print(f"AUC prueba: con EDUCATION={roc_auc_score(yt, probs_test):.4f}  sin EDUCATION={roc_auc_score(yt, p_ng):.4f}")

tabla_e = tabla(yt, p_ng, gt, t_star, t_star)
print(f"Tabla sin EDUCATION, umbral t={t_star:.2f}:")
print(tabla_e.round(4))
verifica_identidad(tabla_e)
print("Diferencias Posgrado - Resto (con EDUCATION -> sin EDUCATION):")
for m in ["rechazo", "FPR", "FNR", "PPV"]:
    print(f"  {m:8s}: {tabla_c.loc['Posgrado (1)', m] - tabla_c.loc['Resto (0)', m]:+.4f} -> "
          f"{tabla_e.loc['Posgrado (1)', m] - tabla_e.loc['Resto (0)', m]:+.4f}")
d_fpr_e, d_fnr_e = bootstrap_dif(yt, p_ng, gt, t_star, t_star)
reporta_ci("FPR (sin EDUCATION)", d_fpr_e)
reporta_ci("FNR (sin EDUCATION)", d_fnr_e)

# Que otras variables se relacionan con el grupo?
print("\nCorrelacion de cada variable con 'posgrado' (ordenadas por |corr|):")
corrs = X.drop(columns="EDUCATION").corrwith(group)
print(corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(10).round(4))

auc_proxy = cross_val_score(GradientBoostingClassifier(n_estimators=100, random_state=helper.SEED),
                            X_train_ng, g_train, cv=3, scoring="roc_auc", n_jobs=-1).mean()
print(f"AUC (CV) de predecir 'posgrado' con las demas variables: {auc_proxy:.4f}  (0.5 = sin informacion)")
print("Perfil promedio por grupo:")
print(X.assign(grupo=group).groupby("grupo")[["LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1"]].mean().round(2))
