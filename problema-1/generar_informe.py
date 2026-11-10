"""
Genera informe.md con las respuestas de (a)-(f) usando los resultados reales de problema_1.py.
Se llama al final de problema_1.py:  generar_informe.generar(globals())
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def _fmt(v, nd=4):
    if isinstance(v, (bool, np.bool_)):
        return "sí" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return "nan" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md_tabla(df, nd=4, indice="") :
    cols = [indice or (df.index.name or "")] + [str(c) for c in df.columns]
    lineas = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i in range(len(df)):
        celdas = [str(df.index[i])] + [_fmt(df[c].iloc[i], nd) for c in df.columns]
        lineas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(lineas)


def _tabla_cal(g, p):
    yt, gt = g["yt"], g["gt"]
    filas = []
    for etiqueta, mask in [("Global", np.ones(len(yt), bool)), ("Posgrado", gt == 1), ("Resto", gt == 0)]:
        filas.append({"grupo": etiqueta, "n": int(mask.sum()), "score promedio": p[mask].mean(),
                      "tasa real": yt[mask].mean(), "brecha (score - real)": p[mask].mean() - yt[mask].mean(),
                      "Brier": brier_score_loss(yt[mask], p[mask])})
    return pd.DataFrame(filas).set_index("grupo")


def _ci(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return lo, hi, bool(lo <= 0 <= hi)


def _tabla_ci(tabla, d_fpr, d_fnr):
    filas = []
    for nombre, d in [("FPR", d_fpr), ("FNR", d_fnr)]:
        lo, hi, cero = _ci(d)
        obs = tabla.loc["Posgrado (1)", nombre] - tabla.loc["Resto (0)", nombre]
        filas.append({"diferencia": nombre, "observada (Posgrado - Resto)": obs, "IC 95% inferior": lo,
                      "IC 95% superior": hi, "¿contiene 0?": cero})
    return pd.DataFrame(filas).set_index("diferencia")


def _veredicto(ci_df):
    partes = []
    for nombre in ci_df.index:
        if ci_df.loc[nombre, "¿contiene 0?"]:
            partes.append(f"la de {nombre} **podría ser ruido** (el intervalo incluye 0)")
        else:
            partes.append(f"la de {nombre} **es real** (el intervalo no incluye 0)")
    return "; ".join(partes)


def generar(g, ruta="informe.md"):
    out = []
    add = out.append
    G, Lc, FAIR, TOL = g["G"], g["L"], g["FAIR"], g["TOL"]
    t_star, t_teo = g["t_star"], g["t_teo"]
    t1_f, t0_f = g["t1_f"], g["t0_f"]
    tabla_c, tabla_d, tabla_e = g["tabla_c"], g["tabla_d"], g["tabla_e"]
    base, justo, comp = g["base"], g["justo"], g["comp"]
    mejor_gb = g["best_grid"] is g["gb_grid"]
    nombre_mejor = "Gradient Boosting" if mejor_gb else "Regresión logística"
    n_total = len(g["df"])
    P, R = "Posgrado (1)", "Resto (0)"

    add("# Problema 1 — Equidad en un modelo de crédito (Default of Credit Card Clients)\n")
    add(f"*Informe generado automáticamente por `problema_1.py`. Grupos: **posgrado** contra el **resto** (EDUCATION). "
        f"Positivo = incumple; predecir 1 = rechazar el crédito. G = {G}, L = {Lc}.*\n")

    # ------------------------------------------------------------------ (a)
    add("## (a) Datos y modelo\n")
    add(f"- {n_total:,} clientes. 30 % para prueba (estratificado por la variable objetivo, con la semilla del curso).")
    add("- Se eliminó `ID` (no es una variable predictiva).")
    add("- Códigos no descritos: `EDUCATION` 0, 5 y 6 → 4 (otros); `MARRIAGE` 0 → 3 (otros). "
        "`PAY_x` con −2 y 0 se dejó como escala ordinal (sin atraso).")
    add("- **El atributo del grupo (`EDUCATION`) entra al modelo** en los incisos (a)-(d); en (e) se quita.")
    add("- Hiperparámetros elegidos con validación cruzada de 5 pliegues **solo en el entrenamiento** (logística: `C`, con escalado; boosting: árboles, tasa de aprendizaje y profundidad).\n")
    modelos = pd.DataFrame({
        "Mejores hiperparámetros": [str(g["lr_grid"].best_params_), str(g["gb_grid"].best_params_)],
        "AUC CV (entrenamiento)": [g["lr_grid"].best_score_, g["gb_grid"].best_score_],
        "AUC en prueba": [g["lr_auc"], g["gb_auc"]]},
        index=["Regresión logística", "Gradient Boosting"])
    add(md_tabla(modelos))
    add(f"\nEl mejor modelo por AUC de validación cruzada es **{nombre_mejor}**.\n")

    # ------------------------------------------------------------------ (b)
    add("## (b) ¿Está calibrado?\n")
    raw, cal = _tabla_cal(g, g["p_raw"]), _tabla_cal(g, g["p_cal"])
    add("**Sin calibrar** (prueba):\n")
    add(md_tabla(raw))
    add("\n**Calibrado** (`CalibratedClassifierCV`, sigmoide, 5 pliegues dentro del entrenamiento):\n")
    add(md_tabla(cal))
    add("\n![Curvas de calibración](curvas_calibracion.png)\n")
    mx_raw, mx_cal = raw["brecha (score - real)"].abs().max(), cal["brecha (score - real)"].abs().max()
    if mx_raw < 0.01:
        estado = "ya estaba razonablemente calibrado en promedio (brechas menores a un punto porcentual)"
    else:
        estado = "no estaba bien calibrado (brecha de al menos un punto porcentual en algún grupo)"
    add(f"El modelo sin calibrar {estado}. La brecha absoluta máxima entre puntaje promedio y tasa real pasó de "
        f"**{mx_raw:.4f}** a **{mx_cal:.4f}** tras calibrar. "
        f"El AUC en prueba pasó de {g['auc_raw']:.4f} a {g['auc_cal']:.4f}: la calibración corrige la *escala* de las probabilidades, "
        "casi no cambia el *orden* de los clientes.")
    dif_g = abs(cal.loc["Posgrado", "brecha (score - real)"] - cal.loc["Resto", "brecha (score - real)"])
    add(f"\nCalibrar de forma global no garantiza calibración por grupo: después de calibrar, la diferencia entre las brechas de "
        f"Posgrado y Resto es **{dif_g:.4f}**. Un grupo con tasa real distinta puede quedar sobre o subestimado aunque el promedio global cuadre. "
        "Todo lo que sigue usa las probabilidades **calibradas**, porque solo así el umbral económico tiene sentido.\n")

    # ------------------------------------------------------------------ umbral
    add("## Umbral del banco\n")
    add(f"Aprobar conviene si (1 − p)·G > p·L, es decir p < G/(G+L) = **{t_teo:.4f}**. "
        f"El umbral óptimo empírico (búsqueda sobre predicciones fuera de muestra del entrenamiento, no sobre la prueba) es **t = {t_star:.2f}**. "
        "Ese es el umbral que se usa en (c).\n")

    # ------------------------------------------------------------------ (c)
    add(f"## (c) Tabla por grupo con el umbral del banco (t = {t_star:.2f})\n")
    add(md_tabla(tabla_c[["n", "pg", "rechazo", "PPV", "FPR", "FNR"]]))
    ident = tabla_c[["rechazo", "identidad"]].copy()
    ident["diferencia"] = (ident["rechazo"] - ident["identidad"]).abs()
    ident.columns = ["tasa de rechazo", "pg·(1−FNR) + (1−pg)·FPR", "|diferencia|"]
    add("\n**Verificación de la identidad** tasa de rechazo = pg·(1−FNR) + (1−pg)·FPR:\n")
    add(md_tabla(ident, nd=6))
    add("\nLa identidad se cumple (diferencias del orden del error de máquina): es la ley de probabilidad total, "
        "así que si los grupos tienen distinta pg no se pueden igualar a la vez rechazo, FPR y FNR.\n")
    ci_c = _tabla_ci(tabla_c, g["d_fpr"], g["d_fnr"])
    add(f"**Bootstrap** ({g['N_BOOT']:,} remuestras de la prueba, semilla fija):\n")
    add(md_tabla(ci_c))
    add(f"\nConclusión: {_veredicto(ci_c)}.\n")

    # ------------------------------------------------------------------ (d)
    quien = "buenos pagadores" if FAIR == "FPR" else "clientes que sí incumplirían"
    error = "rechazados por error" if FAIR == "FPR" else "aprobados por error"
    add("## (d) Intervenir\n")
    add(f"**Definición elegida: igualdad de oportunidades medida con {FAIR}.** Se busca que los {quien} tengan la misma "
        f"probabilidad de ser {error} sin importar su escolaridad. Se usa un umbral por grupo, elegido en el "
        f"entrenamiento (fuera de muestra) como el par de mayor ganancia con |dif. {FAIR}| ≤ {TOL}. "
        f"Umbrales: **posgrado = {t1_f:.2f}, resto = {t0_f:.2f}**.\n")
    add("Tabla de nuevo (prueba) con los umbrales por grupo:\n")
    add(md_tabla(tabla_d[["n", "pg", "rechazo", "PPV", "FPR", "FNR"]]))
    comp_s = comp.astype(object).copy()
    comp_s.iloc[0] = [f"{v:,.1f}" for v in comp.iloc[0]]
    for i in range(1, len(comp_s)):
        comp_s.iloc[i] = [f"{v:.4f}" for v in comp.iloc[i]]
    comp_s.index.name = "medida"
    add("\n**Antes y después** (prueba; ganancia en unidades con G = 1):\n")
    add(md_tabla(comp_s))
    costo = base["ganancia"] - justo["ganancia"]
    pct = costo / base["ganancia"] * 100 if base["ganancia"] != 0 else float("nan")
    add(f"\n**Costo de ser justo:** {costo:,.1f} unidades ({pct:.2f} % de la ganancia sin intervención).\n")
    add("![Ganancia contra disparidad](tradeoff_ganancia.png)\n")
    add("**¿Qué pasó con las otras definiciones?**\n")
    for etiqueta in comp.index[1:]:
        antes, despues = comp.loc[etiqueta, "Sin intervencion"], comp.loc[etiqueta, "Con intervencion"]
        mov = "bajó" if despues < antes else "subió" if despues > antes else "no cambió"
        add(f"- {etiqueta}: {mov} de {antes:.4f} a {despues:.4f}.")
    add("\nComo las tasas reales de incumplimiento difieren entre grupos, igualar una métrica mueve las demás: "
        "no se pueden cumplir todas las definiciones a la vez salvo con un clasificador perfecto.\n")
    ci_d = _tabla_ci(tabla_d, g["d_fpr_f"], g["d_fnr_f"])
    add("Bootstrap después de intervenir:\n")
    add(md_tabla(ci_d))
    add("")

    # ------------------------------------------------------------------ (e)
    add("\n## (e) ¿Basta con no mirar?\n")
    add("Se reentrenó el mejor modelo (mismos hiperparámetros, mismo calibrador) **sin `EDUCATION`**. "
        f"AUC en prueba: con el atributo **{g['auc_cal']:.4f}**, sin él **{g['auc_ng']:.4f}**.\n")
    add(f"Tabla sin el atributo (t = {t_star:.2f}):\n")
    add(md_tabla(tabla_e[["n", "pg", "rechazo", "PPV", "FPR", "FNR"]]))
    filas = []
    for m in ["rechazo", "FPR", "FNR", "PPV"]:
        a = tabla_c.loc[P, m] - tabla_c.loc[R, m]
        b = tabla_e.loc[P, m] - tabla_e.loc[R, m]
        filas.append({"medida": m, "dif. con EDUCATION": a, "dif. sin EDUCATION": b})
    dif_e = pd.DataFrame(filas).set_index("medida")
    add("\nDiferencias Posgrado − Resto:\n")
    add(md_tabla(dif_e))
    ci_e = _tabla_ci(tabla_e, g["d_fpr_e"], g["d_fnr_e"])
    add("\nBootstrap sin el atributo:\n")
    add(md_tabla(ci_e))
    a_fpr, b_fpr = abs(dif_e.loc["FPR", "dif. con EDUCATION"]), abs(dif_e.loc["FPR", "dif. sin EDUCATION"])
    a_fnr, b_fnr = abs(dif_e.loc["FNR", "dif. con EDUCATION"]), abs(dif_e.loc["FNR", "dif. sin EDUCATION"])
    persisten = (b_fpr > 0.5 * a_fpr) or (b_fnr > 0.5 * a_fnr)
    corrs = g["corrs"]
    top = corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(10).to_frame("correlación con posgrado")
    top.index.name = "variable"
    add("\n**Variables relacionadas con el grupo** (correlación con `posgrado`, sin `EDUCATION`):\n")
    add(md_tabla(top))
    auc_p = g["auc_proxy"]
    fuerza = "fuertes" if auc_p >= 0.7 else "moderados" if auc_p >= 0.6 else "débiles"
    add(f"\nPredecir si un cliente es de posgrado con las demás variables da un AUC (CV) de **{auc_p:.4f}** (0.5 = sin información): "
        f"los proxies son **{fuerza}**.\n")
    add("Perfil promedio por grupo (1 = posgrado):\n")
    perfil = g["X"].assign(grupo=g["group"]).groupby("grupo")[["LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1"]].mean()
    perfil.index.name = "grupo"
    add(md_tabla(perfil, nd=2))
    frase_proxy = {
        "fuertes": "Las demás variables reconstruyen bien la escolaridad (proxies fuertes), así que el modelo puede \"mirar\" el grupo aunque no se lo demos.",
        "moderados": "Las demás variables reconstruyen en parte la escolaridad (proxies moderados).",
        "débiles": "Las demás variables reconstruyen poco la escolaridad (proxies débiles)."}[fuerza]
    if persisten:
        add(f"\n**Conclusión:** las diferencias **no desaparecen** al quitar el atributo. {frase_proxy} "
            "Además, si las tasas reales de incumplimiento (pg) difieren entre grupos, un modelo que acierte puede rechazar "
            "a los grupos de forma distinta aun sin ver el atributo. No mirar no es sinónimo de ser equitativo.\n")
    else:
        add(f"\n**Conclusión:** las diferencias **se reducen mucho** al quitar el atributo. {frase_proxy} "
            "Aun así, no mirar no garantiza equidad y conviene medirla siempre por grupo.\n")

    # ------------------------------------------------------------------ (f)
    add("## (f) Memorándum al consejo (borrador, ≤ 1 página)\n")
    rp, rr = tabla_c.loc[P, "rechazo"] * 100, tabla_c.loc[R, "rechazo"] * 100
    fp, fr = tabla_c.loc[P, "FPR"] * 100, tabla_c.loc[R, "FPR"] * 100
    np_, nr_ = tabla_c.loc[P, "FNR"] * 100, tabla_c.loc[R, "FNR"] * 100
    n_prueba = int(tabla_c["n"].sum())
    ruido = ci_c.loc[FAIR, "¿contiene 0?"]
    tipo_error = "rechazar por error a buenos pagadores" if FAIR == "FPR" else "aprobar por error a quienes no pagarán"
    realidad = (f"Con los datos disponibles no podemos descartar que la diferencia entre grupos al {tipo_error} sea casualidad."
                if ruido else f"La diferencia entre grupos al {tipo_error} es estadísticamente real, no casualidad.")
    if FAIR == "FPR":
        propuesta = "que un cliente que sí iba a pagar tenga la misma probabilidad de ser aprobado, sea o no de posgrado"
        a_, b_ = tabla_d.loc[P, "FPR"] * 100, tabla_d.loc[R, "FPR"] * 100
        resultado = f"Con el ajuste, los buenos pagadores rechazados por error son {a_:.1f} y {b_:.1f} de cada 100 (posgrado y resto)."
        razon = "Es la que evita que un cliente cumplido sea rechazado por pertenecer a un grupo."
    else:
        propuesta = "que el banco deje pasar a los clientes que no pagarán con la misma frecuencia en ambos grupos"
        a_, b_ = tabla_d.loc[P, "FNR"] * 100, tabla_d.loc[R, "FNR"] * 100
        resultado = f"Con el ajuste, los clientes que no pagan y fueron aprobados son {a_:.1f} y {b_:.1f} de cada 100 (posgrado y resto)."
        razon = "Es la que iguala el riesgo que el banco asume en cada grupo."
    if persisten:
        quitar = (f"Quitar la escolaridad del modelo no basta: las diferencias se mantienen, en parte porque otras variables "
                  f"(la más ligada es {top.index[0]}) están relacionadas con ella.")
    else:
        quitar = "Quitar la escolaridad del modelo reduce las diferencias, pero no garantiza justicia; hay que seguir midiéndola."
    add("> **Para:** Consejo del banco. **De:** equipo de análisis. **Asunto:** ¿Tratamos igual a clientes con y sin posgrado?\n>")
    add(f"> **Qué encontramos.** Analizamos {n_total:,} clientes de un banco de Taiwán (2005) con un modelo que decide a quién aprobar una tarjeta; "
        f"los resultados vienen de {n_prueba:,} clientes que el modelo no había visto. "
        f"Con un solo criterio para todos, se rechaza al {rp:.1f} % de quienes tienen posgrado y al {rr:.1f} % del resto. "
        f"De cada 100 personas que sí habrían pagado, se rechaza por error a {fp:.1f} con posgrado y a {fr:.1f} del resto; "
        f"de cada 100 que no habrían pagado, se aprueba por error a {np_:.1f} y a {nr_:.1f}, respectivamente. {realidad} {quitar}\n>")
    add(f"> **Qué recomendamos.** Recomendamos {propuesta}. Se logra con un criterio de corte distinto por grupo. {resultado} "
        f"{razon} No es posible igualar todas las medidas de justicia a la vez, porque los grupos no tienen la misma tasa real "
        "de incumplimiento; por eso hay que elegir una.\n>")
    add(f"> **Cuánto cuesta.** La ganancia del banco baja {costo:,.0f} unidades de {base['ganancia']:,.0f} ({pct:.1f} %). "
        "Sugerimos revisar estas diferencias por grupo cada trimestre.\n")

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nInforme escrito en {ruta}")
