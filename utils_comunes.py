"""Utilidades compartidas entre el notebook de línea base y el del modelo secuencial.

Se definen una sola vez para que la partición temporal y la evaluación de los
modelos A y B sean exactamente la misma función en ambos notebooks — así la
comparación entre ambos es válida por construcción, no por copiar y pegar
código y confiar en que quedó igual.

Uso: desde cualquiera de los notebooks del proyecto (viven en la raíz),
    from utils_comunes import particionar_temporal, evaluar_modelo, resumen_economico
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def particionar_temporal(df, col_tiempo="marca_tiempo", frac_train=0.70, frac_val=0.15):
    """Corta `df` en train/val/test por fecha, nunca al azar.

    Los cortes se calculan sobre los cuantiles de `col_tiempo`, así que las tres
    particiones son contiguas en el tiempo: train = lo más antiguo, val = lo
    intermedio, test = lo más reciente. Nada del futuro se mezcla con el pasado.

    Devuelve (train, val, test, cortes), con `cortes = (corte_train_val,
    corte_val_test)` para que ambos notebooks corten exactamente en el mismo punto.
    """
    tiempos_ordenados = df[col_tiempo].sort_values()
    n = len(tiempos_ordenados)
    corte_train_val = tiempos_ordenados.iloc[int(n * frac_train)]
    corte_val_test = tiempos_ordenados.iloc[int(n * (frac_train + frac_val))]

    train = df[df[col_tiempo] < corte_train_val].copy()
    val = df[(df[col_tiempo] >= corte_train_val) & (df[col_tiempo] < corte_val_test)].copy()
    test = df[df[col_tiempo] >= corte_val_test].copy()

    return train, val, test, (corte_train_val, corte_val_test)


def evaluar_modelo(y_true, y_score, umbral=0.5):
    """AUC-PR + precisión/exhaustividad/F1 en `umbral`. Métrica común A vs. B.

    No se reporta exactitud (accuracy) a propósito: con ~1% de fraude, un
    modelo que nunca marca nada ya tendría ~99% de exactitud sin ser útil.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= umbral).astype(int)

    return {
        "auc_pr": average_precision_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "exhaustividad": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "umbral": umbral,
        "n_positivos_predichos": int(y_pred.sum()),
        "n_positivos_reales": int(y_true.sum()),
    }


def curva_precision_exhaustividad(y_true, y_score):
    """Curva precisión-exhaustividad completa, para graficar A y B juntos."""
    precision, exhaustividad, umbrales = precision_recall_curve(y_true, y_score)
    return precision, exhaustividad, umbrales


def barrer_umbrales(y_true, y_score, costo_fn=4200.0, costo_fp=180.0, umbrales=None):
    """Costo total (no mensual todavía) para cada umbral: FN·costo_fn + FP·costo_fp."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if umbrales is None:
        umbrales = np.linspace(0.0, 1.0, 101)

    filas = []
    for u in umbrales:
        y_pred = (y_score >= u).astype(int)
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        filas.append({"umbral": u, "fn": fn, "fp": fp, "tp": tp, "costo": fn * costo_fn + fp * costo_fp})

    return pd.DataFrame(filas)


def resumen_economico(y_true, y_score, dias_periodo, costo_fn=4200.0, costo_fp=180.0, umbrales=None):
    """Umbral que minimiza el costo mensual estimado, y el ahorro frente a no usar modelo.

    Supuesto explícito: no se cuenta con las predicciones del sistema antifraude
    actual para comparar directamente, así que el punto de referencia es "no
    interceptar nada" (todo el fraude pasa, cero falsos positivos). El ahorro
    reportado es frente a ese piso, no frente al sistema de reglas real — es una
    limitación que debe quedar igual de clara en el informe.

    `dias_periodo` es el número de días que cubre `y_true`/`y_score` (p. ej. el
    tamaño del conjunto de prueba); se usa para escalar el costo a "por mes"
    (30 días) y así comparar particiones de distinto tamaño en las mismas unidades.
    """
    barrido = barrer_umbrales(y_true, y_score, costo_fn, costo_fp, umbrales)
    fila_optima = barrido.loc[barrido["costo"].idxmin()]

    escala_mensual = 30.0 / dias_periodo
    n_fraudes = int(np.asarray(y_true).sum())
    costo_sin_modelo = n_fraudes * costo_fn

    return {
        "umbral_optimo": float(fila_optima["umbral"]),
        "costo_mensual_con_modelo": float(fila_optima["costo"] * escala_mensual),
        "costo_mensual_sin_modelo": float(costo_sin_modelo * escala_mensual),
        "ahorro_mensual": float((costo_sin_modelo - fila_optima["costo"]) * escala_mensual),
        "fn_en_optimo": int(fila_optima["fn"]),
        "fp_en_optimo": int(fila_optima["fp"]),
        "barrido": barrido,
    }
