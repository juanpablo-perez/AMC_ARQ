# result_utils.py
# --------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import json, re
import pandas as pd
import numpy as np
from typing import List, Tuple

# --------------------------------------------------------------
# ★ 1. Descubrimiento de pares JSON
# --------------------------------------------------------------
def discover_jsons(root: Path) -> List[Tuple[Path, Path]]:
    """
    Recorre *root* y devuelve una lista de tuplas
    (classification_report.json, effects_report.json).

    Se asume que ambos archivos viven en .../reports/
    """
    rep_paths = sorted(root.rglob("*/reports/classification_report.json"))
    pairs: List[Tuple[Path, Path]] = []
    for rep in rep_paths:
        eff = rep.with_name("effects_report.json")
        if eff.exists():
            pairs.append((rep, eff))
    return pairs


# --------------------------------------------------------------
# ★ 2. Parsers de metadatos a partir de la ruta
# --------------------------------------------------------------
_META_REP = re.compile(r"rep_(\d+)")
_META_FOLD = re.compile(r"fold_(\d+)")

def parse_meta(p: Path) -> Tuple[str, str, int, int | None]:
    """
    Devuelve (architecture, scenario, repeat, fold)
    tomando los nombres de carpeta *ARQ_* y *ESC_*.

    Estructura esperada:
    ⋯/ARQ_1/ESC_3/rep_0/fold_1/reports/classification_report.json
    o
    ⋯/ARQ_1/ESC_3/rep_0/reports/classification_report.json
    """
    parts = p.parts
    # busca los índices donde aparecen los tokens
    idx_arq = next(i for i, part in enumerate(parts) if part.startswith("ARQ_"))
    arch = parts[idx_arq]
    esc  = parts[idx_arq + 1]                # ESC_*
    rep  = int(_META_REP.search(parts[idx_arq + 2])[1])
    fold = None
    if "fold_" in parts[idx_arq + 3]:
        fold = int(_META_FOLD.search(parts[idx_arq + 3])[1])
    return arch, esc, rep, fold


# --------------------------------------------------------------
# ★ 3. Carga de *classification_report.json*
# --------------------------------------------------------------
def _flatten_class_report(js: dict) -> pd.DataFrame:
    """
    Convierte la sección 'classification_report' en un DF "largo":

    cols: class | metric | value
    Incluye clases reales, 'macro avg', 'weighted avg' y 'accuracy'.
    """
    rows = []
    rep = js["classification_report"]

    # (a) Clases y promedios
    for cls, metrics in rep.items():
        if cls == "accuracy":
            continue  # se maneja aparte
        for metric, value in metrics.items():
            rows.append({"class": cls, "metric": metric, "value": value})

    # (b) Accuracy global
    rows.append({"class": "overall", "metric": "accuracy", "value": rep["accuracy"]})
    return pd.DataFrame(rows)


def load_reports(pairs: List[Tuple[Path, Path]]
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve dos DataFrames:

    1. *df_runs*: una fila por corrida con métricas globales
       (accuracy, loss, macro/weighted precision-recall-F1).
    2. *df_classes*: una fila por (corrida × clase × métrica).

    Ambos incluyen columnas de contexto: architecture, scenario, repeat, fold.
    """
    run_rows   = []
    class_rows = []

    for rep_path, _ in pairs:
        with open(rep_path) as f:
            js = json.load(f)
        arch, esc, rep, fold = parse_meta(rep_path)

        # ---------- resumen ----------
        glob = js["classification_report"]
        macro = glob["macro avg"]
        wavg  = glob["weighted avg"]

        run_rows.append({
            "architecture" : arch,
            "scenario"     : esc,
            "repeat"       : rep,
            "fold"         : fold,
            "accuracy"     : js["evaluation"]["accuracy"],
            "loss"         : js["evaluation"]["loss"],
            "macro_precision"  : macro["precision"],
            "macro_recall"     : macro["recall"],
            "macro_f1"         : macro["f1-score"],
            "weighted_precision": wavg["precision"],
            "weighted_recall"   : wavg["recall"],
            "weighted_f1"       : wavg["f1-score"],
            "report_path"  : rep_path,
        })

        # ---------- por clase ----------
        df_flat = _flatten_class_report(js)
        df_flat["architecture"] = arch
        df_flat["scenario"]     = esc
        df_flat["repeat"]       = rep
        df_flat["fold"]         = fold
        class_rows.append(df_flat)

    df_runs    = pd.DataFrame(run_rows)
    df_classes = pd.concat(class_rows, ignore_index=True)

    return df_runs, df_classes


# --------------------------------------------------------------
# ★ 4. Carga de *effects_report.json* (igual, pero tipado)
# --------------------------------------------------------------
def load_effects(pairs: List[Tuple[Path, Path]]) -> pd.DataFrame:
    """
    Devuelve DataFrame largo:

    architecture | scenario | param | bin | accuracy
    """
    long_rows = []
    for rep_path, eff_path in pairs:
        with open(eff_path) as f:
            js = json.load(f)
        arch, esc, rep, fold = parse_meta(rep_path)
        for param, info in js["effects"].items():
            for bin_lbl, vals in info["values"].items():
                acc = vals.get("Éxito") or vals.get("accuracy")
                long_rows.append({
                    "architecture": arch,
                    "scenario"    : esc,
                    "param"       : param,
                    "bin"         : bin_lbl,
                    "accuracy"    : acc,
                    "repeat"      : rep,
                    "fold"        : fold
                })
    return pd.DataFrame(long_rows)
