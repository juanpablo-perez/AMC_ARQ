"""
Carga la configuración de un experimento, convierte notebooks a .py si
es necesario, importa dinámicamente el modelo y prepara los datos
(locales o descargados de Kaggle) en formato NumPy o tf.data.Dataset.

"""

import yaml                          # Lectura y mezcla de archivos YAML
import sys               # Conversión IPython Notebook → .py
from pathlib import Path             # Manejo robusto de rutas
from importlib import import_module  # Import dinámico del modelo
from utils.data.data_module import DataModule



BASE_DIR = Path().resolve() 
CONFIG_ROOT = BASE_DIR / "configs"
MODELS_ROOT = BASE_DIR / "models"
DATA_ROOT   = BASE_DIR / "datasets"


def load_config(exp_name:str):
    exp_path = CONFIG_ROOT / "experiments" / f"{exp_name}.yaml"
    exp_cfg  = yaml.safe_load(exp_path.read_text())

    if "_base_" in exp_cfg:                                # herencia opcional
        base_cfg = yaml.safe_load((CONFIG_ROOT / exp_cfg["_base_"]).read_text())
        cfg = {**base_cfg, **exp_cfg}                      # exp > default
    else:
        cfg = exp_cfg
    return cfg

def load_experiment(
    exp_name: str,
    repeat_index: int,
    ):
    """
    Devuelve:
        cfg          → dict  (configuración combinada)
        ModelClass   → type  (sub‑clase de tu BaseTFModel)
        model_params → dict  (params filtrados para __init__)
        full_dataset 
        train_data   → (X,Y) tf.data.Dataset
        val_data     → idem
        val_indices
    """

    # ─────────────────── 1) Leer YAML ──────────────────────────
    cfg = load_config(exp_name=exp_name)
    seed = cfg["training"]["seed"] + 100 * repeat_index # Variación del seed

    # ─────────────────── 2) Notebook → Python (.py) ──────────────
    model_module = cfg["experiment"]["model_module"]
    py_path      = MODELS_ROOT / f"{model_module}.py"

    if not py_path.exists():
        raise ValueError(
            f"El módulo {model_module} no existe en {MODELS_ROOT}. "
            "Asegúrate de que el archivo .py esté presente."
        )


    # ─────────────────── 3) Import dinámico del modelo ────────────────────
    sys.path.append(str(MODELS_ROOT))
    module      = import_module(model_module)
    ModelClass  = getattr(module, cfg["experiment"]["model_class"])

    # ─────────────────── 4) Filtrar parámetros válidos ────────────────────
    raw_params   = cfg["model"]["params"]
    model_params = {k: v for k, v in raw_params.items()}

    # ─────────────────── 5) Preparar Dataset (Kaggle) ─────────────
    ds_cfg = cfg["dataset"]

    train_dl_cfg = ds_cfg["kaggle"]["train"]
    test_dl_cfg  = ds_cfg["kaggle"]["test"]

    datamodule = DataModule(
        train_kaggle_dataset_id = train_dl_cfg["dataset_id"],
        train_local_download_dir = train_dl_cfg.get("download_dir"),
        test_kaggle_dataset_id  = test_dl_cfg["dataset_id"],
        test_local_download_dir = test_dl_cfg.get("download_dir"),
        keys       = ds_cfg.get("keys"),
        train_pct  = ds_cfg["train_pct"],
        seed       = seed,
    )
    
    # // Modificar subdirectorio de acuerdo a número actual de repetición  \\
    cfg["experiment"]["output_subdir"] = cfg["experiment"]["output_subdir"] + "/" + f"rep_{repeat_index}"
    


    # tf.data.Dataset
    bs = cfg["training"].get("batch_size", 32)

    train_ds = datamodule.to_tf_dataset(
        split="train", batch_size=bs, shuffle=True,  prefetch=True
    )
    val_ds   = datamodule.to_tf_dataset(
        split="val",   batch_size=bs, shuffle=False, prefetch=True
    )
    test_ds_idx = datamodule.to_tf_dataset(
        split="test", batch_size=bs,
        shuffle=False, prefetch=False, include_index=True
    )

    # ─────────────────── 6) Return ────────────────────────────────────────
    return cfg, ModelClass, model_params, datamodule, train_ds, val_ds, test_ds_idx
