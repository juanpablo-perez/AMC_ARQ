# single_hdf5.py
import os, h5py, numpy as np
from pathlib import Path
import tensorflow as tf

# Credenciales (puedes exportarlas en tu entorno y borrar estas líneas)
os.environ["KAGGLE_USERNAME"] = "ilikepizzaanddrones"
os.environ["KAGGLE_KEY"]      = "b7d0370fced8eb934d226172fff8221f"

try:
    from kaggle import KaggleApi
except ModuleNotFoundError:
    raise ImportError("pip install kaggle")

# ────────────────────────────────────────────────────────────────────
class SingleHDF5:
    """
    Envuelve *un* .hdf5 proveniente de Kaggle.

    Parámetros
    ----------
    kaggle_dataset_id : str
        slug «user/dataset» (ej. "ilikepizzaanddrones/modulated-iq-signals")
    local_download_dir : str | Path
        carpeta donde se guardará (y se buscará) el .hdf5
    keys : dict | None
        nombres de los grupos dentro del HDF5 (default {"X","Y","Z"})
    """

    def __init__(
        self,
        *,
        kaggle_dataset_id: str,
        local_download_dir: str,
        keys: dict,
    ) -> None:

        # 0) Descarga / búsqueda local
        file_path = self._download_if_needed(kaggle_dataset_id, local_download_dir)

        # 1) Lectura a memoria
        self.keys = keys or {"X": "X", "Y": "Y", "Z": "Z"}
        with h5py.File(file_path, "r") as f:
            self.X = f[self.keys["X"]][:]
            self.Y = f[self.keys["Y"]][:]
            self.Z = f[self.keys["Z"]][:] if self.keys["Z"] in f else None

            if "Effects" in f:
                grp   = f["Effects"]
                dtype = [(n, grp[n].dtype) for n in grp]
                eff   = np.empty(len(self.X), dtype=dtype)
                for n in grp: eff[n] = grp[n][:]
                self.Effects = eff
            else:
                self.Effects = None

        # índices activos (se sobre-escriben desde DataModule)
        n = len(self.X)
        self.train_idx = np.arange(n, dtype=np.int64)
        self.val_idx   = np.empty(0, dtype=np.int64)

    # ───────────────────────── helpers ──────────────────────────
    @staticmethod
    def _download_if_needed(kaggle_dataset_id, local_dir):
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        h5_files = sorted(local_dir.rglob("*.hdf5"))
        if not h5_files:          # primera vez
            print(f"⬇️  Descargando «{kaggle_dataset_id}» …")
            api = KaggleApi(); api.authenticate()
            api.dataset_download_files(
                kaggle_dataset_id,
                path=str(local_dir),
                unzip=True,
                quiet=False,
            )
            h5_files = sorted(local_dir.rglob("*.hdf5"))

        if not h5_files:
            raise FileNotFoundError("No se encontró ningún .hdf5 en el zip")
        if len(h5_files) > 1:
            raise ValueError("Hay >1 .hdf5 descargado; limpia la carpeta o elige uno.")
        return h5_files[0]

    # ── API mínima (igual que antes) ─────────────────────────────
    def register_indices(self, train_idx, val_idx):
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.val_idx   = np.asarray(val_idx,   dtype=np.int64)

    def get_arrays(self, split: str | None = None):
        if split is None: return self.X, self.Y
        split = split.lower()
        if split == "train": return self.X[self.train_idx], self.Y[self.train_idx]
        if split == "val":   return self.X[self.val_idx],   self.Y[self.val_idx]
        raise ValueError("split debe ser 'train' o 'val'")

    # ————————————————————————————————————————————————————————
    def get_effects(
        self,
        *,
        split: str | None = None,
        fields: list[str] | None = None,
    ):
        """
        Devuelve un structured-array con los efectos alineados al `split`.

        Parameters
        ----------
        split : "train" | "val" | None
            None ⇒ dataset completo (o testset completo si proviene del DataModule).
        fields : list[str] | None
            Sub-conjunto de columnas a devolver. None ⇒ todas.
        """
        if self.Effects is None:
            raise ValueError("Este HDF5 no contiene grupo 'Effects'.")

        # Selección de índices según split
        if split is None:
            idx = (
                np.arange(len(self.X))               # testset completo
                if (not hasattr(self, "train_idx"))   # por seguridad
                else self.train_idx                   # SingleHDF5 sin register
            )
        else:
            split = split.lower()
            if split == "train":
                idx = self.train_idx
            elif split == "val":
                idx = self.val_idx
            else:
                raise ValueError("split debe ser 'train', 'val' o None")

        eff = self.Effects[idx]              # vista alineada
        if fields is not None:
            eff = eff[fields].copy()
        return eff
    
    # ────────────────────────────────────────────────                    
    def to_tf_dataset(
        self,
        *,                                      
        split: str | None = None,
        batch_size: int,
        shuffle: bool = True,
        seed: int,
        prefetch: bool = True,
        include_index: bool = False,
        buffer_size: int | None = None,
    ):
        """
        Devuelve un tf.data.Dataset con (X, Y) o (X, Y, idx).

        Parameters
        ----------
        split : "train" | "val" | None
            None ⇒ dataset completo (sin barajar).
        include_index : bool
            Si True, añade el índice absoluto dentro del HDF5
            (útil para métricas por muestra).
        buffer_size : int | None
            Tamaño del «shuffle buffer». Por defecto = len(split).
        """

        Xs, Ys = self.get_arrays(split)

        # --- índices opcionales ------------------------------------------------
        if include_index:
            if split == "train":
                idx = self.train_idx
            elif split == "val":
                idx = self.val_idx
            else:                               # split None  (o testset completo)
                idx = np.arange(len(self.X), dtype=np.int64)

            ds = tf.data.Dataset.from_tensor_slices((Xs, Ys, idx))
        else:
            ds = tf.data.Dataset.from_tensor_slices((Xs, Ys))

        # --- barajado sólo en train -------------------------------------------
        if shuffle and (split in (None, "train")):
            ds = ds.shuffle(
                buffer_size or len(Xs),
                seed=seed,
                reshuffle_each_iteration=True,
            )

        ds = ds.batch(batch_size)
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

