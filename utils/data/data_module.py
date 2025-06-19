# data_module.py
import numpy as np
from typing import Literal, Optional
from sklearn.model_selection import train_test_split

from .single_hdf5 import SingleHDF5


class DataModule:
    """
    Descarga dos datasets de Kaggle y separa train / val (estratificado).
    """

    def __init__(
        self,
        *,
        # --- TRAIN ---------------------------------------------------------
        train_kaggle_dataset_id: str,
        train_local_download_dir: str,
        # --- TEST ----------------------------------------------------------
        test_kaggle_dataset_id: str,
        test_local_download_dir: str,
        # ------------------------------------------------------------------
        keys: Optional[dict],
        train_pct: float,
        seed: int,
    ):
        if not 0.0 < train_pct < 1.0:
            raise ValueError("train_pct debe estar entre 0 y 1 (exclusivo).")

        self.seed = seed
        # 1) TRAIN / VAL ----------------------------------------------------
        self.trainset = SingleHDF5(
            kaggle_dataset_id=train_kaggle_dataset_id,
            local_download_dir=train_local_download_dir,
            keys=keys,
        )

        y = self.trainset.Y
        idx = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            idx,
            test_size=1.0 - train_pct,
            stratify=y,
            random_state=self.seed,
        )
        self.trainset.register_indices(train_idx, val_idx)

        # 2) TEST -----------------------------------------------------------
        self.testset = SingleHDF5(
            kaggle_dataset_id=test_kaggle_dataset_id,
            local_download_dir=test_local_download_dir,
            keys=keys,
        )

    # ————————————————— API pública —————————————————
    def get_arrays(self, split: Literal["train", "val", "test"]):
        if split in ("train", "val"):
            return self.trainset.get_arrays(split)
        if split == "test":
            return self.testset.get_arrays()
        raise ValueError("split debe ser 'train', 'val' o 'test'")

    def get_effects(self, split: Literal["train", "val", "test"], **kw):
        if split in ("train", "val"):
            return self.trainset.get_effects(split=split, **kw)
        if split == "test":
            return self.testset.get_effects(**kw)
        raise ValueError("split debe ser 'train', 'val' o 'test'")
    
    def to_tf_dataset(
        self,
        *,
        split: Literal["train", "val", "test"],
        batch_size: int,
        shuffle: bool = True,
        prefetch: bool = True,
        **kw,
    ):
        common_kw = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            prefetch=prefetch,
            seed=self.seed,
            **kw,
        )

        if split in ("train", "val"):
            return self.trainset.to_tf_dataset(split=split, **common_kw)
        if split == "test":
            return self.testset.to_tf_dataset(**common_kw)
        raise ValueError("split debe ser 'train', 'val' o 'test'")
