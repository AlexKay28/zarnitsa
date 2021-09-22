import numpy as np
import pandas as pd

from ..DataAugmenter import AbstractDataAugmenter


class DataAugmenterExternally(AbstractDataAugmenter):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def augment_dataframe(
        self, df: pd.DataFrame, aug_params="normal", **kwargs
    ) -> pd.DataFrame:
        """Augment dataframe data. Pandas dataframe"""
        if isinstance(aug_params, str):
            print("Single type of augmentation")
            for col_name in df.columns:
                df[col_name] = self.augment_column(
                    df[col_name], aug_params=aug_params, **kwargs
                )
        elif isinstance(aug_params, dict):
            print("Multiple types of augmentation")
            for col_name, params in aug_params.items():
                df[col_name] = self.augment_column(df[col_name], **params)
        else:
            raise KeyError("Bad type of aug_params variable")
        return df

    def augment_column(self, col: pd.Series, aug_type="normal", **kwargs) -> pd.Series:
        """Augment Serial data. Pandas column"""
        if not col.shape[0]:
            raise ValueError(f"Iterable object <{type(col)}> is empty! Check input!")
        if all(col.isna()):
            col = pd.Series(
                self.augment_distrib_random(aug_type=aug_type, size=col.shape[0])
            )
        else:
            col = col.apply(
                lambda v: v
                if not np.isnan(v)
                else self.augment_distrib_random(aug_type, **kwargs)
            )
        return col

    def _prepare_data_to_aug(self, col: pd.Series, freq=0.2) -> pd.Series:
        raise NotImplemented("External augmentation doesn't utilize data split")

    def augment_distrib_random(self, aug_type="normal", size=1, **kwargs):
        """Return float or array depends on needed size. If size is 1 - returns array of size 1"""
        np_func = getattr(np.random, aug_type, None)
        if np_func:
            return np_func(size=size, **kwargs)
        else:
            raise KeyError(f"Unknown aug type: <{aug_type}>")
