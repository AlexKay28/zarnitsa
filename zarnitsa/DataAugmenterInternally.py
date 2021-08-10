from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .DataAugmenter import AbstractDataAugmenter


class DataAugmenterInternally(AbstractDataAugmenter):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        aug_type="normal",
        freq=0.2,
        return_only_aug=False,
    ) -> pd.DataFrame:
        """Augment dataframe data. Pandas dataframe"""
        augment_column_method = {
            "normal": self.augment_column_norm,
            "uniform": self.augment_column_uniform,
            "permutations": self.augment_column_permut,
        }
        not_to_aug, to_aug = self._prepare_data_to_aug(df, freq=freq)
        for col in df.columns:
            to_aug[col] = augment_column_method[aug_type](
                to_aug[col], freq=1.0
            )
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column(
        self,
        col: pd.Series,
        aug_type="normal",
        freq=0.2,
        return_only_aug=False,
    ) -> pd.Series:
        """Augment Serial data. Pandas column"""
        augment_column_method = {
            "normal": self.augment_column_norm,
            "uniform": self.augment_column_uniform,
            "permutations": self.augment_column_permut,
        }
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        to_aug = augment_column_method[aug_type](to_aug, freq=1.0)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def _prepare_data_to_aug(
        self, data, freq=0.2
    ) -> Tuple[pd.Series, pd.Series]:
        """Get part of data. Not augment all of it excep case freq=1.0"""
        data = (
            pd.Series(data)
            if type(data) is not pd.Series and type(data) is not pd.DataFrame
            else data
        )
        if freq < 1:
            not_to_aug, to_aug = train_test_split(data, test_size=freq)
            return not_to_aug, to_aug
        elif freq == 1:
            return data.sample(0), data
        elif freq == 0:
            return data, data.sample(0)

    def augment_column_permut(
        self, col: pd.Series, freq=0.2, return_only_aug=False
    ) -> pd.Series:
        """Augment column data using permutations. Pandas column"""
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        indices_to_permute = to_aug.index
        to_aug = to_aug.sample(frac=1.0)
        to_aug.index = indices_to_permute
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column_norm(
        self, col: pd.Series, freq=0.2, return_only_aug=False
    ) -> pd.Series:
        """Augment column data using normal distribution. Pandas column"""
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        column_std = col.std()
        to_aug = to_aug.apply(
            lambda value: np.random.normal(value, column_std)
        )
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column_uniform(
        self, col: pd.Series, freq=0.2, n_sigm=3, return_only_aug=False
    ) -> pd.Series:
        """Augment column data using uniform distribution. Pandas column"""
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        column_std = col.std()
        to_aug = to_aug.apply(
            lambda value: np.random.uniform(
                value - n_sigm * column_std, value + n_sigm * column_std
            )
        )
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])
