from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .DataAugmenter import AbstractDataAugmenter


class DataAugmenterInternally(AbstractDataAugmenter):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def augment_dataframe(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Augment dataframe data. Pandas dataframe"""
        augment_column_method = {
            "normal": self.augment_column_norm,
            "uniform": self.augment_column_uniform,
            "permutations": self.augment_column_permut,
        }
        not_to_aug, to_aug = self._prepare_data_to_aug(df, freq=kwargs["freq"])
        kwargs["freq"] = 1.0
        for col in df.columns:
            to_aug[col] = augment_column_method[aug_type](to_aug[col], **kwargs)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column(self, col: pd.Series, aug_type="permutations", **kwargs) -> pd.Series:
        """Augment Serial data. Pandas column"""
        augment_column_method = {
            "normal": self.augment_column_norm,
            "uniform": self.augment_column_uniform,
            "permutations": self.augment_column_permut,
        }
        to_aug = augment_column_method[aug_type](col, **kwargs)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def _prepare_data_to_aug(self, data, freq=0.2) -> Tuple[pd.Series, pd.Series]:
        """
        Get part of data. Not augment all of it excep case freq=1.0
        params: data: iterable data or pd.Series object
        params: freq: part of the data which will be the base for augmentation
        """
        data = pd.Series(data) if not isinstance(data, pd.Series) else data
        if freq < 1:
            not_to_aug, to_aug = train_test_split(data, test_size=freq)
            return not_to_aug, to_aug
        elif freq == 1:
            return data.sample(0), data
        elif freq == 0:
            return data, data.sample(0)

    def augment_column_permut(
        self, col: pd.Series, n_to_aug=None, freq=0.2, return_only_aug=False
    ) -> pd.Series:
        """
        Augment column data using permutations. Pandas column
        params: col: pandas series object
        params: n_to_aug: defined number of augmented examples
        params: freq: part of the data which will be the base for augmentation
        params: return_only_aug: ask return only augmented data
        """
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        indices_to_permute = to_aug.index
        to_aug = to_aug.sample(frac=1.0)
        to_aug.index = indices_to_permute
        if n_to_aug:
            n_to_aug = n_to_aug if n_to_aug < to_aug.shape[0] else to_aug.shape[0]
            to_aug = to_aug.sample(n_to_aug)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column_norm(
        self, col: pd.Series, n_to_aug=None, freq=0.2, return_only_aug=False
    ) -> pd.Series:
        """
        Augment column data using normal distribution. Pandas column
        params: col: pandas series object
        params: n_to_aug: defined number of augmented examples
        params: freq: part of the data which will be the base for augmentation
        params: return_only_aug: ask return only augmented data
        """
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        column_std = col.std()
        to_aug = to_aug.apply(lambda value: np.random.normal(value, column_std))
        if n_to_aug:
            n_to_aug = n_to_aug if n_to_aug < to_aug.shape[0] else to_aug.shape[0]
            to_aug = to_aug.sample(n_to_aug)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column_uniform(
        self, col: pd.Series, n_to_aug=None, freq=0.2, n_sigm=3, return_only_aug=False
    ) -> pd.Series:
        """
        Augment column data using uniform distribution. Pandas column
        params: col: pandas series object
        params: n_to_aug: defined number of augmented examples
        params: freq: part of the data which will be the base for augmentation
        params: n_sigm: the size of std and terms of sigma value
        params: return_only_aug: ask return only augmented data
        """
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        column_std = col.std()
        to_aug = to_aug.apply(
            lambda value: np.random.uniform(
                value - n_sigm * column_std, value + n_sigm * column_std
            )
        )
        if n_to_aug:
            n_to_aug = n_to_aug if n_to_aug < to_aug.shape[0] else to_aug.shape[0]
            to_aug = to_aug.sample(n_to_aug)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])
