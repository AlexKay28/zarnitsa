from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split

from ..DataAugmenter import AbstractDataAugmenter


class DataAugmenterInternally(AbstractDataAugmenter):
    __available_imputers = ["simple", "knn"]

    def __init__(self, imputer_name="simple", n_jobs=1):
        self.imputer_name = imputer_name
        self.n_jobs = n_jobs

        # this variables can be configured manually
        self.imputer_n_neighbors = 3
        self.missing_values = np.nan
        self.imputer_strategy = "mean"

    def augment_dataframe(
        self, df: pd.DataFrame, aug_type="permutations", **kwargs
    ) -> pd.DataFrame:
        """
        Augment dataframe data. Pandas dataframe
        param: aug_type: type of augmentation approach
        param: col: pandas series object
        param: n_to_aug: defined number of augmented examples
        param: freq: part of the data which will be the base for augmentation
        param: n_sigm: the size of std and terms of sigma value
        param: return_only_aug: ask return only augmented data
        """
        augment_column_method = {
            "normal": self.augment_column_norm,
            "uniform": self.augment_column_uniform,
            "permutations": self.augment_column_permut,
        }
        not_to_aug, to_aug = self._prepare_data_to_aug(df, freq=kwargs["freq"])
        kwargs["freq"] = 1.0
        for col in df.columns:
            to_aug[col] = augment_column_method[aug_type](to_aug[col], **kwargs)
        return to_aug

    def augment_column(
        self, col: pd.Series, aug_type="permutations", **kwargs
    ) -> pd.Series:
        """
        Augment Serial data. Pandas column
        param: aug_type: type of augmentation approach
        param: col: pandas series object
        param: n_to_aug: defined number of augmented examples
        param: freq: part of the data which will be the base for augmentation
        param: n_sigm: the size of std and terms of sigma value
        param: return_only_aug: ask return only augmented data
        """
        augment_column_method = {
            "normal": self.augment_column_norm,
            "uniform": self.augment_column_uniform,
            "permutations": self.augment_column_permut,
        }
        to_aug = augment_column_method[aug_type](col, **kwargs)
        return to_aug

    def _apply_imputation(self, data):
        if self.imputer_name == "simple":
            imputer = SimpleImputer(
                missing_values=self.missing_values, strategy=self.imputer_strategy
            )
            data = imputer.fit_transform(data)
        elif self.imputer_name == "knn":
            imputer = KNNImputer(n_neighbors=self.imputer_n_neighbors)
            data = imputer.fit_transform(data)
        else:
            raise KeyError(
                f"Unknown imputer name <{self.imputer_name}>. "
                f"Choose from: {','.join(self.__available_imputers)}."
            )
        return data

    def _prepare_data_to_aug(self, data, freq=0.2) -> Tuple[pd.Series, pd.Series]:
        """
        Get part of data. Not augment all of it except case freq=1.0
        param: data: iterable data or pd.Series object
        param: freq: part of the data which will be the base for augmentation
        """
        data = (
            pd.Series(data) if not isinstance(data, (pd.DataFrame, pd.Series)) else data
        )
        if 0 < freq < 1:
            not_to_aug, to_aug = train_test_split(data, test_size=freq)
            return not_to_aug, to_aug
        elif freq == 1:
            return data.sample(0), data
        elif freq == 0:
            return data, data.sample(0)
        else:
            raise ValueError("freq value not in [0, 1] span")

    def augment_column_permut(
        self, col: pd.Series, n_to_aug=0, freq=0.2, return_only_aug=False
    ) -> pd.Series:
        """
        Augment column data using permutations. Pandas column
        param: col: pandas series object
        param: n_to_aug: defined number of augmented examples
        param: freq: part of the data which will be the base for augmentation
        param: return_only_aug: ask return only augmented data
        """
        if not col.shape[0]:
            raise ValueError(f"Iterable object <{type(col)}> is empty! Check input!")
        if n_to_aug == 0 and not return_only_aug:
            return col
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        indices_to_permute = to_aug.index
        to_aug = to_aug.sample(frac=1.0)
        to_aug.index = indices_to_permute
        n_to_aug = min(n_to_aug, to_aug.shape[0])
        to_aug = to_aug.sample(n_to_aug)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column_norm(
        self, col: pd.Series, n_to_aug=0, freq=0.2, n_sigm=3, return_only_aug=False
    ) -> pd.Series:
        """
        Augment column data using normal distribution. Pandas column
        param: col: pandas series object
        param: n_to_aug: defined number of augmented examples
        param: freq: part of the data which will be the base for augmentation
        param: n_sigm: the size of std and terms of sigma value
        param: return_only_aug: ask return only augmented data
        """
        if not col.shape[0]:
            raise ValueError(f"Iterable object <{type(col)}> is empty! Check input!")
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        column_std = col.std()
        to_aug = to_aug.apply(
            lambda value: np.random.normal(value, n_sigm * column_std)
        )
        if n_to_aug:
            n_to_aug = min(n_to_aug, to_aug.shape[0])
            to_aug = to_aug.sample(n_to_aug)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column_uniform(
        self, col: pd.Series, n_to_aug=0, freq=0.2, n_sigm=3, return_only_aug=False
    ) -> pd.Series:
        """
        Augment column data using uniform distribution. Pandas column
        param: col: pandas series object
        param: n_to_aug: defined number of augmented examples
        param: freq: part of the data which will be the base for augmentation
        param: n_sigm: the size of std and terms of sigma value
        param: return_only_aug: ask return only augmented data
        """
        if not col.shape[0]:
            raise ValueError(f"Iterable object <{type(col)}> is empty! Check input!")
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        column_std = col.std()
        to_aug = to_aug.apply(
            lambda value: np.random.uniform(
                value - n_sigm * column_std, value + n_sigm * column_std
            )
        )
        if n_to_aug:
            n_to_aug = min(n_to_aug, to_aug.shape[0])
            to_aug = to_aug.sample(n_to_aug)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])
