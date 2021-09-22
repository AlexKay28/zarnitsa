import os
from typing import Tuple

from scipy.signal import lfilter, savgol_filter
import numpy as np
import pandas as pd

from ..DataAugmenter import AbstractDataAugmenter


class DataAugmenterTimeSeries(AbstractDataAugmenter):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def augment_dataframe(
        self, df: pd.DataFrame, columns=None, **kwargs
    ) -> pd.DataFrame:
        """Augment dataframe data. Pandas dataframe"""
        columns = columns if columns else df.columns
        for col in columns:
            df[col] = self.augment_column(df[col], **kwargs)
        return df

    def augment_column(
        self,
        col: pd.Series,
        func: str,
        **kwargs,
    ) -> pd.Series:
        """Augment dataframe data. Pandas dataframe"""
        func = {
            "dropping": self.data_dropping,
            "data_interpolation": self.data_interpolation,
            "data_extrapolation": self.data_extrapolation,
            "data_denoiser": self.data_denoiser,
            "data_noiser": self.data_noiser,
        }
        return func[func_name](col, **kwargs)

    def _prepare_data_to_aug(self, data, freq=0.2) -> Tuple[pd.Series, pd.Series]:
        pass

    @staticmethod
    def data_dropping(col, limit=None, freq=0.2, **kwargs) -> pd.Series:
        """
        Extrapolate pandas column
        """
        if not col.shape[0]:
            raise ValueError(f"Iterable object <{type(col)}> is empty! Check input!")
        if limit:
            freq = limit / col.shape[0]
        drop_indices = col.sample(frac=freq).index
        return col[[i for i in col.index if i not in drop_indices]]

    @staticmethod
    def data_interpolation(
        col, method="linear", limit=None, freq=1.0, order=2, **kwargs
    ) -> pd.Series:
        """
        Interpolate pandas column
        filling the gaps
        """
        if not col.shape[0]:
            raise ValueError(f"Iterable object <{type(col)}> is empty! Check input!")
        if not col[col.isna()].shape[0]:
            return col
        if limit is None:
            limit = int(col[col.isna()].shape[0] * freq)

        if method in ("fillna",):
            col = col.fillna(kwargs["value_to_fill"], limit=limit)
        elif method in ("linear", "pad", "polynomial"):
            col = col.interpolate(method=method, limit=limit, **kwargs)
        elif method in ("polyfit",):
            non_missed, missed = col[~col.isna()], col[col.isna()]
            poly = np.poly1d(np.polyfit(non_missed.index, non_missed.values, order))
            na_records = missed.sample(frac=limit / missed.shape[0])
            col.loc[na_records.index] = poly(na_records.index)
        else:
            raise KeyError(f"Undefined method <{method}>!")

        return col

    @staticmethod
    def data_extrapolation(
        col, method="polyfit", limit=None, freq=0.2, h=None, **kwargs
    ) -> pd.Series:
        """
        Extrapolate pandas column
        """
        if not col.shape[0]:
            raise ValueError(f"Iterable object <{type(col)}> is empty! Check input!")
        if limit is None:
            limit = int(col[col.isna()].shape[0] * freq)
        if h is None:
            h = (max(col.index) - min(col.index)) / col.shape[0]

        if method in ("polyfit",):
            poly = np.poly1d(np.polyfit(col.index, col.values, kwargs["order"]))
            indices_to_extrapolate = [
                col.index[-1] + (h * (t + 1)) for t in range(limit)
            ]  # creeate N extra time steps using indices
            col = col.append(
                pd.Series(
                    data=poly(indices_to_extrapolate), index=indices_to_extrapolate
                )
            )
        else:
            raise KeyError(f"Undefined method <{method}>!")
        return col

    @staticmethod
    def data_denoiser(col, method="lfilter", **kwargs) -> pd.Series:
        """
        Denoise pandas column
        """
        if not col.shape[0]:
            raise ValueError(f"Iterable object <{type(col)}> is empty! Check input!")
        if method == "lfilter":
            numerator = kwargs["numerator"]
            denominator = kwargs["denominator"]
            col = pd.Series(
                data=lfilter(numerator, denominator, col.values),
                index=col.index,
            )
        elif method == "savgol_filter":
            polyorder = kwargs["polyorder"]
            window_length = kwargs["window_length"]
            col = pd.Series(
                data=savgol_filter(col.values, polyorder, window_length),
                index=col.index,
            )
        else:
            raise KeyError(f"Undefined method '{method}'!")
        return col

    @staticmethod
    def data_noiser(col, method="normal", limit=None, freq=0.2, **kwargs) -> pd.Series:
        """
        Noise pandas column
        """
        if not col.shape[0]:
            raise ValueError(f"Iterable object <{type(col)}> is empty! Check input!")
        if limit is None:
            limit = int(col[col.isna()].shape[0] * freq)

        if method in ("normal",):
            target_to_noize = col.sample(frac=limit / col.shape[0])
            noise = np.random.normal(
                kwargs["mean"], kwargs["sig"], target_to_noize.shape[0]
            )
            col.loc[target_to_noize.index] = target_to_noize.values + noise
        else:
            raise KeyError(f"Undefined method <{method}>!")
        return col
