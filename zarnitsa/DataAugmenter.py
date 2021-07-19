import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod


class AbstractDataAugmenter(metaclass=ABCMeta):

    @abstractmethod
    def augment_dataframe(self, df: pd.DataFrame ) -> pd.DataFrame:
        "Augmetate dataframe data. Pandas dataframe"

    @abstractmethod
    def augment_column(self, col: pd.Series,) -> pd.Series:
        "Augmetate column data. Pandas column"

    @abstractmethod
    def _prepare_data_to_aug(self, col: pd.Series, freq=0.2) -> pd.Series:
        "Get part of data. Not augment all of it excep case freq=1.0"
