import pandas as pd
from typing import Tuple
from abc import ABCMeta, abstractmethod



class AbstractDataAugmenter(metaclass=ABCMeta):
    @abstractmethod
    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment dataframe data. Pandas dataframe"""

    @abstractmethod
    def augment_column(self, col: pd.Series,) -> pd.Series:
        """Augment column data. Pandas column"""

    @abstractmethod
    def _prepare_data_to_aug(
        self, col: pd.Series, freq=0.2
    ) -> Tuple[pd.Series, pd.Series]:
        """Get part of data. Not augment all of it except case freq=1.0"""
