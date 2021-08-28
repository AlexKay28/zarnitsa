import os
import sys
import pytest
import pandas as pd

sys.path.append("zarnitsa/")

from DataAugmenterExternally import DataAugmenterExternally


@pytest.fixture
def dataug_externally():
    return DataAugmenterExternally()


def test_augment_column(dataug_externally):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_externally.augment_column(pd_series)
        assert False, "ValueError wasnt throwed!"
    except ValueError as e:
        assert True
