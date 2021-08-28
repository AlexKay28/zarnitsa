import os
import pytest
import pandas as pd

import sys

sys.path.append("zarnitsa/")

from DataAugmenterInternally import DataAugmenterInternally


@pytest.fixture
def dataug_internally():
    return DataAugmenterInternally()


def test_augment_column_permute(dataug_internally):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_internally.augment_column_permut(pd_series)
        assert False, "ValueError wasnt throwed!"
    except ValueError as e:
        assert True

    pd_series = pd.Series([0, 1, 2, 3, 4, 5])
    # test non changed table outside
    pd_series_test = dataug_internally.augment_column_permut(
        pd_series, n_to_aug=0, return_only_aug=False
    )
    assert (pd_series == pd_series_test).all(), "Assert return the same Series"

    # test table with no returned values
    pd_series_test = dataug_internally.augment_column_permut(
        pd_series, n_to_aug=0, return_only_aug=True
    )
    assert (
        pd.Series([], dtype="float64") == pd_series_test
    ).all(), "Assert return empty table"


def test_augment_column_norm(dataug_internally):
    """
    Tests for augmentation using normal distribution
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_internally.augment_column_norm(pd_series)
        assert False, "ValueError wasnt throwed!"
    except ValueError as e:
        assert True


def test_augment_column_uniform(dataug_internally):
    """
    Tests for augmentation using normal distribution
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_internally.augment_column_uniform(pd_series)
        assert False, "ValueError wasnt throwed!"
    except ValueError as e:
        assert True
