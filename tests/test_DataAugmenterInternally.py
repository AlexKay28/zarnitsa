import os
import sys
import pytest
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

sys.path.append("zarnitsa/")

from DataAugmenterInternally import DataAugmenterInternally

"""
Under the null hypothesis the two distributions are identical.
If the K-S statistic is small or the p-value is high
(greater than the significance level, say 5%),
then we cannot reject the hypothesis that the distributions of
the two samples are the same.
Conversely, we can reject the null hypothesis if the p-value is low.
"""

N_TO_CHECK = 500


@pytest.fixture
def dataug_internally():
    return DataAugmenterInternally()


def test_augment_column(dataug_internally):
    """
    Augment column
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_internally.augment_column_norm(pd_series)
        assert False, "ValueError was throwed!"
    except ValueError as e:
        assert True


def test_augment_column_permut(dataug_internally):
    """
    Augment column with normal distribution
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_internally.augment_column_norm(pd_series)
        assert False, "ValueError was throwed!"
    except ValueError as e:
        assert True

    sig = 1
    pd_series = pd.Series(
        np.random.normal(0, sig * 3, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_aug = dataug_internally.augment_column_permut(pd_series, freq=0)
    assert ks_2samp(pd_series, pd_series_aug).pvalue > 0.95, "KS criteria"

    sig = 1
    pd_series = pd.Series(
        np.random.normal(0, sig * 3, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_aug = dataug_internally.augment_column_permut(
        pd_series,
        freq=1.0,
        n_to_aug=pd_series.shape[0],
        return_only_aug=True,
    )
    assert ks_2samp(pd_series, pd_series_aug).pvalue > 0.95, "KS criteria"


def test_augment_column_norm(dataug_internally):
    """
    Test case: Augment column with normal distribution
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_internally.augment_column_norm(pd_series)
        assert False, "ValueError was throwed!"
    except ValueError as e:
        assert True

    sig = 1
    pd_series = pd.Series(
        np.random.normal(0, sig * 3, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_aug = dataug_internally.augment_column_norm(pd_series, freq=0)
    assert ks_2samp(pd_series, pd_series_aug).pvalue > 0.95, "KS criteria"

    sig = 1
    n_sig = 0.33
    pd_series = pd.Series(
        np.random.normal(0, sig * 3, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_aug = dataug_internally.augment_column_norm(
        pd_series,
        freq=1,
        n_sigm=n_sig,
        n_to_aug=pd_series.shape[0],
        return_only_aug=True,
    )
    assert ks_2samp(pd_series, pd_series_aug).pvalue > 0.4, "KS criteria"

    sig = 1
    pd_series = pd.Series(
        np.random.normal(0, sig * 3, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_bad = pd.Series(
        np.random.uniform(0, sig * 3, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_aug = dataug_internally.augment_column_uniform(pd_series, freq=1.0)
    assert ks_2samp(pd_series_bad, pd_series_aug).pvalue < 1e-15, "KS criteria"


def test_augment_column_uniform(dataug_internally):
    """
    Test case: Augment column with uniform distribution
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_internally.augment_column_norm(pd_series)
        assert False, "ValueError was throwed!"
    except ValueError as e:
        assert True

    sig = 1
    pd_series = pd.Series(
        np.random.uniform(0, sig * 3, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_aug = dataug_internally.augment_column_uniform(pd_series, freq=0)
    assert ks_2samp(pd_series, pd_series_aug).pvalue > 0.95, "KS criteria"

    sig = 1
    n_sig = 0.33
    pd_series = pd.Series(
        np.random.uniform(0, sig * 3, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_aug = dataug_internally.augment_column_uniform(
        pd_series,
        freq=1.0,
        n_sigm=n_sig,
        n_to_aug=pd_series.shape[0],
        return_only_aug=True,
    )
    assert ks_2samp(pd_series, pd_series_aug).pvalue > 0.4, "KS criteria"

    sig = 1
    pd_series = pd.Series(
        np.random.uniform(0, sig * 3, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_bad = pd.Series(
        np.random.normal(0, 3 * sig, size=N_TO_CHECK), dtype="float64"
    )
    pd_series_aug = dataug_internally.augment_column_uniform(pd_series, freq=1.0)
    assert ks_2samp(pd_series_bad, pd_series_aug).pvalue < 1e-15, "KS criteria"
