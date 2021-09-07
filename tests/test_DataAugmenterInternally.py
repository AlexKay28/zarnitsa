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
SIG = 3
N_SIG = 0.33


@pytest.fixture
def dai():
    return DataAugmenterInternally()


@pytest.fixture
def emty_data():
    return pd.Series([], dtype="float64")


@pytest.fixture
def normal_data():
    return pd.Series(np.random.normal(0, SIG * 3, size=N_TO_CHECK), dtype="float64")


@pytest.fixture
def uniform_data():
    return pd.Series(np.random.uniform(0, SIG * 3, size=N_TO_CHECK), dtype="float64")


def test_augment_column(dai, emty_data):
    """
    Augment column
    """
    with pytest.raises(ValueError):
        dai.augment_column_norm(emty_data)


def test_augment_column_permut_er(dai, emty_data):
    """
    Augment column with normal distribution
    """
    with pytest.raises(ValueError):
        dai.augment_column_norm(emty_data)


def test_augment_column_permut_1(dai, normal_data):
    normal_data_aug = dai.augment_column_permut(normal_data, freq=0)
    assert ks_2samp(normal_data, normal_data_aug).pvalue > 0.95, "KS criteria"


def test_augment_column_permut_2(dai, normal_data):
    normal_data_aug = dai.augment_column_permut(
        normal_data,
        freq=1.0,
        n_to_aug=N_TO_CHECK,
        return_only_aug=True,
    )
    assert ks_2samp(normal_data, normal_data_aug).pvalue > 0.95, "KS criteria"


def test_augment_column_norm_er(dai, emty_data):
    """
    Test case: Augment column with normal distribution
    """
    pd_series = pd.Series([], dtype="float64")
    with pytest.raises(ValueError):
        dai.augment_column_norm(emty_data)


def test_augment_column_norm_1(dai, normal_data):
    normal_data_aug = dai.augment_column_norm(normal_data, freq=0)
    assert ks_2samp(normal_data, normal_data_aug).pvalue > 0.95, "KS criteria"


def test_augment_column_norm_2(dai, normal_data):
    normal_data_aug = dai.augment_column_norm(
        normal_data,
        freq=1,
        n_sigm=N_SIG,
        n_to_aug=N_TO_CHECK,
        return_only_aug=True,
    )
    assert ks_2samp(normal_data, normal_data_aug).pvalue > 0.4, "KS criteria"


def test_augment_column_norm_3(dai, uniform_data):
    normal_data_aug = dai.augment_column_norm(uniform_data, freq=1.0)
    assert ks_2samp(uniform_data, normal_data_aug).pvalue < 1e-8, "KS criteria"


def test_augment_column_uniform_er(dai, emty_data):
    """
    Test case: Augment column with uniform distribution
    """
    pd_series = pd.Series([], dtype="float64")
    with pytest.raises(ValueError):
        dai.augment_column_norm(emty_data)


def test_augment_column_uniform_1(dai, uniform_data):
    uniform_data_aug = dai.augment_column_uniform(uniform_data, freq=0)
    assert ks_2samp(uniform_data, uniform_data_aug).pvalue > 0.95, "KS criteria"


def test_augment_column_uniform_2(dai, uniform_data):
    uniform_data_aug = dai.augment_column_uniform(
        uniform_data,
        freq=1.0,
        n_sigm=N_SIG,
        n_to_aug=N_TO_CHECK,
        return_only_aug=True,
    )
    assert ks_2samp(uniform_data, uniform_data_aug).pvalue > 0.4, "KS criteria"


def test_augment_column_uniform_3(dai, normal_data):
    uniform_data_aug = dai.augment_column_uniform(normal_data, freq=1.0)
    assert ks_2samp(normal_data, uniform_data_aug).pvalue < 1e-8, "KS criteria"
