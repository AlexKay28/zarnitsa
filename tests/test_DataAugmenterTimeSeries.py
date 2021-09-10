import os
import sys
import pytest
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

sys.path.append("zarnitsa/")

from series.DataAugmenterTimeSeries import DataAugmenterTimeSeries


N_TO_CHECK = 500


@pytest.fixture
def dataug_time_series():
    return DataAugmenterTimeSeries()


@pytest.fixture
def empty_pd_series():
    return pd.Series([], dtype="float64")


@pytest.fixture
def normal_pd_series():
    mu, sigma = 0, 500
    x = np.arange(1, 100, 0.1)
    y = x ** 2 + np.random.normal(mu, sigma, len(x))
    return pd.Series(data=y, index=x, dtype="float64")


@pytest.fixture
def pd_series():
    index = [i for i in range(100)]
    data = [i * 3 for i in range(100)]
    return pd.Series(data=data, index=index, dtype="float64")


def test_data_dropping_empty_data(dataug_time_series, empty_pd_series):
    """
    data_dropping
    """
    with pytest.raises(ValueError):
        dataug_time_series.data_dropping(empty_pd_series)


def test_data_dropping_freq_0_5(dataug_time_series, pd_series):
    """
    data_dropping
    """
    freq = 0.5
    new_pd_series = dataug_time_series.data_dropping(pd_series, freq=freq)
    assert pd_series.shape[0] * (1 - freq) == new_pd_series.shape[0]


def test_data_dropping_freq_0_0(dataug_time_series, pd_series):
    """
    data_dropping
    """
    freq = 0.0
    new_pd_series = dataug_time_series.data_dropping(pd_series, freq=freq)
    assert pd_series.shape[0] * (1 - freq) == new_pd_series.shape[0]


def test_data_dropping_freq_1_0(dataug_time_series, pd_series):
    """
    data_dropping
    """
    freq = 1.0
    new_pd_series = dataug_time_series.data_dropping(pd_series, freq=freq)
    assert pd_series.shape[0] * (1 - freq) == new_pd_series.shape[0]


def test_data_dropping_limit_20(dataug_time_series, pd_series):
    """
    data_dropping
    """
    limit = 20
    new_pd_series = dataug_time_series.data_dropping(pd_series, limit=limit)
    assert pd_series.shape[0] * (1 - 0.2) == new_pd_series.shape[0]


def test_data_interpolation_empty_data(dataug_time_series, empty_pd_series):
    """
    data_interpolation
    """
    with pytest.raises(ValueError):
        dataug_time_series.data_interpolation(empty_pd_series)


def test_data_interpolation_fill_na(dataug_time_series, pd_series):
    """
    data_interpolation
    """
    freq = 1.0
    new_pd_series = dataug_time_series.data_interpolation(
        pd_series, value_to_fill=0, method="fillna", freq=freq
    )
    assert new_pd_series[new_pd_series.isna()].shape[0] == 0


def test_data_interpolation_linear(dataug_time_series, pd_series):
    """
    data_interpolation
    """
    freq = 1.0
    new_pd_series = dataug_time_series.data_interpolation(
        pd_series, method="linear", freq=freq
    )
    assert new_pd_series[new_pd_series.isna()].shape[0] == 0


def test_data_interpolation_polyfit(dataug_time_series, pd_series):
    """
    data_interpolation
    """
    freq = 1.0
    new_pd_series = dataug_time_series.data_interpolation(
        pd_series, method="polyfit", freq=freq
    )
    assert new_pd_series[new_pd_series.isna()].shape[0] == 0


def test_data_extrapolation_empty_data(dataug_time_series, empty_pd_series):
    """
    data_extrapolation
    """
    with pytest.raises(ValueError):
        dataug_time_series.data_extrapolation(empty_pd_series)


def test_data_extrapolation_limit_20(dataug_time_series, pd_series):
    """
    data_extrapolation
    """
    limit = 20
    new_pd_series = dataug_time_series.data_extrapolation(
        pd_series, method="polyfit", limit=limit, order=3
    )
    assert new_pd_series.shape[0] == pd_series.shape[0] + limit


def test_data_denoiser_empty_data(dataug_time_series, empty_pd_series):
    """
    data_denoiser
    """
    with pytest.raises(ValueError):
        dataug_time_series.data_denoiser(empty_pd_series)


def test_data_denoiser_lfilter(dataug_time_series, normal_pd_series):
    """
    data_denoiser
    """
    n = 15
    numerator = [1.0 / n] * n
    denominator = 1
    new_pd_series = dataug_time_series.data_denoiser(
        normal_pd_series, method="lfilter", numerator=numerator, denominator=denominator
    )
    assert new_pd_series.shape[0] == normal_pd_series.shape[0]


def test_data_noiser_empty_data(dataug_time_series, empty_pd_series):
    """
    data_noiser
    """
    with pytest.raises(ValueError):
        dataug_time_series.data_noiser(empty_pd_series)


def test_data_noiser_normal(dataug_time_series, pd_series):
    """
    data_noiser
    """
    limit, mean, sig = 10, 0, 500
    new_pd_series = dataug_time_series.data_noiser(
        pd_series, method="normal", limit=limit, mean=mean, sig=sig
    )
    assert new_pd_series.shape[0] == pd_series.shape[0]
