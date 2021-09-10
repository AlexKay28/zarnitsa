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


def test_data_dropping(dataug_time_series):
    """
    data_dropping
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_time_series.data_dropping(pd_series)
        assert False, "ValueError was throwed!"
    except ValueError as e:
        assert True

    index = [i for i in range(100)]
    data = [i * 3 for i in range(100)]
    pd_series = pd.Series(data=data, index=index, dtype="float64")

    freq = 0.5
    new_pd_series = dataug_time_series.data_dropping(pd_series, freq=freq)
    assert pd_series.shape[0] * (1 - freq) == new_pd_series.shape[0]

    freq = 0.0
    new_pd_series = dataug_time_series.data_dropping(pd_series, freq=freq)
    assert pd_series.shape[0] * (1 - freq) == new_pd_series.shape[0]

    freq = 1.0
    new_pd_series = dataug_time_series.data_dropping(pd_series, freq=freq)
    assert pd_series.shape[0] * (1 - freq) == new_pd_series.shape[0]

    limit = 20
    new_pd_series = dataug_time_series.data_dropping(pd_series, limit=limit)
    assert pd_series.shape[0] * (1 - 0.2) == new_pd_series.shape[0]


def test_data_interpolation(dataug_time_series):
    """
    data_interpolation
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_time_series.data_interpolation(pd_series)
        assert False, "ValueError was throwed!"
    except ValueError as e:
        assert True

    index = [i for i in range(100)]
    data = [i * 3 if i % 10 == 0 else np.nan for i in range(100)]
    pd_series = pd.Series(data=data, index=index, dtype="float64")

    freq = 1.0
    new_pd_series = dataug_time_series.data_interpolation(
        pd_series, value_to_fill=0, method="fillna", freq=freq
    )
    assert new_pd_series[new_pd_series.isna()].shape[0] == 0

    freq = 1.0
    new_pd_series = dataug_time_series.data_interpolation(
        pd_series, method="linear", freq=freq
    )
    assert new_pd_series[new_pd_series.isna()].shape[0] == 0

    freq = 1.0
    new_pd_series = dataug_time_series.data_interpolation(
        pd_series, method="polyfit", freq=freq
    )
    assert new_pd_series[new_pd_series.isna()].shape[0] == 0


def test_data_extrapolation(dataug_time_series):
    """
    data_extrapolation
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_time_series.data_extrapolation(pd_series)
        assert False, "ValueError was throwed!"
    except ValueError as e:
        assert True

    index = [i for i in range(80)]
    data = [i * 3 for i in range(80)]
    pd_series = pd.Series(data=data, index=index, dtype="float64")

    limit = 20
    new_pd_series = dataug_time_series.data_extrapolation(
        pd_series, method="polyfit", limit=limit, order=3
    )
    assert new_pd_series.shape[0] == pd_series.shape[0] + limit


def test_data_denoiser(dataug_time_series):
    """
    data_denoiser
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_time_series.data_denoiser(pd_series)
        assert False, "ValueError was throwed!"
    except ValueError as e:
        assert True

    mu, sigma = 0, 500
    x = np.arange(1, 100, 0.1)
    y = x ** 2 + np.random.normal(mu, sigma, len(x))
    pd_series = pd.Series(data=y, index=x, dtype="float64")

    n = 15
    numerator = [1.0 / n] * n
    denominator = 1
    new_pd_series = dataug_time_series.data_denoiser(
        pd_series, method="lfilter", numerator=numerator, denominator=denominator
    )
    assert new_pd_series.shape[0] == pd_series.shape[0]


def test_data_noiser(dataug_time_series):
    """
    data_noiser
    """
    pd_series = pd.Series([], dtype="float64")
    try:
        dataug_time_series.data_noiser(pd_series)
        assert False, "ValueError was throwed!"
    except ValueError as e:
        assert True

    index = [i for i in range(80)]
    data = [i * 3 for i in range(80)]
    pd_series = pd.Series(data=data, index=index, dtype="float64")

    limit, mean, sig = 10, 0, 500
    new_pd_series = dataug_time_series.data_noiser(
        pd_series, method="normal", limit=limit, mean=mean, sig=sig
    )
    assert new_pd_series.shape[0] == pd_series.shape[0]
