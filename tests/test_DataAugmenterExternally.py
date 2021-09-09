import os
import sys
import pytest
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

sys.path.append("zarnitsa/")

from DataAugmenterExternally import DataAugmenterExternally


N_TO_CHECK = 500
SIG = 0.5


@pytest.fixture
def dae():
    return DataAugmenterExternally()


@pytest.fixture
def normal_data():
    return pd.Series(np.random.normal(0, SIG * 3, size=N_TO_CHECK), dtype="float64")


def test_augment_column_permute(dae, normal_data):
    """
    Augment column with normal distribution
    """
    normal_data_aug = dae.augment_distrib_random(
        aug_type="normal", size=N_TO_CHECK, loc=0, scale=SIG * 3
    )
    print(normal_data_aug)
    assert ks_2samp(normal_data, normal_data_aug).pvalue > 0.5, "KS criteria"
