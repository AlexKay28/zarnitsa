import os
import pytest


def test_general_import():
    """
    Augment column with normal distribution
    """
    import zarnitsa


def test_nlp_imports():
    from zarnitsa import nlp
    from zarnitsa.nlp import DataAugmenterNLP

def test_stats_imports():
    from zarnitsa import stats
    from zarnitsa.stats import DataAugmenterExternally
    from zarnitsa.stats import DataAugmenterInternally

def test_series_imports():
    from zarnitsa import series
    from zarnitsa.series import DataAugmenterTimeSeries
