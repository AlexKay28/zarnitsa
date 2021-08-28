import os
import pytest
import pandas as pd

import sys

sys.path.append("zarnitsa/")

from DataAugmenterNLP import DataAugmenterNLP


@pytest.fixture
def dataug_nlp():
    return DataAugmenterNLP()


def test_augment_del(dataug_nlp):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_to_aug = ""
    text_augmented = dataug_nlp.augment_del(text_to_aug)
    assert len(text_augmented) == 0


def test_augment_permut(dataug_nlp):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_to_aug = ""
    text_augmented = dataug_nlp.augment_permut(text_to_aug)
    assert len(text_augmented) == 0


def test_augment_wordnet(dataug_nlp):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_to_aug = ""
    text_augmented = dataug_nlp.augment_wordnet(text_to_aug)
    assert len(text_augmented) == 0


def test_augment_ppdb(dataug_nlp):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_to_aug = ""
    text_augmented = dataug_nlp.augment_ppdb(text_to_aug)
    assert len(text_augmented) == 0


def test_augment_emb(dataug_nlp):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_to_aug = ""
    text_augmented = dataug_nlp.augment_word_emb(text_to_aug)
    assert len(text_augmented) == 0
