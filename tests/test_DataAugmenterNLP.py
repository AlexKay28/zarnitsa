import os
import pytest
import pandas as pd

import sys

sys.path.append("zarnitsa/")

from nlp.DataAugmenterNLP import DataAugmenterNLP


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

    n_to_del = 1
    text_to_aug = "one two three four five"
    text_augmented = dataug_nlp.augment_del(text_to_aug, reps=n_to_del)
    assert len(text_augmented.split()) == len(text_to_aug.split()) - n_to_del

    n_to_del = 3
    text_to_aug = "one two three four five"
    text_augmented = dataug_nlp.augment_del(text_to_aug, reps=n_to_del)
    assert len(text_augmented.split()) == len(text_to_aug.split()) - n_to_del

    n_to_del = 100
    text_to_aug = "one two three four five"
    text_augmented = dataug_nlp.augment_del(text_to_aug, reps=n_to_del)
    assert len(text_augmented.split()) == 1


def test_augment_permut(dataug_nlp):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_to_aug = ""
    text_augmented = dataug_nlp.augment_permut(text_to_aug)
    assert len(text_augmented) == 0

    n_times_permut = 1
    text_to_aug = "one two three four five"
    text_augmented = dataug_nlp.augment_permut(text_to_aug, reps=n_times_permut)
    permutations = [
        1 if f != s else 0 for f, s in zip(text_to_aug.split(), text_augmented.split())
    ]
    assert sum(permutations) == 2


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

    n_reps = 1
    text_to_aug = "queen and king like ice cream"
    text_augmented = dataug_nlp.augment_word_emb(text_to_aug, reps=n_reps)
    words_changed = [
        1 if first != second else 0
        for first, second in zip(text_to_aug.split(), text_augmented.split())
    ]
    assert sum(words_changed) == 1
