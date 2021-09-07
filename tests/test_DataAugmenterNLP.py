import os
import pytest
import pandas as pd

import sys

sys.path.append("zarnitsa/")

from DataAugmenterNLP import DataAugmenterNLP


@pytest.fixture
def dataug_nlp():
    return DataAugmenterNLP()


@pytest.fixture
def emty_text():
    return ""


@pytest.fixture
def filled_text():
    return "one two three four five"


def test_augment_del_1(dataug_nlp, emty_text):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_augmented = dataug_nlp.augment_del(emty_text)
    assert len(text_augmented) == 0


def test_augment_del_2(dataug_nlp, filled_text):
    n_to_del = 1
    text_augmented = dataug_nlp.augment_del(filled_text, reps=n_to_del)
    assert len(text_augmented.split()) == len(filled_text.split()) - n_to_del


def test_augment_del_3(dataug_nlp, filled_text):
    n_to_del = 3
    text_augmented = dataug_nlp.augment_del(filled_text, reps=n_to_del)
    assert len(text_augmented.split()) == len(filled_text.split()) - n_to_del


def test_augment_del_4(dataug_nlp, filled_text):
    n_to_del = 100
    text_augmented = dataug_nlp.augment_del(filled_text, reps=n_to_del)
    assert len(text_augmented.split()) == 1


def test_augment_permut_1(dataug_nlp, emty_text):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_augmented = dataug_nlp.augment_permut(emty_text)
    assert len(text_augmented) == 0


def test_augment_permut_2(dataug_nlp, filled_text):
    n_times_permut = 1
    text_augmented = dataug_nlp.augment_permut(filled_text, reps=n_times_permut)
    permutations = [
        1 if f != s else 0 for f, s in zip(filled_text.split(), text_augmented.split())
    ]
    assert sum(permutations) == 2


def test_augment_wordnet(dataug_nlp, emty_text):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_augmented = dataug_nlp.augment_wordnet(emty_text)
    assert len(text_augmented) == 0


def test_augment_ppdb(dataug_nlp, emty_text):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_augmented = dataug_nlp.augment_ppdb(emty_text)
    assert len(text_augmented) == 0


def test_augment_emb_1(dataug_nlp, emty_text):
    """
    Test permutation approach for augmentation for pandas series
    (or any other iterable) variable
    """
    text_augmented = dataug_nlp.augment_word_emb(emty_text)


def test_augment_emb_2(dataug_nlp, filled_text):
    n_reps = 1
    text_augmented = dataug_nlp.augment_word_emb(filled_text, reps=n_reps)
    words_changed = [
        1 if first != second else 0
        for first, second in zip(filled_text.split(), text_augmented.split())
    ]
    assert sum(words_changed) == 1
