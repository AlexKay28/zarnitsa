import os

from typing import Tuple

import wget
import spacy
import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw

from sklearn.model_selection import train_test_split

from .DataAugmenter import AbstractDataAugmenter


class DataAugmenterNLP(AbstractDataAugmenter):

    __class_local_path = os.path.dirname(os.path.realpath(__file__))
    __aug_wdnt = None
    __aug_ppdb = None

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def _prepare_data_to_aug(self, data, freq=0.2) -> Tuple[pd.Series, pd.Series]:
        """Get part of data. Not augment all of it excep case freq=1.0"""
        data = (
            pd.Series(data)
            if type(data) is not pd.Series and type(data) is not pd.DataFrame
            else data
        )
        if freq < 1:
            not_to_aug, to_aug = train_test_split(data, test_size=freq)
            return not_to_aug, to_aug
        elif freq == 1:
            return data.sample(0), data
        elif freq == 0:
            return data, data.sample(0)

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        freq=0.2,
        return_only_aug=False,
        aug_type="deleting",
        reps=1,
        min_words=1,
        window_size=3,
        columns=None,
    ) -> pd.DataFrame:
        """Augment dataframe data. Pandas dataframe"""
        not_to_aug, to_aug = self._prepare_data_to_aug(df, freq=freq)
        columns = columns if columns else df.columns
        for col in columns:
            to_aug[col] = self.augment_column(
                to_aug[col],
                freq=1.0,
                return_only_aug=return_only_aug,
                aug_type=aug_type,
                reps=reps,
                min_words=min_words,
                window_size=window_size,
            )
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column(
        self,
        col: pd.Series,
        freq=0.2,
        return_only_aug=False,
        aug_type="deleting",
        embeddings_name="en_core_web_md",
        nlp=None,
        reps=1,
        topn=5,
        min_words=1,
        window_size=3,
    ) -> pd.Series:
        """Augment Serial data. Pandas column"""
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        if aug_type == "wordnet":
            to_aug = to_aug.apply(lambda text: self.augment_wordnet(text))
        elif aug_type == "ppdb":
            to_aug = to_aug.apply(lambda text: self.augment_ppdb(text))
        elif aug_type == "embedding":
            to_aug = to_aug.apply(
                lambda text: self.augment_word_emb(
                    text,
                    vocab_name=embeddings_name,
                    nlp=nlp,
                    reps=reps,
                    topn=topn,
                )
            )
        elif aug_type == "deleting":
            to_aug = to_aug.apply(
                lambda text: self.augment_del(text, reps=reps, min_words=min_words)
            )
        elif aug_type == "permutations":
            to_aug = to_aug.apply(
                lambda text: self.augment_permut(
                    text, reps=reps, window_size=window_size
                )
            )
        else:
            raise KeyError(f"Unknown type of NLP augmentation! [{aug_type}]")
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def _check_synset(self, name):
        """Check synset installed"""
        if name == "ppdb":
            my_file = os.path.join(
                self.__class_local_path, "internal_data", "ppdb-2.0-tldr"
            )  # todo: defined but not used
            wget.download("http://nlpgrid.seas.upenn.edu/PPDB/eng/ppdb-2.0-tldr.gz")
        else:
            raise KeyError(f"Synset {name} is unknown! Load manually or fix the name")

    def augment_wordnet(self, text: str) -> str:
        """Augment column data using wordnet synset."""
        if not self.__aug_wdnt:
            print("Load WordNet synset")
            self.__aug_wdnt = naw.SynonymAug(aug_src="wordnet")
        text = self.__aug_wdnt.augment(text)
        return text

    def augment_ppdb(self, text: str) -> str:
        """Augment column data using ppdb synset."""
        if not self.__aug_ppdb:
            print("Load PPDB synset")
            self._check_synset("ppdb")
            self.__aug_ppdb = naw.SynonymAug(
                aug_src="ppdb",
                model_path=self.__class_local_path + "/internal_data/ppdb-2.0-tldr",
            )
        text = self.__aug_ppdb.augment(text)
        return text

    @staticmethod
    def augment_word_emb(
        text: str,
        vocab_name=None,
        nlp=None,
        reps=1,
        topn=5,
    ) -> str:
        """Augment column data using embeddings."""
        if not nlp:
            if not vocab_name:
                print("Use spacy en_core_web_sm vocab!")
                nlp = spacy.load("en_core_web_md")
            else:
                nlp = spacy.load(vocab_name)

        def get_most_similar(word, topn=topn):
            """Get topn words for defined word"""
            word = nlp.vocab[str(word)]
            queries = [
                w
                for w in word.vocab
                if w.is_lower == word.is_lower
                and np.count_nonzero(w.vector)
            ]
            by_similarity = sorted(
                queries, key=lambda w: word.similarity(w), reverse=True
            )
            return [
                w.lower_.capitalize()
                if not word.lower_.islower() and not word.lower_.isupper()
                else w.lower_
                if word.islower()
                else word.lower_.upper()
                for w in by_similarity[: topn + 1]
                if w.lower_ != word.lower_
            ]

        text = text.split()
        text_len = len(text)
        for _ in range(reps):
            random_word_idx = np.random.choice(text_len)
            random_word = text[random_word_idx]
            random_word_similar = get_most_similar(random_word)
            text[random_word_idx] = np.random.choice(random_word_similar)
        return " ".join(text)

    @staticmethod
    def augment_del(text: str, reps=1, min_words=1) -> str:
        """Augment column data using deleting."""
        text = text.split()
        for _ in range(reps):
            text_len = len(text)
            if text_len <= min_words:
                break
            random_word_idx = np.random.randint(min_words - 1, text_len - 1)
            del text[random_word_idx]
        return " ".join(text)

    @staticmethod
    def augment_permut(text: str, reps=1, window_size=3) -> str:
        """Augment column data using permutations."""
        text = text.split()
        text_len = len(text)
        for _ in range(reps):
            window_start = np.random.choice(max(0, text_len - window_size))
            first, second = window_start + np.random.choice(
                window_size, size=2, replace=False
            )
            text[first], text[second] = text[second], text[first]
        return " ".join(text)
