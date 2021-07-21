import os
import wget
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
from .DataAugmenter import AbstractDataAugmenter


class DataAugmenterNLP(AbstractDataAugmenter):

    __class_local_path = os.path.dirname(os.path.realpath(__file__))

    self.aug_wdnt = naw.SynonymAug(aug_src='wordnet')

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def _prepare_data_to_aug(self, data, freq=0.2) -> pd.Series:
        "Get part of data. Not augment all of it excep case freq=1.0"
        data = pd.Series(data) if type(data) is not pd.Series and type(data) is not pd.DataFrame else data
        if freq < 1:
            not_to_aug, to_aug = train_test_split(data, test_size=freq)
            return not_to_aug, to_aug
        elif freq == 1:
            return data.sample(0), data
        elif freq == 0:
            return data, data.sample(0)

    def augment_dataframe(self, df: pd.DataFrame, aug_type='normal', freq=0.2, return_only_aug=False) -> pd.DataFrame:
        "Augmetate dataframe data. Pandas dataframe"
        augment_column_method = {
            "wordnet":      self.augment_column_wordnet,
            "ppdb":         self.augment_column_ppdb,
            "embedding":    self.augment_column_word_emb,
            "deleting":     self.augment_column_del,
            "permutations": self.augment_column_permut,
        }
        not_to_aug, to_aug = self._prepare_data_to_aug(df, freq=freq)
        for col in df.columns:
            to_aug[col] = augment_column_method[aug_type](to_aug[col], freq=1.0)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column(self, col: pd.Series, aug_type='normal', freq=0.2, return_only_aug=False) -> pd.Series:
        "Augmetate Serial data. Pandas column"
        augment_column_method = {
            "wordnet":      self.augment_column_wordnet,
            "ppdb":         self.augment_column_ppdb,
            "embedding":    self.augment_column_word_emb,
            "deleting":     self.augment_column_del,
            "permutations": self.augment_column_permut,
        }
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        to_aug = augment_column_method[aug_type](to_aug, freq=1.0)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def _check_synset(self, name):
        "Check synset installed"
        if name == 'ppdb':
            my_file = os.path.join(self.__class_local_path, 'internal_data', 'ppdb-2.0-tldr')
            wget.download("http://nlpgrid.seas.upenn.edu/PPDB/eng/ppdb-2.0-tldr.gz")
        elif:
            raise KeyError(f"Synset {name} is unknown! load manually or fix the name")

    def augment_column_wordnet(self, col: pd.Series, freq=0.2, return_only_aug=False) -> pd.Series:
        "Augmetate column data using wordnet synset. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        if not self.aug_wdnt:
            self.aug_wdnt = naw.SynonymAug(aug_src='wordnet')

        to_aug = to_aug.progress_apply(
            lambda text: self.aug_wdnt.augment(text)
        )

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])

    def augment_column_ppdb(self, col: pd.Series, freq=0.2, return_only_aug=False) -> pd.Series:
        "Augmetate column data using ppdb synset. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        if not self.aug_wdnt:
            self._check_synset('ppdb')
            self.aug_ppdb = naw.SynonymAug(
                aug_src='ppdb', model_path=self.__class_local_path + '/internal_data/ppdb-2.0-tldr'
            )

        to_aug = to_aug.progress_apply(
            lambda text: self.aug_ppdb.augment(text)
        )

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column_word_emb(self, col: pd.Series, freq=0.2, return_only_aug=False, reps=1,, words_and_vectors=None) -> pd.Series:
        "Augmetate column data using embeddings. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        if type(words_and_vectors) != dict:
            raise "Var words_and_vectors must be dict!"
        elif len(words_and_vectors.keys()) == 0:
            raise "Var words_and_vectors is empty!"
        elif words_and_vectors is None:
            raise "Define words_and_vectors variable!"

        def replace_word_using_embeddings(text):
            text = text.split()
            text_len = len(text)
            for _ in range(reps):
                random_word_idx = np.random.choice(text_len)
                random_word     = text[random_word_idx]
                random_word_vec = words_and_vectors[random_word]
                del words_and_vectors[random_word]
                df_of_vecs = pd.DataFrame({
                    "word" : words_and_vectors.keys(),
                    "vec"  : words_and_vectors.values()
                })
                df_of_vecs['vec'].apply(lambda vec: sum(vec - random_word_vec), inplace=True)
                new_word = df_of_vecs.sort_values('vec', inplace=True).loc[0, 'word']
                text[random_word_idx] = choosed_word
                words_and_vectors[random_word] = random_word_vec

        to_aug = to_aug.apply(
            lambda text: replace_word_using_embeddings(text)
        )

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column_del(self, col: pd.Series, freq=0.2, return_only_aug=False, reps=1, min_words=5) -> pd.Series:
        "Augmetate column data using deleting. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        def delete_random_word(text):
            text = text.split()
            text_len = len(text)
            for _ in range(reps):
                if text_len < min_words:
                    break
                random_word_idx = np.random.randint(min_words, text_len)
                del text[random_word_idx]
            return ' '.join(text)

        to_aug = to_aug.apply(
            lambda text: delete_random_word(text)
        )

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column_permut(self, col: pd.Series, freq=0.2, return_only_aug=False, reps=1) -> pd.Series:
        "Augmetate column data using permutations. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        def permute_text(text):
            text = text.split()
            text_len = len(text)
            for _ in range(reps):
                window_start = np.random.choice(max(0, text_len-window_size))
                window = text[window_start:window_start + window_size]
                first, second = np.random.choice(window_size, size=2, replace=False)
                text[first], text[second] = text[second], text[first]
            return ' '.join(text)

        to_aug = to_aug.apply(
            lambda text: permute_text(text, reps)
        )

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])
