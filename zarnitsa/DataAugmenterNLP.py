import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
from .DataAugmenter import AbstractDataAugmenter


class DataAugmenterNLP(AbstractDataAugmenter):
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
            "wordnet":   self.augment_column_wordnet,
            "ppdb":      self.augment_column_ppdb,
            "embedding": self.augment_column_emb,
            "deleting":  self.augment_column_del,
            "permutations": self.augment_column_permut,
        }
        not_to_aug, to_aug = self._prepare_data_to_aug(df, freq=freq)
        for col in df.columns:
            to_aug[col] = augment_column_method[aug_type](to_aug[col], freq=1.0)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column(self, col: pd.Series, aug_type='normal', freq=0.2, return_only_aug=False) -> pd.Series:
        "Augmetate Serial data. Pandas column"
        augment_column_method = {
            "wordnet":   self.augment_column_wordnet,
            "ppdb":      self.augment_column_ppdb,
            "embedding": self.augment_column_emb,
            "deleting":  self.augment_column_del,
            "permutations": self.augment_column_permut,
        }
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)
        to_aug = augment_column_method[aug_type](to_aug, freq=1.0)
        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column_wordnet(self, col: pd.Series, freq=0.2, return_only_aug=False) -> pd.Series:
        "Augmetate column data using wordnet synset. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        # to aug data
        self.aug_wdnt = naw.SynonymAug(aug_src='wordnet')
        to_aug = to_aug.progress_apply(
            lambda text: self.aug_wdnt.augment(text)
        )

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column_ppdb(self, col: pd.Series, freq=0.2, return_only_aug=False) -> pd.Series:
        "Augmetate column data using ppdb synset. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        # to aug data
        # TODO to_aug

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column_emb(self, col: pd.Series, freq=0.2, return_only_aug=False) -> pd.Series:
        "Augmetate column data using embeddings. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        # to aug data
        # TODO to_aug

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column_del(self, col: pd.Series, freq=0.2, return_only_aug=False) -> pd.Series:
        "Augmetate column data using deleting. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        # to aug data
        # TODO to_aug

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])


    def augment_column_permut(self, col: pd.Series, freq=0.2, return_only_aug=False) -> pd.Series:
        "Augmetate column data using permutations. Pandas column"
        not_to_aug, to_aug = self._prepare_data_to_aug(col, freq=freq)

        # to aug data
        # TODO to_aug

        return to_aug if return_only_aug else pd.concat([not_to_aug, to_aug])
