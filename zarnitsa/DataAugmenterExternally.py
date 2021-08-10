import numpy as np
import pandas as pd

from .DataAugmenter import AbstractDataAugmenter


class DataAugmenterExternally(AbstractDataAugmenter):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def augment_dataframe(
        self, df: pd.DataFrame, aug_params="normal", **kwargs
    ) -> pd.DataFrame:
        """Augment dataframe data. Pandas dataframe"""
        if isinstance(aug_params, str):
            print("Single type of augmentation")
            for col_name in df.columns:
                df[col_name] = self.augment_column(
                    df[col_name], aug_params=aug_params, **kwargs
                )
        elif isinstance(aug_params, dict):
            print("Multiple types of augmentation")
            for col_name, params in aug_params.items():
                df[col_name] = self.augment_column(df[col_name], **params)
        else:
            raise KeyError("Bad type of aug_params variable")
        return df

    def augment_column(
        self, col: pd.Series, aug_type="normal", **kwargs
    ) -> pd.Series:
        """Augment Serial data. Pandas column"""
        if all(col.isna()):
            col = pd.Series(
                self.augment_distrib_random(
                    aug_type=aug_type, size=col.shape[0]
                )
            )
        else:
            col = col.apply(
                lambda v: v
                if not np.isnan(v)
                else self.augment_distrib_random(aug_type, **kwargs)
            )
        return col

    def _prepare_data_to_aug(self, col: pd.Series, freq=0.2) -> pd.Series:
        raise NotImplemented(
            "External augmentation doesn't utilize data split"
        )

    def augment_distrib_random(self, aug_type="normal", size=None, **kwargs):
        """Return float or array depends on needed size. If size is 1 - returns array of size 1"""
        np_distributions = {
            "beta": dict(a=kwargs["a"], b=kwargs["b"], size=size),
            "binomial": dict(n=kwargs["n"], p=kwargs["p"], size=size),
            "chisquare": dict(df=kwargs["df"], size=size),
            "dirichlet": dict(alpha=kwargs["alpha"], size=size),
            "exponential": dict(scale=kwargs["scale"], size=size),
            "f": dict(dfnum=kwargs["dfnum"], dfden=kwargs["dfden"], size=size),
            "gamma": dict(
                shape=kwargs["shape"], scale=kwargs["scale"], size=size
            ),
            "geometric": dict(p=kwargs["p"], size=size),
            "gumbel": dict(
                loc=kwargs["loc"], scale=kwargs["scale"], size=size
            ),
            "hypergeometric": dict(
                ngood=kwargs["ngood"],
                nbad=kwargs["nbad"],
                nsample=kwargs["nsample"],
                size=size,
            ),
            "laplace": dict(
                loc=kwargs["loc"], scale=kwargs["scale"], size=size
            ),
            "logistic": dict(
                loc=kwargs["loc"], scale=kwargs["scale"], size=size
            ),
            "lognormal": dict(
                mean=kwargs["mean"], sigma=kwargs["sigma"], size=size
            ),
            "logseries": dict(p=kwargs["p"], size=size),
            "multinomial": dict(
                n=kwargs["n"], pvals=kwargs["pvals"], size=size
            ),
            "multivariate_norma": dict(
                mean=kwargs["mean"], cov=kwargs["cov"], size=size
            ),
            "negative_binomial": dict(n=kwargs["n"], p=kwargs["p"], size=size),
            "noncentral_chisquare": dict(
                df=kwargs["df"], nonc=kwargs["nonc"], size=size
            ),
            "noncentral_f": dict(
                dfnum=kwargs["dfnum"],
                dfden=kwargs["dfden"],
                nonc=kwargs["nonc"],
                size=size,
            ),
            "normal": dict(
                loc=kwargs["loc"], scale=kwargs["scale"], size=size
            ),
            "pareto": dict(a=kwargs["a"], size=size),
            "poisson": dict(lam=kwargs["lam"], size=size),
            "power": dict(a=kwargs["a"], size=size),
            "rayleigh": dict(scale=kwargs["scale"], size=size),
            "standard_cauchy": dict(size=size),
            "standard_exponential": dict(size=size),
            "standard_gamma": dict(shape=kwargs["shape"], size=size),
            "standard_normal": dict(size=size),
            "standard_t": dict(df=kwargs["df"], size=size),
            "triangular": dict(
                left=kwargs["left"],
                mode=kwargs["mode"],
                right=kwargs["right"],
                size=size,
            ),
            "uniform": dict(low=kwargs["low"], high=kwargs["high"], size=size),
            "vonmises": dict(
                mu=kwargs["mu"], kappa=kwargs["kappa"], size=size
            ),
            "wald": dict(
                mean=kwargs["mean"], scale=kwargs["scale"], size=size
            ),
            "weibull": dict(a=kwargs["a"], size=size),
            "zipf": dict(a=kwargs["a"], size=size),
        }
        np_func = getattr(np.random, aug_type, None)
        if np_func:
            np_func(**np_distributions[aug_type])
        else:
            raise KeyError(f"Unknown aug type: <{aug_type}>")
