import numpy as np
import pandas as pd

from .DataAugmenter import AbstractDataAugmenter

class DataAugmenterExternally(AbstractDataAugmenter):

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def augment_dataframe(self, df: pd.DataFrame, aug_params='normal', **kwargs) -> pd.DataFrame:
        "Augmetate dataframe data. Pandas dataframe"
        if type(aug_params) == str:
            print('Single type of augmentation')
            for col_name in df.columns:
                df[col_name] = self.augment_column(df[col_name], aug_params=aug_params, **kwargs)
        elif type(aug_params) == dict:
            print("Multiple types of augmentation")
            for col_name, params in aug_params.items():
                df[col_name] = self.augment_column(df[col_name], **params)
        else:
            KeyError("Bad type of aug_params variable")
        return df

    def augment_column(self, col: pd.Series, aug_type='normal', **kwargs) -> pd.Series:
        "Augmetate Serial data. Pandas column"
        if all(col.isna()):
            col = pd.Series(self.augment_distrib_random(aug_type=aug_type, size=col.shape[0]))
        else:
            col = col.apply(lambda v: v if not np.isnan(v) else self.augment_distrib_random(aug_type, **kwargs))
        return col

    def _prepare_data_to_aug(self, ) -> pd.Series:
        print("External augmentation doesnt utilize data split")
        pass

    def augment_distrib_random(self, aug_type='normal', size=None, **kwargs):
        "Return float or array depends on needed size. If size is 1 - returns array of size 1"
        if aug_type == "beta":
            return np.random.beta(a=kwargs['a'], b=kwargs['b'], size=size)
        elif aug_type == "binomial":
            return np.random.binomial(n=kwargs['n'], p=kwargs['p'], size=size)
        elif aug_type == "chisquare":
            return np.random.chisquare(df=kwargs['df'], size=size)
        elif aug_type == "dirichlet":
            return np.random.dirichlet(alpha=kwargs['alpha'], size=size)
        elif aug_type == "exponential":
            return np.random.exponential(scale=kwargs['scale'], size=size)
        elif aug_type == "f":
            return np.random.f(dfnum=kwargs['dfnum'], dfden=kwargs['dfden'], size=size)
        elif aug_type == "gamma":
            return np.random.gamma(shape=kwargs['shape'], scale=kwargs['scale'], size=size)
        elif aug_type == "geometric":
            return np.random.geometric(p=kwargs['p'], size=size)
        elif aug_type == "gumbel":
            return np.random.gumbel(loc=kwargs['loc'], scale=kwargs['scale'], size=size)
        elif aug_type == "hypergeometric":
            return np.random.hypergeometric(ngood=kwargs['ngood'], nbad=kwargs['nbad'], nsample=kwargs['nsample'], size=size)
        elif aug_type == "laplace":
            return np.random.laplace(loc=kwargs['loc'], scale=kwargs['scale'], size=size)
        elif aug_type == "logistic":
            return np.random.logistic(loc=kwargs['loc'], scale=kwargs['scale'], size=size)
        elif aug_type == "lognormal":
            return np.random.lognormal(mean=kwargs['mean'], sigma=kwargs['sigma'], size=size)
        elif aug_type == "logseries":
            return np.random.logseries(p=kwargs['p'], size=size)
        elif aug_type == "multinomial":
            return np.random.multinomial(n=kwargs['n'], pvals=kwargs['pvals'], size=size)
        elif aug_type == "multivariate_normal":
            return np.random.multivariate_normal(mean=kwargs['mean'], cov=kwargs['cov'], size=size)
        elif aug_type == "negative_binomial":
            return np.random.negative_binomial(n=kwargs['n'], p=kwargs['p'], size=size)
        elif aug_type == "noncentral_chisquare":
            return np.random.noncentral_chisquare(df=kwargs['df'], nonc=kwargs['nonc'], size=size)
        elif aug_type == "noncentral_f":
            return np.random.noncentral_f(dfnum=kwargs['dfnum'], dfden=kwargs['dfden'], nonc=kwargs['nonc'], size=size)
        elif aug_type == "normal":
            return np.random.normal(loc=kwargs['loc'], scale=kwargs['scale'], size=size)
        elif aug_type == "pareto":
            return np.random.pareto(a=kwargs['a'], size=size)
        elif aug_type == "poisson":
            return np.random.poisson(lam=kwargs['lam'], size=size)
        elif aug_type == "power":
            return np.random.power(a=kwargs['a'], size=size)
        elif aug_type == "rayleigh":
            return np.random.rayleigh(scale=kwargs['scale'], size=size)
        elif aug_type == "standard_cauchy":
            return np.random.standard_cauchy(size=size)
        elif aug_type == "standard_exponential":
            return np.random.standard_exponential(size=size)
        elif aug_type == "standard_gamma":
            return np.random.standard_gamma(shape=kwargs['shape'], size=size)
        elif aug_type == "standard_normal":
            return np.random.standard_normal(size=size)
        elif aug_type == "standard_t":
            return np.random.standard_t(df=kwargs['df'], size=size)
        elif aug_type == "triangular":
            return np.random.triangular(left=kwargs['left'], mode=kwargs['mode'], right=kwargs['right'], size=size)
        elif aug_type == "uniform":
            return np.random.uniform(low=kwargs['low'], high=kwargs['high'], size=size)
        elif aug_type == "vonmises":
            return np.random.vonmises(mu=kwargs['mu'], kappa=kwargs['kappa'], size=size)
        elif aug_type == "wald":
            return np.random.wald(mean=kwargs['mean'], scale=kwargs['scale'], size=size)
        elif aug_type == "weibull":
            return np.random.weibull(a=kwargs['a'], size=size)
        elif aug_type == "zipf":
            return np.random.zipf(a=kwargs['a'], size=size)
        else:
            raise KeyError(f'Unknown distribution type {aug_type}')
