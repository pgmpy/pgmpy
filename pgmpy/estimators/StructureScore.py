#!/usr/bin/env python
from math import lgamma, log

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import gammaln
from scipy.stats import multivariate_normal

from pgmpy.estimators import BaseEstimator


def get_scoring_method(scoring_method, data, use_cache):
    supported_methods = {
        "k2": K2,
        "bdeu": BDeu,
        "bds": BDs,
        "bic-d": BIC,
        "aic-d": AIC,
        "ll-g": LogLikelihoodGauss,
        "aic-g": AICGauss,
        "bic-g": BICGauss,
        "ll-cg": LogLikelihoodCondGauss,
        "aic-cg": AICCondGauss,
        "bic-cg": BICCondGauss,
    }

    if isinstance(scoring_method, str):
        if scoring_method.lower() in [
            "k2score",
            "bdeuscore",
            "bdsscore",
            "bicscore",
            "aicscore",
        ]:
            raise ValueError(
                f"The scoring method names have been changed. Please refer the documentation."
            )
        elif scoring_method.lower() not in list(supported_methods.keys()):
            raise ValueError(
                f"Unknown scoring method. Please refer documentation for a list of supported score metrics."
            )
    elif not isinstance(scoring_method, StructureScore):
        raise ValueError(
            "scoring_method should either be one of k2score, bdeuscore, bicscore, bdsscore, aicscore, or an instance of StructureScore"
        )

    if isinstance(scoring_method, str):
        score = supported_methods[scoring_method.lower()](data=data)
    else:
        score = scoring_method

    if use_cache:
        from pgmpy.estimators.ScoreCache import ScoreCache

        score_c = ScoreCache(score, data)
    else:
        score_c = score

    return score, score_c


class StructureScore(BaseEstimator):
    """
    Abstract base class for structure scoring classes in pgmpy. Use any of the
    derived classes K2, BDeu, BIC or AIC. Scoring classes
    are used to measure how well a model is able to describe the given data
    set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    Reference
    ---------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3
    """

    def __init__(self, data, **kwargs):
        super(StructureScore, self).__init__(data, **kwargs)

    def score(self, model):
        """
        Computes a score to measure how well the given `DiscreteBayesianNetwork` fits
        to the data set.  (This method relies on the `local_score`-method that
        is implemented in each subclass.)

        Parameters
        ----------
        model: DiscreteBayesianNetwork instance
            The Bayesian network that is to be scored. Nodes of the DiscreteBayesianNetwork need to coincide
            with column names of data set.

        Returns
        -------
        score: float
            A number indicating the degree of fit between data and model

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import K2
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 2)), columns=list('AB'))
        >>> data['C'] = data['B']
        >>> K2(data).score(DiscreteBayesianNetwork([['A','B'], ['A','C']]))
        -24242.367348745247
        >>> K2(data).score(DiscreteBayesianNetwork([['A','B'], ['B','C']]))
        -16273.793897051042
        """

        score = 0
        for node in model.nodes():
            score += self.local_score(node, list(model.predecessors(node)))
        score += self.structure_prior(model)
        return score

    def structure_prior(self, model):
        """A (log) prior distribution over models. Currently unused (= uniform)."""
        return 0

    def structure_prior_ratio(self, operation):
        """Return the log ratio of the prior probabilities for a given proposed change to the DAG.
        Currently unused (=uniform)."""
        return 0


class K2(StructureScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with Dirichlet priors.
    The K2 score is the result of setting all Dirichlet hyperparameters/pseudo_counts to 1.
    The `score`-method measures how well a model is able to describe the given data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3.4-18.3.6 (esp. page 806)
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
    http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    def __init__(self, data, **kwargs):
        super(K2, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_gamma_counts = np.zeros_like(counts, dtype=float)

        # Compute log(gamma(counts + 1))
        gammaln(counts + 1, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + var_cardinality, out=log_gamma_conds)

        # Adjustments when using reindex=False as it drops columns of 0 state counts
        gamma_counts_adj = (
            (num_parents_states - counts.shape[1]) * var_cardinality * gammaln(1)
        )
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(
            var_cardinality
        )

        score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * lgamma(var_cardinality)
        )

        return score


class BDeu(StructureScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with Dirichlet priors.
    The BDeu score is the result of setting all Dirichlet hyperparameters/pseudo_counts to
    `equivalent_sample_size/variable_cardinality`.
    The `score`-method measures how well a model is able to describe the given data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    equivalent_sample_size: int (default: 10)
        The equivalent/imaginary sample size (of uniform pseudo samples) for the dirichlet hyperparameters.
        The score is sensitive to this value, runs with different values might be useful.

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3.4-18.3.6 (esp. page 806)
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
    http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        self.equivalent_sample_size = equivalent_sample_size
        super(BDeu, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        # counts size is different because reindex=False is dropping columns.
        counts_size = num_parents_states * len(self.state_names[variable])
        log_gamma_counts = np.zeros_like(counts, dtype=float)
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts_size
        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        # Adjustment because of missing 0 columns when using reindex=False for computing state_counts to save memory.
        gamma_counts_adj = (
            (num_parents_states - counts.shape[1])
            * len(self.state_names[variable])
            * gammaln(beta)
        )
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + num_parents_states * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score


class BDs(BDeu):
    """
    Class for Bayesian structure scoring for BayesianNetworks with
    Dirichlet priors.  The BDs score is the result of setting all Dirichlet
    hyperparameters/pseudo_counts to
    `equivalent_sample_size/modified_variable_cardinality` where for the
    modified_variable_cardinality only the number of parent configurations
    where there were observed variable counts are considered.  The
    `score`-method measures how well a model is able to describe the given
    data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    equivalent_sample_size: int (default: 10)
        The equivalent/imaginary sample size (of uniform pseudo samples) for the dirichlet
        hyperparameters.
        The score is sensitive to this value, runs with different values might be useful.

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Scutari, Marco. An Empirical-Bayes Score for Discrete Bayesian Networks.
    Journal of Machine Learning Research, 2016, pp. 438–48

    """

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        super(BDs, self).__init__(data, equivalent_sample_size, **kwargs)

    def structure_prior_ratio(self, operation):
        """Return the log ratio of the prior probabilities for a given proposed change to
        the DAG.
        """
        if operation == "+":
            return -log(2.0)
        if operation == "-":
            return log(2.0)
        return 0

    def structure_prior(self, model):
        """
        Implements the marginal uniform prior for the graph structure where each arc
        is independent with the probability of an arc for any two nodes in either direction
        is 1/4 and the probability of no arc between any two nodes is 1/2."""
        nedges = float(len(model.edges()))
        nnodes = float(len(model.nodes()))
        possible_edges = nnodes * (nnodes - 1) / 2.0
        score = -(nedges + possible_edges) * log(2.0)
        return score

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        # counts size is different because reindex=False is dropping columns.
        counts_size = num_parents_states * len(self.state_names[variable])
        log_gamma_counts = np.zeros_like(counts, dtype=float)
        alpha = self.equivalent_sample_size / state_counts.shape[1]
        beta = self.equivalent_sample_size / counts_size
        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        # Adjustment because of missing 0 columns when using reindex=False for computing state_counts to save memory.
        gamma_counts_adj = (
            (num_parents_states - counts.shape[1])
            * len(self.state_names[variable])
            * gammaln(beta)
        )
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + state_counts.shape[1] * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score


class BIC(StructureScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with
    Dirichlet priors.  The BIC/MDL score ("Bayesian Information Criterion",
    also "Minimal Descriptive Length") is a log-likelihood score with an
    additional penalty for network complexity, to avoid overfitting.  The
    `score`-method measures how well a model is able to describe the given
    data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3.4-18.3.6 (esp. page 802)
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
    http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    def __init__(self, data, **kwargs):
        super(BIC, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        sample_size = len(self.data)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        score = np.sum(log_likelihoods)
        score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score


class AIC(StructureScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with
    Dirichlet priors.  The AIC score ("Akaike Information Criterion) is a log-likelihood score with an
    additional penalty for network complexity, to avoid overfitting.  The
    `score`-method measures how well a model is able to describe the given
    data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3.4-18.3.6 (esp. page 802)
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
    http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    def __init__(self, data, **kwargs):
        super(AIC, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        sample_size = len(self.data)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        score = np.sum(log_likelihoods)
        score -= num_parents_states * (var_cardinality - 1)

        return score


class LogLikelihoodGauss(StructureScore):
    def __init__(self, data, **kwargs):
        super(LogLikelihoodGauss, self).__init__(data, **kwargs)

    def _log_likelihood(self, variable, parents):
        if len(parents) == 0:
            glm_model = smf.glm(formula=f"{variable} ~ 1", data=self.data).fit()
        else:
            glm_model = smf.glm(
                formula=f"{variable} ~ {' + '.join(parents)}", data=self.data
            ).fit()

        return (glm_model.llf, glm_model.df_model)

    def local_score(self, variable, parents):
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        return ll


class BICGauss(LogLikelihoodGauss):
    def __init__(self, data, **kwargs):
        super(BICGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        # Adding +2 to model df to compute the likelihood df.
        return ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))


class AICGauss(LogLikelihoodGauss):
    def __init__(self, data, **kwargs):
        super(AICGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        # Adding +2 to model df to compute the likelihood df.
        return ll - (df_model + 2)


class LogLikelihoodCondGauss(StructureScore):
    """
    References
    ----------
    [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian
        Networks of Mixed Variables. International journal of data science and
        analytics, 6(1), 3–18. https://doi.org/10.1007/s41060-017-0085-7
    """

    def __init__(self, data, **kwargs):
        super(LogLikelihoodCondGauss, self).__init__(data, **kwargs)

    @staticmethod
    def _adjusted_cov(df):
        """
        Computes an adjusted covariance matrix from the given dataframe.
        """
        # If a number of rows less than number of variables, return variance 1 with no covariance.
        if (df.shape[0] == 1) or (df.shape[0] < len(df.columns)):
            return pd.DataFrame(
                np.eye(len(df.columns)), index=df.columns, columns=df.columns
            )

        # If the matrix is not positive semidefinite, add a small error to make it.
        df_cov = df.cov()
        if np.any(np.isclose(np.linalg.eig(df_cov)[0], 0)):
            df_cov = df_cov + 1e-6
        return df_cov

    def _cat_parents_product(self, parents):
        k = 1
        for pa in parents:
            if self.dtypes[pa] != "N":
                n_states = self.data[pa].nunique()
                if n_states > 1:
                    k *= self.data[pa].nunique()
        return k

    def _get_num_parameters(self, variable, parents):
        parent_dtypes = [self.dtypes[pa] for pa in parents]
        n_cont_parents = parent_dtypes.count("N")

        if self.dtypes[variable] == "N":
            k = self._cat_parents_product(parents=parents) * (n_cont_parents + 2)
        else:
            if n_cont_parents == 0:
                k = self._cat_parents_product(parents=parents) * (
                    self.data[variable].nunique() - 1
                )
            else:
                k = (
                    self._cat_parents_product(parents=parents)
                    * (self.data[variable].nunique() - 1)
                    * (n_cont_parents + 2)
                )

        return k

    def _log_likelihood(self, variable, parents):
        df = self.data.loc[:, [variable] + parents]

        # If variable is continuous, the probability is computed as:
        # P(C1 | C2, D) = p(C1, C2 | D) / p(C2 | D)
        if self.dtypes[variable] == "N":
            c1 = variable
            c2 = [var for var in parents if self.dtypes[var] == "N"]
            d = list(set(parents) - set(c2))

            # If D = {}, p(C1, C2 | D) = p(C1, C2) and p(C2 | D) = p(C2)
            if len(d) == 0:
                # If C2 = {}, p(C1, C2 | D) = p(C1) and p(C2 | D) = 1.
                if len(c2) == 0:
                    p_c1c2_d = multivariate_normal.pdf(
                        x=df,
                        mean=df.mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df),
                        allow_singular=True,
                    )
                    return np.sum(np.log(p_c1c2_d))
                else:
                    p_c1c2_d = multivariate_normal.pdf(
                        x=df,
                        mean=df.mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df),
                        allow_singular=True,
                    )
                    df_c2 = df.loc[:, c2]
                    p_c2_d = np.maximum(
                        1e-8,
                        multivariate_normal.pdf(
                            x=df_c2,
                            mean=df_c2.mean(axis=0),
                            cov=LogLikelihoodCondGauss._adjusted_cov(df_c2),
                            allow_singular=True,
                        ),
                    )

                    return np.sum(np.log(p_c1c2_d / p_c2_d))
            else:
                log_like = 0
                for d_states, df_d in df.groupby(d, observed=True):
                    p_c1c2_d = multivariate_normal.pdf(
                        x=df_d.loc[:, [c1] + c2],
                        mean=df_d.loc[:, [c1] + c2].mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(
                            df_d.loc[:, [c1] + c2]
                        ),
                        allow_singular=True,
                    )
                    if len(c2) == 0:
                        p_c2_d = 1
                    else:
                        p_c2_d = np.maximum(
                            1e-8,
                            multivariate_normal.pdf(
                                x=df_d.loc[:, c2],
                                mean=df_d.loc[:, c2].mean(axis=0),
                                cov=LogLikelihoodCondGauss._adjusted_cov(
                                    df_d.loc[:, c2]
                                ),
                                allow_singular=True,
                            ),
                        )

                    log_like += np.sum(np.log(p_c1c2_d / p_c2_d))
                return log_like

        # If variable is discrete, the probability is computed as:
        # P(D1 | C, D2) = (p(C| D1, D2) p(D1, D2)) / (p(C| D2) p(D2))
        else:
            d1 = variable
            c = [var for var in parents if self.dtypes[var] == "N"]
            d2 = list(set(parents) - set(c))

            log_like = 0
            for d_states, df_d1d2 in df.groupby([d1] + d2, observed=True):
                # Check if df_d1d2 also has the discrete variables.
                # If C={}, p(C | D1, D2) = 1.
                if len(c) == 0:
                    p_c_d1d2 = 1
                else:
                    p_c_d1d2 = multivariate_normal.pdf(
                        x=df_d1d2.loc[:, c],
                        mean=df_d1d2.loc[:, c].mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df_d1d2.loc[:, c]),
                        allow_singular=True,
                    )

                # P(D1, D2)
                p_d1d2 = np.repeat(df_d1d2.shape[0] / df.shape[0], df_d1d2.shape[0])

                # If D2 = {}, p(D1 | C, D2) = (p(C | D1, D2) p(D1, D2)) / p(C)
                if len(d2) == 0:
                    if len(c) == 0:
                        p_c_d2 = 1
                    else:
                        p_c_d2 = np.maximum(
                            1e-8,
                            multivariate_normal.pdf(
                                x=df_d1d2.loc[:, c],
                                mean=df.loc[:, c].mean(axis=0),
                                cov=LogLikelihoodCondGauss._adjusted_cov(df.loc[:, c]),
                                allow_singular=True,
                            ),
                        )

                    log_like += np.sum(np.log(p_c_d1d2 * p_d1d2 / p_c_d2))
                else:
                    if len(c) == 0:
                        p_c_d2 = 1
                    else:
                        df_d2 = df
                        for var, state in zip(d2, d_states[1:]):
                            df_d2 = df_d2.loc[df_d2[var] == state]

                        p_c_d2 = np.maximum(
                            1e-8,
                            multivariate_normal.pdf(
                                x=df_d1d2.loc[:, c],
                                mean=df_d2.loc[:, c].mean(axis=0),
                                cov=LogLikelihoodCondGauss._adjusted_cov(
                                    df_d2.loc[:, c]
                                ),
                                allow_singular=True,
                            ),
                        )

                    p_d2 = df.groupby(d2, observed=True).count() / df.shape[0]
                    for var, value in zip(d2, d_states[1:]):
                        p_d2 = p_d2.loc[p_d2.index.get_level_values(var) == value]

                    log_like += np.sum(
                        np.log((p_c_d1d2 * p_d1d2) / (p_c_d2 * p_d2.values.ravel()[0]))
                    )
            return log_like

    def local_score(self, variable, parents):
        ll = self._log_likelihood(variable=variable, parents=parents)
        return ll


class BICCondGauss(LogLikelihoodCondGauss):
    def __init__(self, data, **kwargs):
        super(BICCondGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - ((k / 2) * np.log(self.data.shape[0]))


class AICCondGauss(LogLikelihoodCondGauss):
    def __init__(self, data, **kwargs):
        super(AICCondGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - k
