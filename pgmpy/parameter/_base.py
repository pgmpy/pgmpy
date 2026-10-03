import numpy as np
import pandas as pd
from skbase.base import BaseEstimator as _BaseEstimator

from pgmpy.causal_discovery._base import BaseCausalDiscovery


class BaseParameter(_BaseEstimator):
    """Base class for all parameter classes in pgmpy."""

    _tags = {
        "object_type": set,  # {"continuous", "discrete", "mixture"}
        "fit_mode": set,  # {"supervise", "unsupervise", "untraninable"}
        "python_dependencies": set,
        "local:plug_in": set,
        "global:plug_in": set,
        "local:full_bayesian": set,
        "global:full_bayesian": set,
    }

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        """Fit parameters to training data.

        If a parameter corresponds to a root node of the Bayesian network,
        only unsupervised learning is supported.
        If a parameter does not correspond to a root node of the Bayesian network,
        only supervised learning is supported.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data for supervised or unsupervised learning.

        y : pandas.DataFrame, (default=None)
            Target labels for supervised learning.
            Must be None for unsupervised learning.

        sample_weight : numpy.ndarray, (default=None)
            Sample weights. If None, all samples are given equal weight.

        Returns
        -------
        self : pgmpy parameter class
            Returns the parameter instance with the fitted attributes.
        """
        X, y, sample_weight = self._check_fit_data(X, y, sample_weight)
        self._fit(X, y, sample_weight)

        if hasattr(self, "object_type_") is False:
            raise AttributeError(f"{self.__class__.__name__}._fit() must set the 'object_type_' attribute.")
        if hasattr(self, "fit_mode_") is False:
            raise AttributeError(f"{self.__class__.__name__}._fit() must set the 'fit_mode_' attribute.")

        self._is_fitted = True
        return self

    def _fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def predict_proba(self, X):
        """Return the predicted conditional probability distribution for the given data.

        Only supported when the parameter was fitted using supervised learning.
        If the parameter was fitted using unsupervised learning, refer to the ``sample()`` method instead.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data used to predict the conditional probability distribution.
            Must have the same columns as the training data.

        Returns
        -------
        y : skpro distribution class
            Predicted conditional probability distribution.

        See Also
        --------
        sample : Generate samples from the distribution and their corresponding log probabilities.
        """
        X = self._check_predict_proba_data(X)
        y_pred = self._predict_proba(X)
        return y_pred

    def _predict_proba(self, X):
        raise NotImplementedError

    def sample(self, X=None, n_samples=None):
        """Draw samples from the probability distribution or conditional probability distribution.

        If the parameter was fitted using supervised learning, samples are drawn from
        the conditional probability distribution given X.
        If the parameter was fitted using unsupervised learning, samples are drawn from
        the (unconditional) probability distribution.

        Parameters
        ----------
        X : pandas.DataFrame, (default=None)
            Required when the parameter was fitted with supervised learning.
            Raises an error if the parameter was fitted with unsupervised learning;
            use ``n_samples`` instead.

        n_samples : int, (default=None)
            Required when the parameter was fitted with unsupervised learning.
            Raises an error if the parameter was fitted with supervised learning;
            use ``X`` instead.

        Returns
        -------
        y : numpy.ndarray or torch.Tensor
            Samples drawn from the distribution.

        log_proba : numpy.ndarray or torch.Tensor
            Log pdf or log pmf values corresponding to the drawn samples.

        See Also
        --------
        predict_proba : Return the predicted conditional probability distribution.
        """
        X, n_samples = self._check_sample_data(X, n_samples)
        samples, log_proba = self._sample(X, n_samples)
        return samples, log_proba

    def _sample(self, X, n_samples):
        raise NotImplementedError

    def _check_fit_data(self, X, y=None, sample_weight=None):
        """check train data with tag"""
        if (y is None) and ("unsupervise" in self.get_class_tag("fit_mode")):
            transformer = BaseCausalDiscovery()
            X = transformer._check_fit_data(X)
            self.fit_mode_ = "unsupervise"
            self.evidences_ = None
            self.variables_ = transformer.feature_names_in_
            if ("continuous" in self.get_class_tag("object_type")) and (
                pd.api.types.is_numeric_dtype(X[self.variables_[0]])
            ):
                self.object_type_ = "continuous"
            elif ("discrete" in self.get_class_tag("object_type")) and (
                pd.api.types.is_string_dtype(X[self.variables_[0]])
            ):
                self.object_type_ = "discrete"
            else:
                raise ValueError(
                    f"This {self.__class__.__name__} instance only supports data of type "
                    f"{self.get_class_tag('object_type')}."
                )

        elif (y is not None) and ("supervise" in self.get_class_tag("fit_mode")):
            transformer = BaseCausalDiscovery()
            X = transformer._check_fit_data(X)
            self.fit_mode_ = "supervise"
            self.evidences_ = transformer.feature_names_in_
            y = transformer._check_fit_data(y)
            self.variables_ = transformer.feature_names_in_
            if ("continuous" in self.get_class_tag("object_type")) and (
                pd.api.types.is_numeric_dtype(y[self.variables_[0]])
            ):
                self.object_type_ = "continuous"
            elif ("discrete" in self.get_class_tag("object_type")) and (
                pd.api.types.is_string_dtype(y[self.variables_[0]])
            ):
                self.object_type_ = "discrete"
            else:
                raise ValueError(
                    f"This {self.__class__.__name__} instance only supports data of type "
                    f"{self.get_class_tag('object_type')}."
                )
        else:
            raise ValueError(
                f"This {self.__class__.__name__} instance only supports {self.get_class_tag('fit_mode')} learning."
            )

        # TODO: Implement missing data in suvervised, unsuvervised learning (EM algo)

        # Check sample_weight
        n_samples = len(X)
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)

            if len(sample_weight) != n_samples:
                raise ValueError(f"sample_weight must have length {n_samples}. Got {len(sample_weight)}.")

            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative values.")

        return X, y, sample_weight

    def _check_predict_proba_data(self, X):
        if not self.is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' before calling 'predict_proba'."
            )
        if self.fit_mode_ in {"unsupervise", "untrainable"}:
            raise NotImplementedError(
                f"This {self.__class__.__name__} instance is {self.fit_mode_}. It does not support `predict_proba()`."
            )
        elif self.fit_mode_ == "supervise":
            transformer = BaseCausalDiscovery()
            X = transformer._check_fit_data(X)

        return X

    def _check_sample_data(self, X=None, n_samples=None):
        if not self.is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' before calling 'sample'."
            )

        if (X is None) and (n_samples) and (self.fit_mode_ in {"unsupervise", "untrainable"}):
            return X, n_samples

        elif (X is not None) and (n_samples is None) and (self.fit_mode_ in {"supervise"}):
            transformer = BaseCausalDiscovery()
            X = transformer._check_fit_data(X)
            return X, n_samples
        else:
            raise ValueError(
                f"This {self.__class__.__name__} instance supports {self.fit_mode_}. "
                f"Please use `X` or `n_samples` accordingly."
            )

        return X, n_samples
