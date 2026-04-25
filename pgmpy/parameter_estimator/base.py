from __future__ import annotations

import numpy as np
from skbase.base import BaseEstimator

from pgmpy.base import DAG
from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.utils import build_state_names, preprocess_data


class BaseParameterEstimator(BaseEstimator):
    """
    Thin base class for all parameter estimators.

    Subclasses must set the `supported_model_types` tag and implement `fit`.

    Attributes
    ----------
    parameters_ : list
        Learned parameters produced by `fit`. The element type depends on the
        concrete subclass (for example, `TabularCPD` for discrete estimators
        and `LinearGaussianCPD` for Gaussian estimators).
    """

    _tags = {
        "supported_model_types": (),
        "supports_weighted_data": False,
    }

    def fit(self, model, data, sample_weight=None):
        """
        Fit the estimator on a model and dataset.

        Parameters
        ----------
        sample_weight: array-like of shape (n_samples,), optional
            Per-row weights for the data. Only accepted by estimators whose
            `supports_weighted_data` tag is True.

        Returns
        -------
        self
            Fitted estimator with learned parameters stored in `parameters_`.
        """
        raise NotImplementedError

    def _validate_inputs(self, model, data, sample_weight=None):
        supported_model_types = self.get_tag("supported_model_types")
        if not isinstance(model, supported_model_types):
            raise NotImplementedError(
                f"{type(self).__name__} is only implemented for "
                f"{', '.join(cls.__name__ for cls in supported_model_types)}"
            )
        if sample_weight is not None:
            if not bool(self.get_tag("supports_weighted_data")):
                raise ValueError(f"{type(self).__name__} does not support `sample_weight`.")
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()
            if sample_weight.shape[0] != data.shape[0]:
                raise ValueError(
                    f"sample_weight has length {sample_weight.shape[0]} but data has {data.shape[0]} rows."
                )
        return model, sample_weight

    def _sort_parameters(self, parameters: list) -> list:
        order = {var: index for index, var in enumerate(self._model.nodes())}
        return sorted(parameters, key=lambda cpd: order[cpd.variable])


class DiscreteParameterEstimator(BaseParameterEstimator):
    """
    Base class for discrete parameter estimators.

    Parameters
    ----------
    state_names: dict, optional
        A dict indicating, for each variable, the discrete set of states that the variable can take. If unspecified, the
        observed values in the data set are taken to be the only possible states.

    Attributes
    ----------
    parameters_ : list of TabularCPD
        Learned conditional probability distributions, one per variable in the
        model, ordered by `self._model.nodes()`. Populated by `fit`.

    state_names_ : dict
        Mapping from variable name to the list of states for that variable,
        inferred from the data (or taken from `state_names` when supplied).
        Populated by `fit`.
    """

    _tags = {
        "supported_model_types": (DAG, DiscreteBayesianNetwork),
        "supports_latent_variables": False,
        "supports_weighted_data": False,
    }

    def __init__(self, state_names: dict | None = None) -> None:
        self.state_names = state_names
        super().__init__()

    def fit(self, model: DAG | DiscreteBayesianNetwork, data, sample_weight=None):
        """
        Fit the estimator on a model and dataset.

        Parameters
        ----------
        model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork
            The model structure for which to estimate parameters.

        data: pandas.DataFrame
            DataFrame object with column names identical to the variable names of the network. If some values are
            missing, the corresponding cells should be set to `numpy.nan`.

        sample_weight: array-like of shape (n_samples,), optional
            Per-row weights for `data`. Only accepted by estimators whose
            `supports_weighted_data` tag is True.

        Returns
        -------
        self
            Fitted estimator with learned parameters stored in `parameters_` and inferred state names stored in
            `state_names_`.
        """
        raise NotImplementedError

    def _validate_inputs(
        self,
        model: DAG | DiscreteBayesianNetwork,
        data,
        sample_weight=None,
    ) -> tuple[DiscreteBayesianNetwork, np.ndarray | None]:
        model, sample_weight = super()._validate_inputs(model, data, sample_weight=sample_weight)

        if isinstance(model, DAG) and not isinstance(model, DiscreteBayesianNetwork):
            model_bn = DiscreteBayesianNetwork(model.edges())
            model_bn.add_nodes_from(model.nodes())
            model_bn.latents = set(model.latents)
            model = model_bn

        supports_latent_variables = self.get_tag("supports_latent_variables")
        if (not supports_latent_variables) and model.latents:
            raise ValueError(
                f"Found latent variables: {model.latents}. {type(self).__name__} doesn't support latent variables."
            )

        observed_nodes = set(model.nodes()) - set(model.latents)
        missing_nodes = observed_nodes - set(data.columns)
        if missing_nodes:
            raise ValueError(
                "Nodes detected in the model that are not present in the dataset: "
                f"{missing_nodes}. Refine the model so that all parameters can be estimated from the data."
            )

        return model, sample_weight

    def _build_fitted_state_names(
        self,
        model: DiscreteBayesianNetwork,
        data,
    ) -> dict:
        supplied_state_names = self.state_names if isinstance(self.state_names, dict) else None
        model_columns = [var for var in data.columns if var in model.nodes()]
        state_names = build_state_names(data.loc[:, model_columns], state_names=supplied_state_names)

        if supplied_state_names is not None:
            for var in model.nodes():
                if (var not in state_names) and (var in supplied_state_names):
                    state_names[var] = supplied_state_names[var]

        return {var: list(states) for var, states in state_names.items()}

    def _initialize_fit(
        self,
        model: DAG | DiscreteBayesianNetwork,
        data,
        sample_weight=None,
    ) -> None:
        data, _ = preprocess_data(data)
        model, sample_weight = self._validate_inputs(model, data, sample_weight=sample_weight)
        self._model = model
        self._data = data
        self._sample_weight = sample_weight
        self.state_names_ = self._build_fitted_state_names(model, data)


class GaussianParameterEstimator(BaseParameterEstimator):
    """
    Base class for LinearGaussian parameter estimators.

    Attributes
    ----------
    parameters_ : list of LinearGaussianCPD
        Learned Gaussian conditional probability distributions, one per
        variable in the model, ordered by `self._model.nodes()`. Populated by
        `fit`.
    """

    _tags = {
        "supported_model_types": (LinearGaussianBayesianNetwork,),
        "supports_latent_variables": False,
        "supports_weighted_data": False,
    }

    def fit(self, model: LinearGaussianBayesianNetwork, data, sample_weight=None):
        """
        Fit the estimator on a model and dataset.

        Parameters
        ----------
        model: pgmpy.models.LinearGaussianBayesianNetwork
            The model structure for which to estimate parameters.

        data: pandas.DataFrame
            DataFrame object with column names identical to the variable names
            of the network.

        sample_weight: array-like of shape (n_samples,), optional
            Per-row weights for `data`. Only accepted by estimators whose
            `supports_weighted_data` tag is True.

        Returns
        -------
        self
            Fitted estimator with learned parameters stored in `parameters_`.
        """
        raise NotImplementedError

    def _validate_inputs(
        self,
        model: LinearGaussianBayesianNetwork,
        data,
        sample_weight=None,
    ) -> tuple[LinearGaussianBayesianNetwork, np.ndarray | None]:
        model, sample_weight = super()._validate_inputs(model, data, sample_weight=sample_weight)
        missing_nodes = set(model.nodes()) - set(data.columns)
        if missing_nodes:
            raise ValueError(
                "Nodes detected in the model that are not present in the dataset: "
                f"{missing_nodes}. Refine the model so that all parameters can be estimated from the data."
            )
        return model, sample_weight

    def _initialize_fit(
        self,
        model: LinearGaussianBayesianNetwork,
        data,
        sample_weight=None,
    ) -> None:
        model, _ = self._validate_inputs(model, data, sample_weight=sample_weight)
        self._model = model
        self._data = data


_BaseParameterEstimator = BaseParameterEstimator
_BaseDiscreteParameterEstimator = DiscreteParameterEstimator
_BaseGaussianParameterEstimator = GaussianParameterEstimator
