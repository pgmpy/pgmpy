from __future__ import annotations

from skbase.base import BaseEstimator

from pgmpy.base import DAG
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import build_state_names, preprocess_data


class _BaseDiscreteParameterEstimator(BaseEstimator):
    """
    Base class for fit-based discrete parameter estimators.

    Parameters
    ----------
    state_names: dict, optional
        A dict indicating, for each variable, the discrete set of states that the variable can take. If unspecified, the
        observed values in the data set are taken to be the only possible states.
    """

    _tags = {
        "supported_model_types": (DAG, DiscreteBayesianNetwork),
        "supports_latent_variables": False,
        "supports_weighted_data": False,
    }

    def __init__(self, state_names: dict | None = None) -> None:
        self.state_names = state_names
        super().__init__()

    def fit(self, model: DAG | DiscreteBayesianNetwork, data):
        """
        Fit the estimator on a model and dataset.

        Parameters
        ----------
        model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork
            The model structure for which to estimate parameters.

        data: pandas.DataFrame
            DataFrame object with column names identical to the variable names of the network. If some values are
            missing, the corresponding cells should be set to `numpy.nan`.

        Returns
        -------
        self
            Fitted estimator with learned parameters stored in `parameters_` and inferred state names stored in
            `state_names_`.
        """
        raise NotImplementedError

    def _coerce_model(self, model: DAG | DiscreteBayesianNetwork) -> DiscreteBayesianNetwork:
        supported_model_types = self._tags["supported_model_types"]

        if not isinstance(model, supported_model_types):
            raise NotImplementedError(
                f"{type(self).__name__} is only implemented for "
                f"{', '.join(cls.__name__ for cls in supported_model_types)}"
            )

        if isinstance(model, DAG) and not isinstance(model, DiscreteBayesianNetwork):
            model_bn = DiscreteBayesianNetwork(model.edges())
            model_bn.add_nodes_from(model.nodes())
            model_bn.latents = set(model.latents)
            return model_bn

        return model

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

    def _validate_model_data(self, model: DiscreteBayesianNetwork, data) -> None:
        supports_latent_variables = bool(self._tags["supports_latent_variables"])
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

    def _initialize_fit(self, model: DAG | DiscreteBayesianNetwork, data) -> None:
        model = self._coerce_model(model)
        data, _ = preprocess_data(data)

        self._validate_model_data(model, data)

        self._model = model
        self._data = data
        self.state_names_ = self._build_fitted_state_names(model, data)
