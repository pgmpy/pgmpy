from typing import Iterable, Optional, Union

from pgmpy.base import DAG


class SimpleCausalModel(DAG):
    """
    A specialized DAG class for simple causal models.

    This class simplifies the creation of causal graphs commonly used in causal inference,
    where the structure consists of exposures, outcomes, covariates (confounders), mediators, and instruments.
    It automatically adds the standard edges:
        - Exposures -> Outcomes
        - Covariates -> Exposures
        - Covariates -> Outcomes
        - Instruments -> Exposures
        - Exposures -> Mediators
        - Mediators -> Outcomes

    Parameters
    ----------
    exposures: str, int, or iterable
        The exposure variable(s). If an int is provided, the variable name will be generated as "Var_<int>".

    outcomes: str, int, or iterable
        The outcome variable(s). If an int is provided, the variable name will be generated as "Var_<int>".

    covariates: str, int, iterable, or None (default: None)
        The covariate (confounder) variable(s). If an int is provided, the variable name will be
        generated as "Var_<int>".

    mediators: str, int, iterable, or None (default: None)
        The mediator variable(s). If an int is provided, the variable name will be generated as "Var_<int>".

    instruments: str, int, iterable, or None (default: None)
        The instrumental variable(s). If an int is provided, the variable name will be generated as "Var_<int>".

    latents: iterable or None (default: None)
        List of latent variables.

    Notes
    -----
    If any of the `exposures`, `outcomes`, `covariates`, `mediators`, or `instruments` arguments are
    provided as integers, they will be automatically converted to variable names in the format
    "Var_<int>".

    Examples
    --------
    >>> from pgmpy.base.simple_causal_model import SimpleCausalModel
    >>> model = SimpleCausalModel(
    ...     exposures="X", outcomes="Y", covariates="Z", mediators="M", instruments="I"
    ... )
    >>> model.edges()
    OutEdgeView([('X', 'Y'), ('Z', 'X'), ('Z', 'Y'), ('I', 'X'), ('X', 'M'), ('M', 'Y')])
        If any of the `exposures`, `outcomes`, `covariates`, `mediators`, or `instruments` arguments are
        provided as integers, they will be automatically converted to variable names in the format "Var_<int>".
    >>> model2 = SimpleCausalModel(
    ...     exposures=1, outcomes=2, covariates=3, mediators=4, instruments=5
    ... )
    >>> model2.nodes()
    ['Var_1', 'Var_2', 'Var_3', 'Var_4', 'Var_5']
    >>> model2.edges()
    OutEdgeView([
            ('Var_1', 'Var_2'), ('Var_3', 'Var_1'), ('Var_3', 'Var_2'),
            ('Var_5', 'Var_1'), ('Var_1', 'Var_4'), ('Var_4', 'Var_2')])
    """

    @staticmethod
    def _to_list(var):
        if var is None:
            return []
        elif isinstance(var, str):
            return [var]
        elif isinstance(var, int):
            return [f"Var_{var}"]
        return list(var)

    def __init__(
        self,
        exposures: Union[str, int, Iterable[Union[str, int]]],
        outcomes: Union[str, int, Iterable[Union[str, int]]],
        covariates: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
        mediators: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
        instruments: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
        latents: Optional[Iterable[str]] = None,
    ):
        exposures = self._to_list(exposures)
        outcomes = self._to_list(outcomes)
        covariates = self._to_list(covariates)
        mediators = self._to_list(mediators)
        instruments = self._to_list(instruments)
        latents = list(latents) if latents is not None else []

        edges = []

        # Add edges from exposures to outcomes
        edges += [(exp, out) for exp in exposures for out in outcomes]

        # Add edges from covariates to exposures and outcomes
        edges += [(cov, exp) for cov in covariates for exp in exposures]
        edges += [(cov, out) for cov in covariates for out in outcomes]

        # Add edges from instruments to exposures
        edges += [(inst, exp) for inst in instruments for exp in exposures]

        # Add edges from exposures to mediators and mediators to outcomes
        edges += [(exp, med) for exp in exposures for med in mediators]
        edges += [(med, out) for med in mediators for out in outcomes]

        super().__init__(edges, latents=latents)

        # Add latent nodes if not already present
        for latent in latents:
            if latent not in self.nodes:
                self.add_node(latent)

        self.variable_roles = {
            "exposure": set(exposures),
            "outcome": set(outcomes),
            "covariate": set(covariates),
            "mediator": set(mediators),
            "instrument": set(instruments),
        }
