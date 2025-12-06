from typing import Iterable, Optional, Union

from pgmpy.base import DAG


class SimpleCausalModel(DAG):
    """
    A specialized DAG class for simple causal models.

    This class simplifies the creation of causal graphs commonly used in causal inference,
    where the structure consists of exposures, outcomes, covariates (confounders), mediators, and instruments.
    It automatically adds the standard edges:
        - Exposures -> Outcomes (only if there are no mediators)
        - Covariates -> Exposures
        - Covariates -> Outcomes
        - Instruments -> Exposures
        - Exposures -> Mediators
        - Mediators -> Outcomes

    If you want more control over the model structure, use the DAG class directly.

    Notes
    -----
    A standard causal diagram (with mediators):

        I ---> E ---> M ---> O
               ^             ^
               |             |
               X-------------+

    Where:
        I: Instrument
        E: Exposure
        M: Mediator
        O: Outcome
        X: Covariate (affects both E and O)


    If no mediators:
        I ---> E ---> O
               ^      ^
               |      |
               X------+

    Parameters
    ----------
    exposures: str, int, or iterable
        The exposure variable(s). If an int 'n' is provided, 'n' variables
        will be generated with role-based prefixes: ``E_0, E_1, ..., E_{n-1}``.

    outcomes: str, int, or iterable
        The outcome variable(s). If an int 'n' is provided, 'n' variables
        will be generated with role-based prefixes: ``O_0, O_1, ..., O_{n-1}``.

    covariates: str, int, iterable, or None (default: None)
        The covariate (confounder) variable(s). If an int 'n' is provided,
        'n' variables will be generated with role-based prefixes:
        ``X_0, X_1, ..., X_{n-1}``.

    mediators: str, int, iterable, or None (default: None)
        The mediator variable(s). If an int 'n' is provided, 'n' variables
        will be generated with role-based prefixes: ``M_0, M_1, ..., M_{n-1}``.

    instruments: str, int, iterable, or None (default: None)
        The instrumental variable(s). If an int 'n' is provided, 'n' variables
        will be generated with role-based prefixes: ``I_0, I_1, ..., I_{n-1}``.

    latents: iterable or None (default: None)
        List of latent variables.

    Examples
    --------

    >>> from pgmpy.base import SimpleCausalModel
    >>> model = SimpleCausalModel(
    ...     exposures="X", outcomes="Y", covariates="Z", mediators="M", instruments="I"
    ... )
    >>> model.edges()
    OutEdgeView([('Z', 'X'), ('Z', 'Y'), ('I', 'X'), ('X', 'M'), ('M', 'Y')])

    >>> model2 = SimpleCausalModel(
    ...     exposures=1, outcomes=2, covariates=2, mediators=None, instruments=1
    ... )
    >>> sorted(model2.nodes())
    ['E_0', 'I_0', 'O_0', 'O_1', 'X_0', 'X_1']
    >>> from pprint import pprint
    >>> pprint(sorted(model2.edges()))
    [('E_0', 'O_0'),
     ('E_0', 'O_1'),
     ('I_0', 'E_0'),
     ('X_0', 'E_0'),
     ('X_0', 'O_0'),
     ('X_0', 'O_1'),
     ('X_1', 'E_0'),
     ('X_1', 'O_0'),
     ('X_1', 'O_1')]

    """

    @staticmethod
    def _to_list(var, role=None):
        if var is None:
            return []
        elif isinstance(var, str):
            return [var]
        elif isinstance(var, int):
            prefix = {
                "exposures": "E_",
                "outcomes": "O_",
                "covariates": "X_",
                "mediators": "M_",
                "instruments": "I_",
            }.get(role, "Var_")
            return [f"{prefix}{i}" for i in range(var)]
        elif isinstance(var, Iterable):
            return [str(v) for v in var]
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
        exposures = self._to_list(exposures, "exposures")
        outcomes = self._to_list(outcomes, "outcomes")
        covariates = self._to_list(covariates, "covariates")
        mediators = self._to_list(mediators, "mediators")
        instruments = self._to_list(instruments, "instruments")
        latents = list(latents) if latents is not None else []

        edges = []

        # Add edges from exposures to outcomes only if there are no mediators
        if not mediators:
            edges += [(exp, out) for exp in exposures for out in outcomes]

        # Add edges from covariates to exposures and outcomes
        edges += [(cov, exp) for cov in covariates for exp in exposures]
        edges += [(cov, out) for cov in covariates for out in outcomes]

        # Add edges from instruments to exposures
        edges += [(inst, exp) for inst in instruments for exp in exposures]

        # Add edges from exposures to mediators and mediators to outcomes
        edges += [(exp, med) for exp in exposures for med in mediators]
        edges += [(med, out) for med in mediators for out in outcomes]

        roles = {
            "exposures": set(exposures),
            "outcomes": set(outcomes),
            "covariates": set(covariates),
            "mediators": set(mediators),
            "instruments": set(instruments),
        }

        super().__init__(edges, roles=roles)

        latents_set = set(latents) if latents else set()
        for latent in latents_set:
            if latent not in self.nodes():
                raise ValueError(
                    f"Latent variable '{latent}' is not in the graph nodes."
                )
        self.latents = latents_set
