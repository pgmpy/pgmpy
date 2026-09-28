from collections.abc import Hashable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pgmpy.parameterization._base import BaseParameter
from pgmpy.parameterization.distributions import NominalDistribution
from pgmpy.utils import build_state_names, encode_columns, get_state_counts


class TabularCPD(BaseParameter):
    """Tabular conditional probability distribution of a discrete variable given discrete parents.

    ``fit(X, y)`` estimates, by maximum likelihood, the distribution of ``y`` for every combination of the parents'
    states in ``X``, with the same counts as ``DiscreteMLE``. Parents are sorted by name, and parent combinations
    without data get a uniform distribution. Fit a root variable with ``X=None``. ``from_values`` creates a fitted
    instance from a known table instead. Without ``X``, ``predict_proba`` gives the marginal distribution of the target
    over the parent combinations seen in ``fit``; for an instance from ``from_values``, only a root variable has one.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    state_names : dict, optional
        States of the target and of its parents, as ``{variable: [states]}``. The listed states must include every state
        in the data; variables that aren't listed get the sorted states seen in the data.

    Attributes
    ----------
    CPT_ : numpy.ndarray of shape (n_states, n_parent_combinations)
        Probability of each state of the target (rows) for each combination of parent states (columns). The combinations
        follow the product of the parents' states in ``evidence_`` order, with the last parent varying fastest. A root
        variable has a single column.
    state_names_ : dict
        States of the target and of each parent.
    variable_ : hashable
        Name of the target variable.
    evidence_ : list
        Names of the parent variables, sorted.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.parameterization.cpds import TabularCPD
    >>> data = pd.DataFrame({"rain": ["no", "yes", "no", "no"], "wet": ["no", "yes", "yes", "no"]})
    >>> cpd = TabularCPD().fit(data[["rain"]], data["wet"])
    >>> cpd.CPT_.round(2).tolist()
    [[0.67, 0.0], [0.33, 1.0]]
    >>> cpd.sample(pd.DataFrame({"rain": ["yes", "yes"]}), random_state=0)["wet"].tolist()
    ['yes', 'yes']

    Without ``X``, ``predict_proba`` gives the marginal distribution over the parent values seen in ``fit``:

    >>> float(cpd.predict_proba().pmf("yes"))
    0.5

    A root variable is fitted without parents, or created from known probabilities with the arguments of
    ``pgmpy.factors.discrete.TabularCPD``:

    >>> root = TabularCPD.from_values("rain", 2, [[0.75], [0.25]], state_names={"rain": ["no", "yes"]})
    >>> root == TabularCPD().fit(None, data["rain"])
    True
    >>> float(root.predict_proba().pmf("yes"))
    0.25
    """

    _tags = {
        "variable_type": "discrete",
        "supports_weighted_data": True,
        "python_dependencies": "skpro",
        "local:plug_in": ["mle"],
    }

    def __init__(self, state_names: dict | None = None) -> None:
        self.state_names = state_names
        super().__init__()

    @classmethod
    def from_values(
        cls,
        variable: Hashable,
        variable_card: int,
        values: ArrayLike,
        evidence: list | None = None,
        evidence_card: list | None = None,
        state_names: dict | None = None,
    ) -> "TabularCPD":
        """Create a fitted TabularCPD from a known probability table.

        The arguments are those of ``pgmpy.factors.discrete.TabularCPD``, in the same order.

        Parameters
        ----------
        variable : hashable
            Name of the target variable.
        variable_card : int
            Number of states of ``variable``.
        values : array-like of shape (variable_card, product of evidence_card)
            Probability of each state of ``variable`` (rows) for each combination of parent states (columns). The
            combinations follow the product of the parents' states in ``evidence`` order, with the last parent varying
            fastest. Each column must sum to 1, within 0.01.
        evidence : list, optional
            Names of the parent variables. ``None`` for a root variable.
        evidence_card : list, optional
            Number of states of each parent, in ``evidence`` order.
        state_names : dict, optional
            States of ``variable`` and of each parent, as ``{variable: [states]}``. Variables that aren't listed get the
            states ``0, 1, ...``.

        Returns
        -------
        TabularCPD
            A fitted instance. Its parents are sorted by name, with the table reordered to match.
        """
        evidence = [] if evidence is None else list(evidence)
        evidence_card = [] if evidence_card is None else list(evidence_card)
        if len(evidence_card) != len(evidence):
            raise ValueError(f"evidence_card must have one entry per parent in {evidence}, but is {evidence_card}.")
        cardinalities = dict(zip([variable, *evidence], [variable_card, *evidence_card]))
        states = {name: list((state_names or {}).get(name, range(card))) for name, card in cardinalities.items()}
        for name, card in cardinalities.items():
            if len(states[name]) != card:
                raise ValueError(
                    f"{name!r} has {card} states, but state_names lists {len(states[name])}: {states[name]}."
                )
        values = np.asarray(values, dtype=float)
        shape = (variable_card, int(np.prod(evidence_card)))
        if values.shape != shape:
            raise ValueError(
                f"values must have shape {shape} for variable_card and evidence_card, but has {values.shape}."
            )
        if (values < 0).any() or not np.allclose(values.sum(axis=0), 1, atol=0.01):
            raise ValueError("values must be non-negative, and each column must sum to 1.")

        order = sorted(range(len(evidence)), key=lambda position: evidence[position])
        values = values.reshape(variable_card, *evidence_card).transpose(0, *(1 + position for position in order))

        cpd = cls(state_names=state_names)
        cpd.variable_ = variable
        cpd.evidence_ = sorted(evidence)
        cpd.variable_type_ = "discrete"
        cpd._set_table(values.reshape(shape), states)
        cpd._marginal = None if cpd.evidence_ else cpd.CPT_[:, 0]
        cpd._is_fitted = True
        return cpd

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        self.evidence_ = sorted(self.evidence_)
        data = pd.concat([y, X[self.evidence_]], axis=1) if self.evidence_ else y
        state_names = build_state_names(data, self.state_names)
        counts = get_state_counts(data, state_names, self.variable_, self.evidence_, sample_weight)
        counts = counts.to_numpy(dtype=float, copy=True)
        self._marginal = counts.sum(axis=1) / counts.sum()
        counts[:, (counts == 0).all(axis=0)] = 1.0
        self._set_table(counts / counts.sum(axis=0), state_names)

    def _set_table(self, cpt: np.ndarray, state_names: dict) -> None:
        self.CPT_ = cpt
        self.state_names_ = {variable: list(state_names[variable]) for variable in [self.variable_, *self.evidence_]}

    def _predict_proba(self, X: pd.DataFrame | None) -> NominalDistribution:
        if X is None:
            if self._marginal is None:
                raise ValueError(
                    "A TabularCPD with parents created by from_values has no data on its parents, so its marginal "
                    "distribution is unknown; pass X."
                )
            return NominalDistribution(probs=self._marginal, categories=self.state_names_[self.variable_])
        codes, cardinalities = encode_columns(X, self.state_names_)
        columns = np.zeros(len(X), dtype=int)
        for parent in self.evidence_:
            unknown = codes[parent] < 0
            if unknown.any():
                raise ValueError(
                    f"X[{parent!r}] has values not seen in fit: {pd.unique(X[parent][unknown]).tolist()}. Known "
                    f"states: {self.state_names_[parent]}."
                )
            columns = columns * cardinalities[parent] + codes[parent]
        return NominalDistribution(
            probs=self.CPT_.T.take(columns, axis=0),
            categories=self.state_names_[self.variable_],
            index=X.index,
            columns=[self.variable_],
        )
