from collections.abc import Hashable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pgmpy.parameterization._base import BaseParameter, _checked_evidence, _parent_order
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
    >>> cpd.predict(pd.DataFrame({"rain": ["no", "yes"]}))["wet"].tolist()
    ['no', 'yes']
    >>> cpd.log_likelihood(data[["rain"]], data["wet"]).round(2)["wet"].tolist()
    [-0.41, 0.0, -1.1, -0.41]

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
        "name": "tabular_cpd",
        "variable_type": ["discrete"],
        "parent_data_types": ["discrete"],
        "supports_weighted_data": True,
        "capability:factor": True,
        "capability:exact_inference": True,
        "python_dependencies": "skpro",
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
        evidence: list | tuple | None = None,
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
        evidence : list or tuple, optional
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
        # Step 1: Check the parents and their cardinalities, and give every variable its states.
        evidence = _checked_evidence(variable, evidence)
        evidence_card = [] if evidence_card is None else list(evidence_card)
        if len(evidence_card) != len(evidence):
            raise ValueError(f"evidence_card must have one entry per parent in {evidence}, but is {evidence_card}.")
        given = _checked_state_names(state_names)
        cardinalities = dict(zip([variable, *evidence], [variable_card, *evidence_card]))
        states = {name: given.get(name, list(range(card))) for name, card in cardinalities.items()}
        for name, card in cardinalities.items():
            if len(states[name]) != card:
                raise ValueError(
                    f"{name!r} has {card} states, but state_names lists {len(states[name])}: {states[name]}."
                )
        # Step 2: Check that values is a table of probabilities of the right shape.
        values = np.array(values, dtype=float)
        shape = (variable_card, int(np.prod(evidence_card)))
        if values.shape != shape:
            raise ValueError(
                f"values must have shape {shape} for variable_card and evidence_card, but has {values.shape}."
            )
        if (values < 0).any() or not np.allclose(values.sum(axis=0), 1, atol=0.01):
            raise ValueError("values must be non-negative, and each column must sum to 1.")

        # Step 3: Sort the parents by name, and reorder the table's columns to match.
        order = _parent_order(evidence)
        values = values.reshape(variable_card, *evidence_card).transpose(0, *(1 + position for position in order))

        # Step 4: Create the fitted instance. Only a root has a known marginal.
        cpd = cls(state_names=state_names)
        cpd.variable_ = variable
        cpd.evidence_ = [evidence[position] for position in order]
        cpd.variable_type_ = "discrete"
        cpd.CPT_ = values.reshape(shape)
        cpd.state_names_ = {name: states[name] for name in [variable, *cpd.evidence_]}
        cpd._marginal = None if cpd.evidence_ else cpd.CPT_[:, 0]
        cpd._is_fitted = True
        return cpd

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        # Step 1: Find the states of every variable: the given ones, else the sorted states in the data.
        given = _checked_state_names(self.state_names)
        data = (pd.concat([y, X], axis=1) if self.evidence_ else y).reset_index(drop=True)
        state_names = build_state_names(data, given)

        # Step 2: Count, with any weights, the rows of each state of the target for each combination of parent states.
        if sample_weight is not None:
            sample_weight = sample_weight / sample_weight.max()
        counts = get_state_counts(data, state_names, self.variable_, self.evidence_, sample_weight)
        counts = counts.to_numpy(dtype=float, copy=True)
        if not np.isclose(counts.sum(), len(data) if sample_weight is None else sample_weight.sum()):
            raise ValueError(
                "Some values in the data don't match state_names, e.g. [0, 1] given for boolean data; list the states "
                "with the data's own types."
            )
        # Step 3: Normalize the counts into the marginal, and into the table, where parent combinations without data get
        # a uniform distribution.
        self._marginal = counts.sum(axis=1) / counts.sum()
        counts[:, (counts == 0).all(axis=0)] = 1.0
        self.CPT_ = counts / counts.sum(axis=0)
        self.state_names_ = state_names

    def _predict_proba(self, X: pd.DataFrame | None) -> NominalDistribution:
        # Step 1: Without X, return the marginal, known after fit and for a root.
        if X is None:
            if self._marginal is None:
                raise ValueError(
                    "A TabularCPD with parents created by from_values has no data on its parents, so its marginal "
                    "distribution is unknown; pass X."
                )
            return NominalDistribution(probs=self._marginal.copy(), categories=list(self.state_names_[self.variable_]))

        # Step 2: Find each row's column of the table from its parent states, rejecting states not seen in fit.
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

        # Step 3: Give each row the target's distribution in its column.
        return NominalDistribution(
            probs=self.CPT_.T.take(columns, axis=0),
            categories=list(self.state_names_[self.variable_]),
            index=X.index,
            columns=[self.variable_],
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self) or not (self.is_fitted and other.is_fitted):
            return super().__eq__(other)
        names = [self.variable_, *self.evidence_]
        if (self.variable_, self.evidence_) != (other.variable_, other.evidence_) or any(
            set(self.state_names_[name]) != set(other.state_names_[name]) for name in names
        ):
            return False
        # Put other's table in this CPD's state order before comparing, as DiscreteFactor.__eq__ does.
        positions = [[other.state_names_[name].index(state) for state in self.state_names_[name]] for name in names]
        cardinalities = [len(self.state_names_[name]) for name in names]
        return np.allclose(self.CPT_, other.CPT_.reshape(cardinalities)[np.ix_(*positions)].reshape(self.CPT_.shape))

    __hash__ = BaseParameter.__hash__


def _checked_state_names(state_names: dict | None) -> dict:
    """Return the given state names as lists, rejecting anything but a dict and repeated states."""
    if state_names is None:
        return {}
    if not isinstance(state_names, dict):
        raise TypeError(f"state_names must be a dict of {{variable: [states]}}, but is a {type(state_names).__name__}.")
    state_names = {variable: list(states) for variable, states in state_names.items()}
    for variable, states in state_names.items():
        if len(set(states)) != len(states):
            raise ValueError(f"Repeated state names for variable {variable!r}: {states}.")
    return state_names
