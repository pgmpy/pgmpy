from collections.abc import Hashable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pgmpy.parameterization._base import (
    BaseParameterization,
    _checked_evidence,
    _checked_state_names,
    _parent_order,
)
from pgmpy.parameterization.cpds.tabular_estimators import BaseTabularEstimator, TabularMLE
from pgmpy.parameterization.distributions import NominalDistribution
from pgmpy.utils import collect_state_names, encode_columns, get_state_counts_array


class TabularCPD(BaseParameterization):
    """Tabular conditional probability distribution of a discrete variable given discrete parents.

    ``fit(X, y)`` counts the rows with each state of ``y`` for every combination of the parents' states in ``X``, with
    any sample weights, and ``estimator`` turns the counts into the table: by maximum likelihood by default, where
    parent combinations without data get a uniform distribution, or with a Dirichlet prior with ``TabularBayesian``.
    Parents are sorted by name. Fit a root variable with ``X=None``. ``from_values`` creates a fitted instance from a
    known table instead.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    state_names : dict, optional
        States of the target and of its parents, as ``{variable: [states]}``. The listed states must include every state
        in the data; variables that aren't listed get the sorted states seen in the data.
    estimator : BaseTabularEstimator, optional
        Turns the counts into the table, e.g. ``TabularBayesian(prior_type="BDeu")``. ``None`` for ``TabularMLE()``.

    Attributes
    ----------
    cpt_ : numpy.ndarray of shape (variable_card_, product of evidence_card_)
        Probability of each state of the target (rows) for each combination of parent states (columns). The combinations
        follow the product of the parents' states in ``evidence_`` order, with the last parent varying fastest. A root
        variable has a single column.
    state_names_ : dict
        States of the target and of each parent.
    variable_card_ : int
        Number of states of the target.
    evidence_card_ : list of int
        Number of states of each parent, in ``evidence_`` order; empty for a root.
    variable_ : hashable
        Name of the target variable.
    evidence_ : list
        Names of the parent variables, sorted.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.parameterization import TabularCPD
    >>> data = pd.DataFrame({"rain": ["no", "yes", "no", "no"], "wet": ["no", "yes", "yes", "no"]})
    >>> cpd = TabularCPD().fit(data[["rain"]], data["wet"])
    >>> cpd.cpt_.round(2).tolist()
    [[0.67, 0.0], [0.33, 1.0]]
    >>> cpd.variable_card_, cpd.evidence_card_
    (2, [2])
    >>> cpd.sample(pd.DataFrame({"rain": ["yes", "yes"]}), random_state=0)["wet"].tolist()
    ['yes', 'yes']
    >>> cpd.predict(pd.DataFrame({"rain": ["no", "yes"]}))["wet"].tolist()
    ['no', 'yes']
    >>> cpd.log_likelihood(data[["rain"]], data["wet"]).round(2)["wet"].tolist()
    [-0.41, 0.0, -1.1, -0.41]

    A root variable is fitted without parents, or created from known probabilities:

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

    def __init__(self, state_names: dict | None = None, estimator: BaseTabularEstimator | None = None) -> None:
        self.state_names = state_names
        self.estimator = estimator
        super().__init__()

        # The estimator decides whether the counts can be weighted.
        if estimator is not None:
            if not isinstance(estimator, BaseTabularEstimator):
                raise TypeError(
                    "estimator must be a tabular estimator, such as TabularMLE() or TabularBayesian(), but is a "
                    f"{type(estimator).__name__}."
                )
            self.set_tags(supports_weighted_data=estimator.get_tag("supports_weighted_data"))

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
        # Step 1: Check the parents and their cardinalities, sort the parents by name, and give the target and each
        # parent its states: the given ones, else 0, 1, ....
        evidence = _checked_evidence(variable, evidence)
        evidence_card = [] if evidence_card is None else list(evidence_card)
        if len(evidence_card) != len(evidence):
            raise ValueError(f"evidence_card must have one entry per parent in {evidence}, but is {evidence_card}.")
        order = _parent_order(evidence)
        names = [variable, *(evidence[position] for position in order)]
        cardinalities = [variable_card, *(evidence_card[position] for position in order)]
        given = _checked_state_names(state_names)
        states = {name: given.get(name, list(range(card))) for name, card in zip(names, cardinalities)}
        for name, card in zip(names, cardinalities):
            if len(states[name]) != card:
                raise ValueError(
                    f"{name!r} has {card} states, but state_names lists {len(states[name])}: {states[name]}."
                )

        # Step 2: Check that values is a table of probabilities of the right shape, and reorder its columns to the
        # sorted parents.
        values = np.array(values, dtype=float)
        shape = (variable_card, int(np.prod(evidence_card)))
        if values.shape != shape:
            raise ValueError(
                f"values must have shape {shape} for variable_card and evidence_card, but has {values.shape}."
            )
        if (values < 0).any() or not np.allclose(values.sum(axis=0), 1, atol=0.01):
            raise ValueError("values must be non-negative, and each column must sum to 1.")
        values = values.reshape(variable_card, *evidence_card).transpose(0, *(1 + position for position in order))

        # Step 3: Create the fitted instance.
        cpd = cls(state_names=state_names)
        cpd.variable_, *cpd.evidence_ = names
        cpd.variable_card_, *cpd.evidence_card_ = cardinalities
        cpd.state_names_ = states
        cpd.cpt_ = values.reshape(shape)
        cpd._is_fitted = True
        return cpd

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        # Step 1: Find the states of every variable: the given ones, else the sorted states in the data.
        names = [self.variable_, *self.evidence_]
        given = _checked_state_names(self.state_names)
        data = pd.concat([y, X], axis=1, ignore_index=True)
        states = {}
        for position, name in enumerate(names):
            observed = collect_state_names(data, position)
            if name in given and not set(observed) <= set(given[name]):
                raise ValueError(f"Data contains unexpected states for variable: {name!r}.")
            states[name] = given[name] if name in given else observed

        # Step 2: Count, with any weights, the rows of each state of the target for each combination of parent states.
        # The weights keep their scale, as it sets how much the data count against a prior. Positions label the
        # columns here, as selecting columns by label fails for some names, e.g. booleans.
        with np.errstate(over="ignore"):
            total = len(data) if sample_weight is None else sample_weight.sum()
        if not np.isfinite(total):
            raise ValueError("sample_weight values sum to more than a float can hold; divide them by a constant.")
        codes, cardinalities = encode_columns(data, dict(enumerate(states.values())))
        counts = get_state_counts_array(codes, cardinalities, 0, range(1, len(names)), sample_weight).astype(float)
        if not np.isclose(counts.sum(), total):
            raise ValueError(
                "Some values in the data don't match state_names, e.g. [0, 1] given for boolean data; list the states "
                "with the data's own types."
            )

        # Step 3: Turn the counts into the table.
        self.cpt_ = (TabularMLE() if self.estimator is None else self.estimator).estimate(counts)
        self.state_names_ = states
        self.variable_card_, *self.evidence_card_ = cardinalities.values()

    def _predict_proba(self, X: pd.DataFrame | None) -> NominalDistribution:
        # Step 1: Without X, return a root's distribution, the table's single column.
        if X is None:
            return NominalDistribution(probs=self.cpt_[:, 0].copy(), categories=list(self.state_names_[self.variable_]))

        # Step 2: Find each row's column of the table from its parent states, rejecting states not seen in fit.
        codes, _ = encode_columns(X, self.state_names_)
        columns = np.zeros(len(X), dtype=int)
        for parent, card in zip(self.evidence_, self.evidence_card_):
            unknown = codes[parent] < 0
            if unknown.any():
                raise ValueError(
                    f"X[{parent!r}] has values not seen in fit: {pd.unique(X[parent][unknown]).tolist()}. Known "
                    f"states: {self.state_names_[parent]}."
                )
            columns = columns * card + codes[parent]

        # Step 3: Give each row the target's distribution in its column.
        return NominalDistribution(
            probs=self.cpt_.T.take(columns, axis=0),
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
        # Put other's table in this CPD's state order before comparing.
        positions = [[other.state_names_[name].index(state) for state in self.state_names_[name]] for name in names]
        cardinalities = [self.variable_card_, *self.evidence_card_]
        return np.allclose(self.cpt_, other.cpt_.reshape(cardinalities)[np.ix_(*positions)].reshape(self.cpt_.shape))

    __hash__ = BaseParameterization.__hash__
