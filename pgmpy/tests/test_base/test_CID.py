import numpy as np
import pandas as pd
import pytest

from pgmpy.base import CausalInfluenceDiagram
from pgmpy.parameterization._base import BaseParameter


class FixedParameter(BaseParameter):
    """Small fitted parameterization used to test CID's BaseParameter integration."""

    _tags = {"variable_type": "discrete", "parent_type": "discrete"}

    def __init__(self, variable, evidence, states, probabilities):
        self.variable = variable
        self.evidence = evidence
        self.states = states
        self.probabilities = probabilities
        super().__init__()
        self.variable_ = variable
        self.evidence_ = sorted(evidence)
        self.variable_type_ = "discrete"
        self._is_fitted = True

    def _fit(self, X, y, sample_weight):
        raise NotImplementedError

    def _predict_proba(self, X):
        raise NotImplementedError

    def _sample(self, X, n_samples, random_state):
        rng = np.random.default_rng(random_state)
        if X is None:
            index = pd.RangeIndex(n_samples)
            parent_values = [()] * n_samples
        else:
            index = X.index
            parent_values = X.itertuples(index=False, name=None)

        values = [
            rng.choice(self.states, p=self.probabilities[tuple(parent_assignment)])
            for parent_assignment in parent_values
        ]
        return pd.DataFrame({self.variable_: values}, index=index)


def parameter(variable, states, probabilities, evidence=None):
    return FixedParameter(variable, [] if evidence is None else evidence, states, probabilities)


def test_roles_default_to_chance_and_can_be_declared():
    cid = CausalInfluenceDiagram(
        ebunch=[("Treatment", "Health"), ("Treatment", "Utility"), ("Health", "Utility")],
        decision_nodes=["Treatment"],
        utility_nodes=["Utility"],
    )

    assert cid.chance_nodes == ["Health"]
    assert cid.decision_nodes == ["Treatment"]
    assert cid.utility_nodes == ["Utility"]

    cid.add_node("Age")
    cid.add_node("Risk", role="chance")
    assert cid.chance_nodes == ["Health", "Age", "Risk"]

    cid_b = CausalInfluenceDiagram()
    cid_b.add_edge("A", "B")

    assert cid_b.chance_nodes == ["A", "B"]
    assert cid_b.decision_nodes == []
    assert cid_b.utility_nodes == []


def test_bulk_node_updates_preserve_roles_and_clear_removed_assignments():
    cid = CausalInfluenceDiagram()
    cid.add_nodes_from(["X", ("Decision", {"role": "decision"})])

    assert cid.chance_nodes == ["X"]
    assert cid.decision_nodes == ["Decision"]

    cid.set_probabilities("X", parameter("X", [0, 1], {(): [0.5, 0.5]}))
    cid.remove_nodes_from(["Decision", "X"])
    cid.add_node("X")

    with pytest.raises(ValueError, match="No probability parameterization assigned"):
        cid.check_model()


def test_role_validation_and_utility_leaf_invariant():
    with pytest.raises(ValueError, match="more than one role"):
        CausalInfluenceDiagram(decision_nodes=["A"], utility_nodes=["A"])

    with pytest.raises(ValueError, match="role must be one of"):
        CausalInfluenceDiagram().add_node("A", role="observation")

    cid = CausalInfluenceDiagram([("Utility", "Outcome")], utility_nodes=["Utility"])
    with pytest.raises(ValueError, match="Utility nodes must be leaves"):
        cid.check_model()

    cid = CausalInfluenceDiagram(chance_nodes=["X"])


def test_acyclicity():
    with pytest.raises(ValueError, match="Cycles are not allowed"):
        CausalInfluenceDiagram([("A", "B"), ("B", "B")])


def test_copy_retains_assignments_and_parent_changes_are_revalidated():
    cid = CausalInfluenceDiagram([("X", "Y")])
    cid.set_probabilities("X", parameter("X", [0, 1], {(): [0.5, 0.5]}))
    cid.set_probabilities("Y", parameter("Y", [0, 1], {(0,): [0.5, 0.5], (1,): [0.5, 0.5]}, ["X"]))

    copied = cid.copy()
    pd.testing.assert_frame_equal(cid.simulate(n_samples=5, seed=1), copied.simulate(n_samples=5, seed=1))

    cid.add_node("Z")
    cid.add_edge("Z", "Y")
    cid.set_probabilities("Z", parameter("Z", [0, 1], {(): [0.5, 0.5]}))
    for validate in (cid.check_model, lambda: cid.simulate(n_samples=5, seed=1)):
        with pytest.raises(ValueError, match="evidence .* does not match the parents of node 'Y'"):
            validate()


def test_parameter_assignments_validate_role_target_and_parents():
    cid = CausalInfluenceDiagram([("X", "Y"), ("Y", "Utility")], utility_nodes=["Utility"])
    root = parameter("X", ["no", "yes"], {(): [0.5, 0.5]})
    child = parameter("Y", ["no", "yes"], {("no",): [0.5, 0.5], ("yes",): [0.5, 0.5]}, ["X"])
    wrong_parents = parameter("Y", ["no", "yes"], {(): [0.5, 0.5]})

    with pytest.raises(ValueError, match="expected a chance node"):
        cid.set_probabilities("Utility", root)
    with pytest.raises(ValueError, match="expected a decision node"):
        cid.set_policy("X", root)
    with pytest.raises(ValueError, match="expected a utility node"):
        cid.set_utility("X", lambda values: np.zeros(len(values)))

    with pytest.raises(TypeError, match="Expected pgmpy BaseParameter"):
        cid.set_probabilities("X", object())
    with pytest.raises(TypeError, match="Expected pgmpy BaseParameter"):
        CausalInfluenceDiagram(decision_nodes=["Decision"]).set_policy("Decision", object())
    with pytest.raises(ValueError, match="targets node 'Y'"):
        cid.set_probabilities("X", child)
    with pytest.raises(ValueError, match="evidence .* does not match the parents of node 'Y'"):
        cid.set_probabilities("Y", wrong_parents)

    cid.set_probabilities("X", root)
    cid.set_probabilities("Y", child)


def test_simulate_distinguishes_missing_and_unfitted_parameterizations():
    cid = CausalInfluenceDiagram(chance_nodes=["X"])

    for validate in (cid.check_model, lambda: cid.simulate(n_samples=1)):
        with pytest.raises(ValueError, match="No probability parameterization assigned"):
            validate()

    unfitted = BaseParameter()
    cid.set_probabilities("X", unfitted)
    for validate in (cid.check_model, lambda: cid.simulate(n_samples=1)):
        with pytest.raises(ValueError, match="The probability parameterization assigned to node 'X' is not fitted"):
            validate()


def test_check_model_validates_roles_and_utility_assignments():
    cid = CausalInfluenceDiagram([("X", "Utility")], utility_nodes=["Utility"])
    cid.set_probabilities("X", parameter("X", [0, 1], {(): [0.5, 0.5]}))

    with pytest.raises(ValueError, match="No utility function assigned"):
        cid.check_model()

    cid.set_utility("Utility", lambda values: np.zeros(len(values)))
    assert cid.check_model() is True

    cid.nodes["X"]["role"] = "invalid"
    with pytest.raises(ValueError, match="invalid role"):
        cid.check_model()


def test_simulate_samples_policies_and_additive_utility_with_seed():
    cid = CausalInfluenceDiagram(
        ebunch=[
            ("Weather", "Treatment"),
            ("Weather", "Health"),
            ("Treatment", "Health"),
            ("Health", "HealthUtility"),
            ("Treatment", "TreatmentUtility"),
        ],
        decision_nodes=["Treatment"],
        utility_nodes=["HealthUtility", "TreatmentUtility"],
    )
    cid.set_probabilities(
        "Weather",
        parameter("Weather", ["sun", "rain"], {(): [0.5, 0.5]}),
    )
    cid.set_policy(
        "Treatment",
        parameter(
            "Treatment",
            ["control", "drug"],
            {("rain",): [1.0, 0.0], ("sun",): [0.0, 1.0]},
            ["Weather"],
        ),
    )
    cid.set_probabilities(
        "Health",
        parameter(
            "Health",
            ["well", "ill"],
            {
                ("control", "rain"): [0.6, 0.4],
                ("control", "sun"): [0.9, 0.1],
                ("drug", "rain"): [0.7, 0.3],
                ("drug", "sun"): [0.95, 0.05],
            },
            ["Treatment", "Weather"],
        ),
    )
    cid.set_utility(
        "HealthUtility",
        lambda values: pd.Series(np.where(values["Health"] == "well", 10.0, 0.0), index=values.index),
    )
    cid.set_utility("TreatmentUtility", lambda values: -2.0 * (values["Treatment"] == "drug"))

    samples = cid.simulate(n_samples=100, seed=12)
    repeated = cid.simulate(n_samples=100, seed=12)

    pd.testing.assert_frame_equal(samples, repeated)
    assert set(samples["Treatment"]) == {"control", "drug"}
    assert set(samples["HealthUtility"]) <= {0.0, 10.0}
    assert set(samples["TreatmentUtility"]) <= {0.0, -2.0}
    assert samples[["HealthUtility", "TreatmentUtility"]].sum(axis=1).notna().all()


@pytest.mark.parametrize(
    ("utility", "message"),
    [
        (lambda values: np.full(len(values), np.nan), "finite"),
        (lambda values: np.zeros(len(values) + 1), "one value per row"),
        (lambda values: np.full((len(values), 2), 1.0), "one-dimensional"),
        (lambda values: ["not numeric"] * len(values), "numeric"),
    ],
)
def test_simulate_validates_utility_outputs(utility, message):
    cid = CausalInfluenceDiagram(utility_nodes=["Utility"])
    cid.set_utility("Utility", utility)

    with pytest.raises(ValueError, match=message):
        cid.simulate(n_samples=2)


def test_tabular_cpd_can_be_assigned_when_skpro_is_available():
    pytest.importorskip("skpro")
    from pgmpy.parameterization import TabularCPD

    cid = CausalInfluenceDiagram(chance_nodes=["X"])
    cid.set_probabilities("X", TabularCPD.from_values("X", 2, [[0.5], [0.5]]))

    assert len(cid.simulate(n_samples=5, seed=0)) == 5
