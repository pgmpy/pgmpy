import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.factors.discrete import TabularCPD as LegacyTabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.parameterization import TabularBayesian, TabularCPD, TabularMLE

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)

# Counts of a binary target for two parent combinations, the second without data.
COUNTS = np.array([[3.0, 0.0], [1.0, 0.0]])


def test_tags():
    # Local estimators of a TabularCPD, which pass them weighted counts.
    for cls, name in ((TabularMLE, "tabular_mle"), (TabularBayesian, "tabular_bayesian")):
        assert (cls.get_class_tag("object_type"), cls.get_class_tag("name")) == ("local_estimator", name)
        assert cls.get_class_tag("parameterization") == ["tabular_cpd"]
        assert cls.get_class_tag("supports_weighted_data") is True


class TestTabularMLE:
    def test_estimate(self):
        # Each column divided by its total; a column without data gets a uniform distribution.
        np.testing.assert_allclose(TabularMLE().estimate(COUNTS), [[0.75, 0.5], [0.25, 0.5]])


class TestTabularBayesian:
    def test_priors(self):
        # Each prior adds pseudo-counts to every cell before normalizing: 1 for K2, the equivalent sample size spread
        # over the 4 cells for BDeu, and the given ones for a Dirichlet prior.
        cases = [
            (TabularBayesian(prior_type="K2"), [[4 / 6, 0.5], [2 / 6, 0.5]]),
            (TabularBayesian(prior_type="BDeu", equivalent_sample_size=8), [[5 / 8, 0.5], [3 / 8, 0.5]]),
            (TabularBayesian(prior_type="bdeu", equivalent_sample_size=8), [[5 / 8, 0.5], [3 / 8, 0.5]]),
            (TabularBayesian(prior_type="dirichlet", pseudo_counts=0.5), [[3.5 / 5, 0.5], [1.5 / 5, 0.5]]),
            (TabularBayesian(prior_type="dirichlet", pseudo_counts=[[1, 3], [0, 1]]), [[4 / 5, 0.75], [1 / 5, 0.25]]),
        ]
        for estimator, expected in cases:
            np.testing.assert_allclose(estimator.estimate(COUNTS), expected)

    def test_errors(self):
        with pytest.raises(ValueError, match="prior_type must be"):
            TabularBayesian(prior_type="uniform")
        with pytest.raises(ValueError, match="needs pseudo_counts"):
            TabularBayesian(prior_type="dirichlet")
        with pytest.raises(ValueError, match="only for prior_type='dirichlet'"):
            TabularBayesian(prior_type="BDeu", pseudo_counts=1.0)
        for size in (0, -1, np.inf):
            with pytest.raises(ValueError, match="equivalent_sample_size must be"):
                TabularBayesian(equivalent_sample_size=size)
        for pseudo_counts in (-1.0, [[1.0, -1.0]], [1.0, 2.0]):
            with pytest.raises(ValueError, match="pseudo_counts must be"):
                TabularBayesian(prior_type="dirichlet", pseudo_counts=pseudo_counts)
        with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
            TabularBayesian(prior_type="dirichlet", pseudo_counts=[[1.0, 1.0]]).estimate(COUNTS)


class TestTabularCPDEstimator:
    def test_matches_discrete_bayesian_estimator(self):
        # A random table over three parents shows any column mix-up. Rows with (x1, x2) = (2, 1) are dropped so that
        # combination is never seen, and zero weights on (x1, x3) = (0, 1) leave another combination without data.
        rng = np.random.default_rng(0)
        model = DiscreteBayesianNetwork([("x1", "y"), ("x2", "y"), ("x3", "y")])
        model.add_cpds(
            LegacyTabularCPD("x1", 3, [[0.3], [0.3], [0.4]]),
            LegacyTabularCPD("x2", 2, [[0.5], [0.5]]),
            LegacyTabularCPD("x3", 2, [[0.6], [0.4]]),
            LegacyTabularCPD("y", 3, rng.dirichlet([1, 1, 1], size=12).T, ["x1", "x2", "x3"], [3, 2, 2]),
        )
        data = model.simulate(500, seed=0, show_progress=False)
        data = data[~((data["x1"] == 2) & (data["x2"] == 1))]
        weights = np.where((data["x1"] == 0) & (data["x3"] == 1), 0.0, rng.uniform(0.5, 2, len(data)))

        priors = [
            {},
            {"prior_type": "BDeu", "equivalent_sample_size": 7},
            {"prior_type": "K2"},
            {"prior_type": "dirichlet", "pseudo_counts": 0.3},
        ]
        for prior in priors:
            for sample_weight in (None, weights):
                fitted = DiscreteBayesianEstimator(**prior).fit(model, data, sample_weight).parameters_
                expected = {cpd.variable: cpd for cpd in fitted}
                for node in model.nodes:
                    parents = sorted(model.get_parents(node))
                    cpd = TabularCPD(estimator=TabularBayesian(**prior)).fit(
                        data[parents] if parents else None, data[[node]], sample_weight=sample_weight
                    )
                    np.testing.assert_allclose(cpd.cpt_, expected[node].get_values())

    def test_weights_count_rows_under_a_prior(self):
        # A weight of 3 counts a row three times against the prior, as three copies of it do: (3 + 1) / 7 with K2.
        y = pd.Series(["a", "b", "b"], name="y")
        estimator = TabularBayesian(prior_type="K2")
        weighted = TabularCPD(estimator=estimator).fit(None, y, sample_weight=[3.0, 1.0, 1.0])
        np.testing.assert_allclose(weighted.cpt_, [[4 / 7], [3 / 7]])
        copies = TabularCPD(estimator=estimator).fit(None, pd.Series(["a", "a", "a", "b", "b"], name="y"))
        np.testing.assert_allclose(weighted.cpt_, copies.cpt_)

    def test_weights_follow_the_estimator(self):
        # A TabularCPD takes sample weights only if its estimator does.
        class UnweightedMLE(TabularMLE):
            _tags = {"name": "unweighted_mle", "supports_weighted_data": False}

        cpd = TabularCPD(estimator=UnweightedMLE())
        assert cpd.get_tag("supports_weighted_data") is False
        with pytest.raises(ValueError, match="does not support sample_weight"):
            cpd.fit(None, pd.Series(["a", "b"], name="y"), sample_weight=[1.0, 2.0])
        assert TabularCPD(estimator=TabularBayesian()).get_tag("supports_weighted_data") is True

    def test_estimator_parameter(self):
        # Maximum likelihood is the default, and the estimator must be a tabular one, not a network-level estimator.
        X = pd.DataFrame({"x": ["p", "p", "q"]})
        y = pd.Series(["a", "b", "a"], name="y")
        assert TabularCPD().fit(X, y) == TabularCPD(estimator=TabularMLE()).fit(X, y)
        with pytest.raises(TypeError, match="tabular estimator"):
            TabularCPD(estimator=DiscreteBayesianEstimator())
