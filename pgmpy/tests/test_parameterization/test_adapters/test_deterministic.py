import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.parameterization import DeterministicAdapter

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)


def bmi(X):
    return X["weight"] / X["height"] ** 2


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"weight": rng.uniform(50, 100, size=20), "height": rng.uniform(1.5, 2.0, size=20)},
        index=[f"r{i}" for i in range(20)],
    )
    return X, bmi(X).rename("bmi")


class TestDeterministicAdapter:
    def test_tags(self):
        # The data is ignored, so none is needed and weights change nothing.
        assert DeterministicAdapter.get_class_tag("variable_type") == ["continuous"]
        assert DeterministicAdapter.get_class_tag("requires_data") is False
        assert DeterministicAdapter.get_class_tag("supports_weighted_data") is True
        assert DeterministicAdapter.get_class_tag("capability:distribution") is True
        with pytest.raises(TypeError, match="callable"):
            DeterministicAdapter("not a function")

    def test_predictions(self, data):
        X, y = data
        node = DeterministicAdapter.from_values("bmi", bmi, evidence=["weight", "height"])
        assert node.is_fitted and (node.variable_, node.evidence_) == ("bmi", ["height", "weight"])

        # Each row gets the function's value as a Delta with X's index, so predict and sample give the values.
        dist = node.predict_proba(X)
        assert type(dist).__name__ == "Delta"
        assert (dist.index.tolist(), dist.columns.tolist()) == (X.index.tolist(), ["bmi"])
        np.testing.assert_allclose(dist.mean()["bmi"], y)
        pd.testing.assert_frame_equal(node.predict(X), y.to_frame())
        np.testing.assert_allclose(node.sample(X, n_samples=3, random_state=0)["bmi"], np.tile(y, 3))
        assert node.sample(X, n_samples=0).shape == (0, 1)

        # A root's function gets rows without columns; without X, the root gives its Delta as a scalar.
        root = DeterministicAdapter.from_values("c", lambda X: np.full(len(X), 5.0))
        assert root.predict_proba().mean() == 5.0
        assert root.sample(n_samples=4, random_state=0)["c"].tolist() == [5.0] * 4

    def test_fit(self, data):
        # fit ignores the values, and says so: it only records the target and its parents, so it needs no rows.
        X, y = data
        with pytest.warns(UserWarning, match="ignores the data"):
            node = DeterministicAdapter(bmi).fit(X, y + 100)
        assert (node.variable_, node.evidence_) == ("bmi", ["height", "weight"])
        np.testing.assert_allclose(node.predict(X)["bmi"], y)
        with pytest.warns(UserWarning, match="ignores the data"):
            assert DeterministicAdapter(bmi).fit(X.iloc[:0], y.iloc[:0]).is_fitted

    def test_log_likelihood(self, data):
        # 0 where y matches f(x) within numpy.isclose's tolerance, and minus infinity elsewhere. The Delta from
        # predict_proba compares exactly.
        X, y = data
        node = DeterministicAdapter.from_values("bmi", bmi, evidence=["height", "weight"])
        close, even = y * (1 + 1e-9), np.arange(len(y)) % 2 == 0
        scores = node.log_likelihood(X, close.where(even, y + 0.1))
        assert scores.index.equals(X.index)
        np.testing.assert_array_equal(scores["bmi"], np.where(even, 0.0, -np.inf))
        assert (node.predict_proba(X).log_pmf(close.to_frame()) == -np.inf).all().all()

    def test_errors(self, data):
        # The function must give one finite value per row; the error names the positions of the rows that don't.
        X, _ = data
        for function, match in (
            (lambda X: np.ones(len(X) + 1), "one value per row"),
            (lambda X: np.where(X["weight"] > 75, np.nan, 1.0), r"rows at positions \[0, 4"),
        ):
            node = DeterministicAdapter.from_values("bmi", function, evidence=["height", "weight"])
            with pytest.raises(ValueError, match=match):
                node.predict(X)
