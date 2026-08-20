import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from skpro.distributions.normal import Normal

from pgmpy.distributions.nominal import NominalDistribution
from pgmpy.parameter.adapter.SklearnAdapter import SklearnAdapter


@pytest.fixture
def discrete_data():
    """
    # MLE result
    # evidences: {x1: (0, 1, 2), x2: (0, 1)}
    # variable: {y: (0, 1)}
    +---------+-----------------------+-----------------------+-----------------------+
    | x1      |           0           |           1           |           2           |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | x2      |     0     |     1     |     0     |     1     |     0     |     1     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | y = 0   |     7     |     6     |    11     |     9     |    11     |    12     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | y = 1   |     7     |     7     |    10     |     3     |     5     |    12     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    """
    rng = np.random.default_rng(seed=42)
    n_samples = 100

    X = pd.DataFrame(
        {
            "x1": rng.integers(0, 3, size=n_samples),  # {0, 1, 2}
            "x2": rng.integers(0, 2, size=n_samples),  # {0, 1}
        }
    )

    y = pd.DataFrame({"y": rng.integers(0, 2, size=n_samples)})

    return X, y


@pytest.fixture(scope="module")
def nonlinear_data():
    def _conditional_mean(X):
        """
        Y | A, B, C ~ Normal(
            1.25 + 2.50 sin(A) + 0.75 B**2 - 1.50 C,
            0.20**2
        )
        """
        return 1.25 + 2.50 * np.sin(X["A"]) + 0.75 * X["B"] ** 2 - 1.50 * X["C"]

    rng = np.random.default_rng(seed=42)

    X_train = pd.DataFrame(
        {
            "A": rng.uniform(-2.5, 2.5, size=1000),
            "B": rng.uniform(-2.5, 2.5, size=1000),
            "C": rng.uniform(-2.5, 2.5, size=1000),
        },
        index=pd.RangeIndex(1000),
    )

    X_test = pd.DataFrame(
        {
            "A": rng.uniform(-2.5, 2.5, size=200),
            "B": rng.uniform(-2.5, 2.5, size=200),
            "C": rng.uniform(-2.5, 2.5, size=200),
        },
        index=pd.RangeIndex(200),
    )

    y_train = pd.DataFrame(
        {
            "Y": (
                _conditional_mean(X_train)
                + rng.normal(
                    loc=0.0,
                    scale=0.20,
                    size=len(X_train),
                )
            )
        },
        index=X_train.index,
    )

    expected_test_mean = _conditional_mean(X_test)

    return X_train, y_train, X_test, expected_test_mean


class TestSklearnAdapter:
    def test_base_parameter_regressor(self):
        estimator = LinearRegression()
        parameter = SklearnAdapter(estimator=estimator)

        assert parameter.__class__.__name__ == "SklearnAdapter"
        assert parameter.estimator is estimator
        assert parameter._get_delegate() is estimator
        assert parameter.get_params(deep=False)["estimator"] is estimator

        assert parameter.get_tag("parameter_type") == "regressor"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("can_be_root") is False

    def test_base_parameter_classifier(self):
        estimator = LogisticRegression()
        parameter = SklearnAdapter(estimator=estimator)

        assert parameter.__class__.__name__ == "SklearnAdapter"
        assert parameter.estimator is estimator
        assert parameter._get_delegate() is estimator
        assert parameter.get_params(deep=False)["estimator"] is estimator

        assert parameter.get_tag("parameter_type") == "classifier"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("can_be_root") is False

    def test_fit_classifier(self, discrete_data):
        X, y = discrete_data
        estimator = LogisticRegression()
        parameter = SklearnAdapter(estimator=estimator)

        parameter.fit(X, y)

        assert parameter.is_fitted is True

    def test_fit_regressor(self, nonlinear_data):
        X, y, _, _ = nonlinear_data
        estimator = LinearRegression()
        parameter = SklearnAdapter(estimator=estimator)

        parameter.fit(X, y)

        assert parameter.is_fitted is True

    def test_predict_proba_classifier(self, discrete_data):
        X, y = discrete_data
        estimator = LogisticRegression()
        parameter = SklearnAdapter(estimator=estimator)

        parameter.fit(X, y)

        results = parameter.predict_proba(X)

        assert isinstance(results, NominalDistribution)
        assert results.index.equals(X.index)

    def test_predict_proba_regressor(self, nonlinear_data):
        X, y, _, _ = nonlinear_data
        estimator = LinearRegression()
        parameter = SklearnAdapter(estimator=estimator)

        parameter.fit(X, y)

        results = parameter.predict_proba(X)

        assert isinstance(results, Normal)
        assert results.index.equals(X.index)

    def test_fails(self):
        # Case 1: Not regression or classifier model
        from sklearn.cluster._bicluster import SpectralBiclustering

        estimator = SpectralBiclustering()

        with pytest.raises(TypeError):
            SklearnAdapter(estimator=estimator)

        # Case 2: Do not have `predict_proba()` method in classifier model
        from sklearn.linear_model import RidgeClassifier

        estimator = RidgeClassifier()

        with pytest.raises(TypeError):
            SklearnAdapter(estimator=estimator)
