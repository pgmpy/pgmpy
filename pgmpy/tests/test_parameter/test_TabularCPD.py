import numpy as np
import pandas as pd
import pytest

from pgmpy.parameter.TabularCPD import TabularCPD


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

    # Case 1: root node with MLE case
    # evidences : None
    # variable: {y: (0, 1)}
    +---------+-------------+
    | y       | probability |
    +---------+-------------+
    | y = 0   |    0.56     |
    +---------+-------------+
    | y = 1   |    0.44     |
    +---------+-------------+

    # Case 2: root node with MLE, sample_weight case
    # evidences : None
    # variable: {y: (0, 1)}
    # sample_weight
    # # if y == 0: sample_weight = 0.44
    # # if y == 1: sample_weight = 0.56
    +---------+-------------+
    | y       | probability |
    +---------+-------------+
    | y = 0   |    0.50     |
    +---------+-------------+
    | y = 1   |    0.50     |
    +---------+-------------+

    # Case 3: not root node with MLE case
    # evidences: {x1: (0, 1, 2), x2: (0, 1)}
    # variable: {y: (0, 1)}
    +---------+-----------------------+-----------------------+-----------------------+
    | x1      |           0           |           1           |           2           |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | x2      |     0     |     1     |     0     |     1     |     0     |     1     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | y = 0   |   0.5     | 0.461538  | 0.52381   |   0.75    |  0.6875   |   0.5     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | y = 1   |   0.5     | 0.538462  | 0.47619   |   0.25    |  0.3125   |   0.5     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+

    # Case 4: not root node with MLE, sample_weight case
    # evidences: {x1: (0, 1, 2), x2: (0, 1)}
    # variable: {y: (0, 1)}
    # sample_weight
    #
    # if x1 == 0 and x2 == 0:
    #     if y == 0: sample_weight = 0.5
    #     if y == 1: sample_weight = 0.5
    #
    # if x1 == 0 and x2 == 1:
    #     if y == 0: sample_weight = 0.538462
    #     if y == 1: sample_weight = 0.461538
    #
    # if x1 == 1 and x2 == 0:
    #     if y == 0: sample_weight = 0.476190
    #     if y == 1: sample_weight = 0.523810
    #
    # if x1 == 1 and x2 == 1:
    #     if y == 0: sample_weight = 0.25
    #     if y == 1: sample_weight = 0.75
    #
    # if x1 == 2 and x2 == 0:
    #     if y == 0: sample_weight = 0.3125
    #     if y == 1: sample_weight = 0.6875
    #
    # if x1 == 2 and x2 == 1:
    #     if y == 0: sample_weight = 0.5
    #     if y == 1: sample_weight = 0.5
    +---------+-----------------------+-----------------------+-----------------------+
    | x1      |           0           |           1           |           2           |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | x2      |     0     |     1     |     0     |     1     |     0     |     1     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | y = 0   |    0.5    |    0.5    |    0.5    |    0.5    |    0.5    |    0.5    |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | y = 1   |    0.5    |    0.5    |    0.5    |    0.5    |    0.5    |    0.5    |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    """
    rng = np.random.default_rng(seed=42)
    n_samples = 100

    X = pd.DataFrame(
        {
            "x1": rng.integers(0, 3, size=n_samples),  # {0, 1, 2}
            "x2": rng.integers(0, 2, size=n_samples),  # {0, 1}
        }
    ).astype(str)

    y = pd.DataFrame({"y": rng.integers(0, 2, size=n_samples)}).astype(str)

    return X, y


class TestTabularCPD:
    """Test TabularCPD class"""

    def test_base_parameter_default(self):
        parameter = TabularCPD()

        assert parameter.__class__.__name__ == "TabularCPD"
        assert parameter.get_class_tag("object_type") == {"discrete"}
        assert parameter.get_class_tag("fit_mode") == {"supervise", "unsupervise"}
        assert parameter.get_class_tag("python_dependencies") == set()
        assert parameter.get_class_tag("local:plug_in") == {"mle"}
        assert parameter.get_class_tag("global:plug_in") == set()
        assert parameter.get_class_tag("local:full_bayesian") == set()
        assert parameter.get_class_tag("global:full_bayesian") == set()

    def test_fit(self, discrete_data):
        # Case 1: root node with MLE case
        _, y = discrete_data
        parameter = TabularCPD()
        parameter.fit(y)

        assert parameter.is_fitted is True
        assert parameter._y_transformer.__class__.__name__ == "LabelBinarizer"
        np.testing.assert_allclose(
            parameter.CPT_,
            np.array(
                [
                    [0.56],
                    [0.44],
                ]
            ),
        )
        np.testing.assert_array_equal(parameter.categories_["y"], np.array(["0", "1"]))
        assert next(iter(parameter.categories_)) == "y"

        # Case 2: root node with MLE, sample_weight case
        _, y = discrete_data
        sample_weight = np.where(
            y["y"].to_numpy() == "0",
            0.44,
            0.56,
        )
        parameter = TabularCPD()
        parameter.fit(y, sample_weight=sample_weight)

        assert parameter.is_fitted is True
        assert parameter._y_transformer.__class__.__name__ == "LabelBinarizer"
        np.testing.assert_allclose(
            parameter.CPT_,
            np.array(
                [
                    [0.50],
                    [0.50],
                ]
            ),
        )
        np.testing.assert_array_equal(parameter.categories_["y"], np.array(["0", "1"]))
        assert next(iter(parameter.categories_)) == "y"

        # Case 3: not root node with MLE case
        X, y = discrete_data
        parameter = TabularCPD()

        parameter.fit(X, y)

        expected_CPT = np.array(
            [
                [
                    7 / 14,
                    6 / 13,
                    11 / 21,
                    9 / 12,
                    11 / 16,
                    12 / 24,
                ],
                [
                    7 / 14,
                    7 / 13,
                    10 / 21,
                    3 / 12,
                    5 / 16,
                    12 / 24,
                ],
            ]
        )

        assert parameter.is_fitted is True
        assert parameter._y_transformer.__class__.__name__ == "LabelBinarizer"
        np.testing.assert_allclose(
            parameter.CPT_,
            expected_CPT,
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_array_equal(parameter.categories_["y"], np.array(["0", "1"]))
        assert next(iter(parameter.categories_)) == "y"

        # Case 4: not root node with MLE, sample_weight case
        X, y = discrete_data
        sample_weight_map = {
            (0, 0, 0): 1 / 2,
            (0, 0, 1): 1 / 2,
            (0, 1, 0): 7 / 13,
            (0, 1, 1): 6 / 13,
            (1, 0, 0): 10 / 21,
            (1, 0, 1): 11 / 21,
            (1, 1, 0): 1 / 4,
            (1, 1, 1): 3 / 4,
            (2, 0, 0): 5 / 16,
            (2, 0, 1): 11 / 16,
            (2, 1, 0): 1 / 2,
            (2, 1, 1): 1 / 2,
        }

        sample_weight = np.array(
            [
                sample_weight_map[(x1, x2, target)]
                for x1, x2, target in zip(
                    X["x1"].to_numpy().astype(int),
                    X["x2"].to_numpy().astype(int),
                    y["y"].to_numpy().astype(int),
                )
            ],
            dtype=float,
        )
        parameter = TabularCPD()

        parameter.fit(X, y, sample_weight)

        expected_CPT = np.array(
            [
                [
                    1 / 2,
                    1 / 2,
                    1 / 2,
                    1 / 2,
                    1 / 2,
                    1 / 2,
                ],
                [
                    1 / 2,
                    1 / 2,
                    1 / 2,
                    1 / 2,
                    1 / 2,
                    1 / 2,
                ],
            ]
        )

        assert parameter.is_fitted is True
        assert parameter._y_transformer.__class__.__name__ == "LabelBinarizer"
        np.testing.assert_allclose(
            parameter.CPT_,
            expected_CPT,
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_array_equal(parameter.categories_["y"], np.array(["0", "1"]))
        assert next(iter(parameter.categories_)) == "y"

    def test_predict_proba(self, discrete_data):
        X, y = discrete_data

        # Case １: root node case
        parameter = TabularCPD()
        parameter.fit(y)

        with pytest.raises(NotImplementedError):
            parameter.predict_proba(y)

        # Case ２: non-root node case
        parameter = TabularCPD()
        parameter.fit(X, y)
        dist = parameter.predict_proba(X)

        expected_probs = np.array(
            [
                [6 / 13, 7 / 13],
                [11 / 16, 5 / 16],
                [3 / 4, 1 / 4],
                [11 / 21, 10 / 21],
                [3 / 4, 1 / 4],
            ]
        )

        np.testing.assert_array_equal(dist.probs[:5], expected_probs)
        assert dist.__class__.__name__ == "NominalDistribution"
        assert list(dist.categories) == ["0", "1"]
        assert list(dist.columns) == ["y"]

        with pytest.raises(RuntimeError):
            parameter = TabularCPD()
            parameter.predict_proba(X)

    def test_sample(self, discrete_data):
        X, y = discrete_data

        parameter = TabularCPD()
        parameter.fit(y)

        samples, log_pmf = parameter.sample(n_samples=5)

        assert isinstance(samples, np.ndarray)
        assert samples.shape == (5, 1)
        assert isinstance(log_pmf, np.ndarray)
        assert log_pmf.shape == (5, 1)

        # Case ２: non-root node case
        parameter = TabularCPD()
        parameter.fit(X, y)
        samples, log_pmf = parameter.sample(X[:5])

        assert isinstance(samples, np.ndarray)
        assert samples.shape == (5, 1)
        assert isinstance(log_pmf, np.ndarray)
        assert log_pmf.shape == (5, 1)

        with pytest.raises(RuntimeError):
            parameter = TabularCPD()
            parameter.sample(X)

    def test_set_fitted_params(self):
        parameter = TabularCPD()

        assert hasattr(parameter, "CPT_") is False
        assert hasattr(parameter, "categories_") is False
        assert hasattr(parameter, "evidences_") is False
        assert parameter.is_fitted is False

        CPT = np.array(
            [
                [0.2, 0.3],
                [0.8, 0.7],
            ]
        )
        columns = ["grade"]
        categories = ["P", "F"]
        evidences = {"x1": np.array([0, 1, 2]), "x2": np.array([0, 1])}

        parameter.set_fitted_params(
            CPT=CPT,
            columns=columns,
            categories=categories,
            evidences=evidences,
            is_fitted=True,
        )

        assert hasattr(parameter, "CPT_")
        assert hasattr(parameter, "categories_")
        assert hasattr(parameter, "evidences_")
        assert parameter.is_fitted is True
