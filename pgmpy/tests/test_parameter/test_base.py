import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_moons

from pgmpy.parameter._base import BaseParameter


class TestBaseParameter:
    """Test BaseParameter class"""

    def test_base_parameter_default(self):
        parameter = BaseParameter()

        assert parameter.__class__.__name__ == "BaseParameter"
        assert parameter.get_class_tag("parameter_type") is str
        assert parameter.get_class_tag("produces_factor") is bool
        assert parameter.get_class_tag("missing") is bool
        assert parameter.get_class_tag("is_linear_gaussian") is bool
        assert parameter.get_class_tag("supports_fit_joint") is bool

    def test_fit(self):
        X_arr, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
        X = pd.DataFrame(X_arr[:, 0].reshape(-1, 1), columns=["x"])
        y = pd.DataFrame(X_arr[:, 1].reshape(-1, 1), columns=["y"])
        parameter = BaseParameter()

        with pytest.raises(NotImplementedError):
            parameter.fit(X, y)

    def test_predict_proba(self):
        X_arr, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
        X = pd.DataFrame(X_arr[:, 0].reshape(-1, 1), columns=["x"])
        parameter = BaseParameter()

        with pytest.raises(RuntimeError):
            parameter.predict_proba(X)

    def test_check_data(self):
        def make_parameter(missing=False, can_be_root=True):
            class TempParameter(BaseParameter):
                _tags = {
                    "parameter_type": "discrete",
                    "produces_factor": False,
                    "is_linear_gaussian": False,
                    "missing": missing,
                    "supports_fit_joint": False,
                    "can_be_root": can_be_root,
                    "python_dependencies": (),
                }

                def __init__(self, missing=False, can_be_root=True):
                    super().__init__()
                    self.set_tags(missing=missing, can_be_root=can_be_root)

            parameter = TempParameter(missing, can_be_root)
            return parameter

        # Case 1: valid unsupervised data
        parameter = make_parameter()
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

        out_X, out_y, out_weight = parameter._check_data(X)

        pd.testing.assert_frame_equal(out_X, X)
        assert out_y is None
        np.testing.assert_array_equal(out_weight, np.ones(3))

        # Case 2: X Series is converted to DataFrame
        parameter = make_parameter()
        X = pd.Series([1.0, 2.0, 3.0], name="x")

        out_X, out_y, out_weight = parameter._check_data(X)

        assert isinstance(out_X, pd.DataFrame)
        assert list(out_X.columns) == ["x"]
        assert out_y is None
        np.testing.assert_array_equal(out_weight, np.ones(3))

        # Case 3: X must be DataFrame
        parameter = make_parameter()

        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            parameter._check_data([[1.0], [2.0]])

        # Case 4: X must have rows
        parameter = make_parameter()
        X = pd.DataFrame({"x": []})

        with pytest.raises(ValueError, match="X must have at least one row"):
            parameter._check_data(X)

        # Case 5: X must have columns
        parameter = make_parameter()
        X = pd.DataFrame(index=[0, 1, 2])

        with pytest.raises(ValueError, match="X must have at least one column"):
            parameter._check_data(X)

        # Case 6: X missing values are rejected
        parameter = make_parameter(missing=False)
        X = pd.DataFrame({"x": [1.0, np.nan, 3.0]})

        with pytest.raises(ValueError, match="cannot deal with missing values"):
            parameter._check_data(X)

        # Case 7: X missing values are allowed
        parameter = make_parameter(missing=True)
        X = pd.DataFrame({"x": [1.0, np.nan, 3.0]})

        out_X, out_y, out_weight = parameter._check_data(X)

        pd.testing.assert_frame_equal(out_X, X)
        assert out_y is None
        np.testing.assert_array_equal(out_weight, np.ones(3))

        # Case 8: valid supervised data
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.DataFrame({"target": ["a", "b", "a"]})

        out_X, out_y, out_weight = parameter._check_data(X, y)

        pd.testing.assert_frame_equal(out_X, X)
        pd.testing.assert_frame_equal(out_y, y)
        np.testing.assert_array_equal(out_weight, np.ones(3))

        # Case 9: y Series is converted to DataFrame
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.Series(["a", "b", "a"], name="target")

        out_X, out_y, out_weight = parameter._check_data(X, y)

        assert isinstance(out_y, pd.DataFrame)
        assert list(out_y.columns) == ["target"]
        np.testing.assert_array_equal(out_weight, np.ones(3))

        # Case 10: y must be DataFrame
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})

        with pytest.raises(TypeError, match="y must be a pandas DataFrame"):
            parameter._check_data(X, ["a", "b", "a"])

        # Case 11: y must have rows
        parameter = make_parameter()
        X = pd.DataFrame({"x1": []})
        y = pd.DataFrame({"target": []})

        with pytest.raises(ValueError, match="X must have at least one row"):
            parameter._check_data(X, y)

        # Case 12: y must have columns
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.DataFrame(index=[0, 1, 2])

        with pytest.raises(ValueError, match="y must have at least one column"):
            parameter._check_data(X, y)

        # Case 13: y must have one target column
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.DataFrame(
            {
                "target1": ["a", "b", "a"],
                "target2": ["x", "y", "x"],
            }
        )

        with pytest.raises(ValueError, match="y must contain exactly one target column"):
            parameter._check_data(X, y)

        # Case 14: X and y length must match
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.DataFrame({"target": ["a", "b"]})

        with pytest.raises(ValueError, match="X and y must have the same number of rows"):
            parameter._check_data(X, y)

        # Case 15: X and y index must match
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        y = pd.DataFrame({"target": ["a", "b", "a"]}, index=[10, 11, 12])

        with pytest.raises(ValueError, match="X and y must have the same index"):
            parameter._check_data(X, y)

        # Case 16: valid sample weight
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.DataFrame({"target": ["a", "b", "a"]})
        sample_weight = [0.5, 1.0, 2.0]

        out_X, out_y, out_weight = parameter._check_data(X, y, sample_weight)

        np.testing.assert_array_equal(out_weight, np.array([0.5, 1.0, 2.0]))

        # Case 17: sample weight is flattened
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.DataFrame({"target": ["a", "b", "a"]})
        sample_weight = np.array([[0.5], [1.0], [2.0]])

        out_X, out_y, out_weight = parameter._check_data(X, y, sample_weight)

        np.testing.assert_array_equal(out_weight, np.array([0.5, 1.0, 2.0]))

        # Case 18: sample weight length must match
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.DataFrame({"target": ["a", "b", "a"]})

        with pytest.raises(ValueError, match="sample_weight must have length 3"):
            parameter._check_data(X, y, sample_weight=[1.0, 2.0])

        # Case 19: sample weight cannot be negative
        parameter = make_parameter()
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        y = pd.DataFrame({"target": ["a", "b", "a"]})

        with pytest.raises(ValueError, match="sample_weight cannot contain negative values"):
            parameter._check_data(X, y, sample_weight=[1.0, -1.0, 2.0])
