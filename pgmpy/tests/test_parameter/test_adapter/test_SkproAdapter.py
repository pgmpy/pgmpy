import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.preprocessing import SplineTransformer
from skpro.distributions.normal import Normal
from skpro.regression.gam import GAMRegressor
from skpro.regression.linear import GLMRegressor

from pgmpy.parameter.adapter.SkproAdapter import SkproAdapter


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


@pytest.fixture(scope="module")
def fitted_GLM_parameter(nonlinear_data):
    X_train, y_train, _, _ = nonlinear_data

    spline = SplineTransformer(n_knots=5, degree=3)
    X_train_spline = spline.fit_transform(X_train)

    parameter = SkproAdapter(
        estimator=GLMRegressor(
            maxiter=200,
            tol=1e-4,
        )
    )

    parameter.fit(X_train_spline, y_train)

    return parameter, spline


class TestSkproAdapter:
    def test_base_parameter_default(self):
        estimator = GLMRegressor()
        parameter = SkproAdapter(estimator=estimator)

        assert parameter.__class__.__name__ == "SkproAdapter"
        assert parameter.estimator is estimator
        assert parameter._get_delegate() is estimator
        assert parameter.get_params(deep=False)["estimator"] is estimator

        assert parameter.get_class_tag("parameter_type") == "regressor"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("can_be_root") is False

    def test_fit(
        self,
        nonlinear_data,
    ):
        X_train, y_train, _, _ = nonlinear_data
        spline = SplineTransformer(n_knots=5, degree=3)
        X_train_spline = spline.fit_transform(X_train)

        parameter = SkproAdapter(
            estimator=GLMRegressor(
                maxiter=200,
                tol=1e-4,
            )
        )

        parameter.fit(X_train_spline, y_train)

        assert parameter.is_fitted is True

    def test_predict_proba(
        self,
        nonlinear_data,
        fitted_GLM_parameter,
    ):
        _, _, X_test, _ = nonlinear_data
        parameter, spline = fitted_GLM_parameter
        X_test_spline = spline.transform(X_test)

        results = parameter.predict_proba(X_test_spline)

        assert isinstance(results, Normal)
        assert results.index.equals(X_test.index)

        pd.testing.assert_index_equal(
            results.mean().index,
            X_test.index,
        )
        pd.testing.assert_index_equal(
            results.mean().columns,
            pd.Index(["Y"]),
        )

    def test_predict_proba_recovers_nonlinear_mean_and_noise(
        self,
        nonlinear_data,
        fitted_GLM_parameter,
    ):
        _, _, X_test, expected_test_mean = nonlinear_data
        parameter, spline = fitted_GLM_parameter
        X_test_spline = spline.transform(X_test)

        pred = parameter.predict_proba(X_test_spline)
        pred_mean = pred.mean()["Y"].to_numpy()
        pred_var = pred.var()["Y"].to_numpy()
        expected = expected_test_mean.to_numpy()

        rmse = np.sqrt(np.mean((pred_mean - expected) ** 2))
        assert rmse < 0.30

        assert np.isfinite(pred_mean).all()
        assert np.isfinite(pred_var).all()
        assert (pred_var > 0).all()


@pytest.fixture(scope="module")
def fitted_GAM_parameter(nonlinear_data):
    if not _check_soft_dependencies("pygam", severity="none"):
        pytest.skip("execute only if required dependency present")

    X_train, y_train, _, _ = nonlinear_data

    parameter = SkproAdapter(
        estimator=GAMRegressor(
            max_iter=200,
            tol=1e-4,
        )
    )
    parameter.fit(X_train, y_train)

    return parameter


@pytest.mark.skipif(
    not _check_soft_dependencies("pygam", severity="none"),
    reason="execute only if required dependency present",
)
class TestSkproAdapterPygam:
    def test_base_parameter_default(self):
        estimator = GAMRegressor()
        parameter = SkproAdapter(estimator=estimator)

        assert parameter.__class__.__name__ == "SkproAdapter"
        assert parameter.estimator is estimator
        assert parameter._get_delegate() is estimator
        assert parameter.get_params(deep=False)["estimator"] is estimator

        assert parameter.get_class_tag("parameter_type") == "regressor"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("can_be_root") is False

    def test_fit(
        self,
        nonlinear_data,
    ):
        X_train, y_train, _, _ = nonlinear_data

        parameter = SkproAdapter(estimator=GAMRegressor(max_iter=200, tol=1e-4))
        parameter.fit(X_train, y_train)

        assert parameter.is_fitted is True

        np.testing.assert_array_equal(
            parameter.feature_names_in_,
            X_train.columns,
        )

    def test_predict_proba(
        self,
        nonlinear_data,
        fitted_GAM_parameter,
    ):
        _, _, X_test, _ = nonlinear_data
        parameter = fitted_GAM_parameter

        results = parameter.predict_proba(X_test)

        assert isinstance(results, Normal)
        assert results.index.equals(X_test.index)

        pd.testing.assert_index_equal(
            results.mean().index,
            X_test.index,
        )
        pd.testing.assert_index_equal(
            results.mean().columns,
            pd.Index(["Y"]),
        )

    def test_predict_proba_recovers_nonlinear_mean_and_noise(
        self,
        nonlinear_data,
        fitted_GAM_parameter,
    ):
        _, _, X_test, expected_test_mean = nonlinear_data
        parameter = fitted_GAM_parameter

        pred = parameter.predict_proba(X_test)
        pred_mean = pred.mean()["Y"].to_numpy()
        pred_var = pred.var()["Y"].to_numpy()
        expected = expected_test_mean.to_numpy()

        rmse = np.sqrt(np.mean((pred_mean - expected) ** 2))
        assert rmse < 0.30

        assert np.isfinite(pred_mean).all()
        assert np.isfinite(pred_var).all()
        assert (pred_var > 0).all()
