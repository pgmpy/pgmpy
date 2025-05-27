import numpy as np
from scipy.stats import entropy, pearsonr
from sklearn.linear_model import LinearRegression


def infer_causal_direction(X: np.ndarray, Y: np.ndarray, method="entropy") -> str:
    X = (X - X.min()) / (X.max() - X.min())
    Y = (Y - Y.min()) / (Y.max() - Y.min())

    if method == "entropy":
        bins = 30
        hist_x, _ = np.histogram(X, bins=bins, density=True)
        hist_y, _ = np.histogram(Y, bins=bins, density=True)
        H_x = entropy(hist_x + 1e-10)
        H_y = entropy(hist_y + 1e-10)
        return "X -> Y" if H_x < H_y else "Y -> X"

    elif method == "regression_residual":
        X_ = X.reshape(-1, 1)
        Y_ = Y.reshape(-1, 1)

        # Regress Y on X
        model_y_on_x = LinearRegression().fit(X_, Y)
        residuals_y_on_x = Y - model_y_on_x.predict(X_)

        # Regress X on Y
        model_x_on_y = LinearRegression().fit(Y_, X)
        residuals_x_on_y = X - model_x_on_y.predict(Y_)

        # Measure independence (correlation) between variables and residuals
        corr_x = abs(pearsonr(X, residuals_y_on_x.flatten())[0])
        corr_y = abs(pearsonr(Y, residuals_x_on_y.flatten())[0])

        # Add a margin to correlation comparison for robustness
        if corr_x + 1e-5 < corr_y:
            return "X -> Y"
        else:
            return "Y -> X"

    raise ValueError("Unsupported method. Choose 'entropy' or 'regression_residual'.")
