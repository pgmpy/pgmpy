import numpy as np
from scipy.stats import entropy


def infer_causal_direction(X: np.ndarray, Y: np.ndarray, method="entropy") -> str:
    X = (X - X.min()) / (X.max() - X.min())  # Normalize
    Y = (Y - Y.min()) / (Y.max() - Y.min())

    if method == "entropy":
        bins = 30
        hist_x, _ = np.histogram(X, bins=bins, density=True)
        hist_y, _ = np.histogram(Y, bins=bins, density=True)

        H_x = entropy(hist_x + 1e-10)  # Add epsilon to avoid log(0)
        H_y = entropy(hist_y + 1e-10)

        if H_x < H_y:
            return "X -> Y"
        else:
            return "Y -> X"

    raise ValueError("Unsupported method. Choose 'entropy'.")
