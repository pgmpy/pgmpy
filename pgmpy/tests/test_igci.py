import numpy as np
from pgmpy.causal.igci import infer_causal_direction


def test_entropy_direction():
    np.random.seed(42)
    X = np.random.uniform(0, 1, 1000)
    noise = np.random.normal(0, 0.01, 1000)
    Y = X**2 + noise
    direction = infer_causal_direction(X, Y, method="regression_residual")
    assert direction == "X -> Y", f"Expected 'X -> Y', but got {direction}"
