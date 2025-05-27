import numpy as np
from pgmpy.causal.igci import infer_causal_direction

def test_entropy_direction():
    X = np.random.uniform(0, 1, 1000)
    Y = X ** 2 + np.random.normal(0, 0.01, 1000)
    direction = infer_causal_direction(X, Y, method="entropy")
    assert direction == "X -> Y"
