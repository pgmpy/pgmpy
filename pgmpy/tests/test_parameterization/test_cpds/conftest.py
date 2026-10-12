import numpy as np
import pytest

from pgmpy.factors.discrete import TabularCPD as LegacyTabularCPD
from pgmpy.models import DiscreteBayesianNetwork


@pytest.fixture(scope="module")
def legacy_network():
    """A network where y has three parents, data simulated from it, and sample weights, to compare TabularCPD with the
    legacy network estimators.

    A random table over three parents shows any column mix-up. Rows with (x1, x2) = (2, 1) are dropped so that
    combination is never seen, and zero weights on (x1, x3) = (0, 1) leave another combination without data.
    """
    rng = np.random.default_rng(0)
    model = DiscreteBayesianNetwork([("x1", "y"), ("x2", "y"), ("x3", "y")])
    model.add_cpds(
        LegacyTabularCPD("x1", 3, [[0.3], [0.3], [0.4]]),
        LegacyTabularCPD("x2", 2, [[0.5], [0.5]]),
        LegacyTabularCPD("x3", 2, [[0.6], [0.4]]),
        LegacyTabularCPD("y", 3, rng.dirichlet([1, 1, 1], size=12).T, ["x1", "x2", "x3"], [3, 2, 2]),
    )
    data = model.simulate(1000, seed=0, show_progress=False)
    data = data[~((data["x1"] == 2) & (data["x2"] == 1))]
    weights = np.where((data["x1"] == 0) & (data["x3"] == 1), 0.0, rng.uniform(0.5, 2, len(data)))
    return model, data, weights
