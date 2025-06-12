import pandas as pd
from impact_forecaster import CausalImpactForecaster

def test_forecaster():
    data = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5],
        "x2": [5, 4, 3, 2, 1],
    })
    target = pd.Series([2, 3, 4, 5, 6])

    model = CausalImpactForecaster()
    model.fit(data, target)
    assert model.model is not None