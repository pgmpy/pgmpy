import warnings

import pandas as pd
import pytest

from pgmpy.causal_discovery import ChowLiu

constant_continuous = pd.DataFrame(
    {
        "const": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [2.0, 1.0, 3.0, 2.0, 1.0, 3.0],
    }
)
constant_discrete = pd.DataFrame(
    {
        "const": ["a", "a", "a", "a", "a", "a"],
        "x": ["p", "q", "p", "q", "q", "p"],
        "y": ["m", "n", "n", "m", "n", "m"],
    }
)


@pytest.mark.parametrize("data", [constant_continuous, constant_discrete], ids=["continuous", "discrete"])
def test_fit_warns_on_constant_column(data):
    with pytest.warns(UserWarning, match="constant"):
        ChowLiu().fit(data)


def test_fit_no_warning_without_constant_column():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [2.0, 1.0, 3.0, 2.0, 1.0, 3.0]})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ChowLiu().fit(data)
    assert not [w for w in record if issubclass(w.category, UserWarning) and "constant" in str(w.message)]
