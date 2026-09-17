import pandas as pd
import pytest

from pgmpy.causal_discovery import ChowLiu


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(
            {
                "const": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [2.0, 1.0, 3.0, 2.0, 1.0, 3.0],
            }
        ),
        pd.DataFrame(
            {
                "const": ["a", "a", "a", "a", "a", "a"],
                "x": ["p", "q", "p", "q", "q", "p"],
                "y": ["m", "n", "n", "m", "n", "m"],
            }
        ),
    ],
    ids=["continuous", "discrete"],
)
def test_fit_warns_on_constant_column(data):
    with pytest.warns(UserWarning, match="constant"):
        ChowLiu().fit(data)
