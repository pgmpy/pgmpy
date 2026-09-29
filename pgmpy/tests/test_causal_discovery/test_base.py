import numpy as np
import pandas as pd
import pytest
from skbase.lookup import all_objects

from pgmpy.causal_discovery import ANM, GES, PC, ChowLiu
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.causal_discovery.bivariate_scores import IndependenceScore
from pgmpy.ci_tests import GCM


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


def test_fit_narrows_union_tags_to_component():
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    data = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(size=200), "z": rng.normal(size=200)})

    assert PC.get_class_tag("assumption:linearity") is False
    pc = PC(ci_test="pearsonr").fit(data)
    assert pc.get_tag("data_types") == ["continuous"]
    assert pc.get_tag("assumption:linearity") is True
    assert pc.get_tag("assumption:faithfulness") is True

    pc.set_params(ci_test=lambda X, Y, Z, **kwargs: True).fit(data)
    assert pc.get_tag("assumption:linearity") is False
    assert pc.get_tag("data_types") == PC.get_class_tag("data_types")

    assert GCM(data=data).get_tag("assumption:linearity") is True
    assert GCM.get_class_tag("assumption:linearity") is False

    ges = GES(scoring_method="bic-g").fit(data)
    assert ges.get_tag("assumption:gaussian_noise") is True
    assert ges.get_tag("data_types") == ["continuous"]
    ges.set_params(scoring_method="bic-cg").fit(data)
    assert ges.get_tag("data_types") == ["discrete", "continuous", "mixed"]

    assert IndependenceScore(ci_test="chi_square").get_tag("data_types") == ["discrete"]
    anm = ANM().fit(data[["x", "y"]])
    assert anm.get_tag("data_types") == ["continuous"]


def test_filter_algorithms_by_tags():
    names = {
        cls.__name__
        for cls in all_objects(
            object_types=BaseCausalDiscovery,
            package_name="pgmpy.causal_discovery",
            return_names=False,
            filter_tags={"data_types": "continuous", "assumption:linearity": False, "capability:multivariate": True},
        )
    }
    assert {"PC", "GES"} <= names
    assert not names & {"ANM", "IGCI", "VarSort", "R2Sort"}

    dag_learners = {
        cls.__name__
        for cls in all_objects(
            object_types=BaseCausalDiscovery,
            package_name="pgmpy.causal_discovery",
            return_names=False,
            filter_tags={"identifiable_graph": "dag", "capability:multivariate": True},
        )
    }
    assert dag_learners == {"ExpertInLoop", "VarSort", "R2Sort"}
