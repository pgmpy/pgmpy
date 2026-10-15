import numpy as np
import pandas as pd
import pytest
from skbase.lookup import all_objects

from pgmpy.causal_discovery import ANM, GES, IGCI, PC, ChowLiu, ExpertInLoop
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.causal_discovery.bivariate_scores import BaseBivariateScore
from pgmpy.ci_tests import GCM, BaseCITest, IndependenceMatch, get_ci_test
from pgmpy.ci_tests._base import _CITestResult
from pgmpy.independencies import Independencies
from pgmpy.structure_score import get_scoring_method


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


def test_estimators_use_identity_equality_and_hash():
    pc = PC()
    assert pc == pc
    assert PC() != PC()
    assert len({pc, PC()}) == 2


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

    anm = ANM().fit(data[["x", "y"]])
    assert anm.get_tag("data_types") == ["continuous"]

    # Storing the resolved component as `ci_test_` or `scoring_method_` is enough; `fit` narrows the tags.
    class MinimalPC(PC):
        def _fit(self, X, independencies=None):
            self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
            return self

    class MinimalGES(GES):
        def _fit(self, X):
            self.scoring_method_ = get_scoring_method(self.scoring_method, X)
            return self

    assert MinimalPC(ci_test="pearsonr").fit(data).get_tag("assumption:linearity") is True
    assert MinimalGES(scoring_method="bic-g").fit(data).get_tag("assumption:gaussian_noise") is True


def test_fit_narrowing_handles_untagged_and_wider_objects():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 300)
    data = pd.DataFrame({"x": x, "y": x**3 + rng.normal(0, 0.5, 300)})

    class UntaggedCITest(BaseCITest):
        def _compute_result(self, X, Y, Z):
            return _CITestResult(statistic=0.0, p_value=1.0)

    pc = PC(ci_test=UntaggedCITest()).fit(data)
    assert pc.get_tag("data_types") == ["discrete", "continuous", "mixed"]
    assert pc.get_tag("assumption:linearity") is False

    class AnyDataScore(BaseBivariateScore):
        _tags = {"name": "any_data", "input_type": "cause_residual", "data_types": ["discrete", "continuous", "mixed"]}

        def __call__(self, x, y):
            return float(np.var(y))

    anm = ANM(scoring_method=AnyDataScore()).fit(data)
    assert anm.get_tag("data_types") == ["continuous"]
    assert anm.get_tag("assumption:linearity") is False

    class UntaggedAlgorithm(BaseCausalDiscovery):
        def _fit(self, X):
            self.ci_test_ = get_ci_test(test="pearsonr", data=X)
            return self

    assert UntaggedAlgorithm().fit(data).get_tag("data_types") == ["continuous"]


def test_fit_narrows_tags_to_components_used():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 300)
    data = pd.DataFrame({"x": x, "y": x**3 + rng.normal(0, 0.5, 300), "z": rng.normal(size=300)})

    pc = PC(show_progress=False).fit(data, independencies=Independencies(["x", "z"], ["y", "z"]))
    assert isinstance(pc.ci_test_, IndependenceMatch)
    assert pc.get_tag("data_types") == ["discrete", "continuous", "mixed"]
    assert pc.get_tag("assumption:linearity") is False

    expert_in_loop = ExpertInLoop(pairwise_estimator=IGCI(), show_progress=False).fit(data)
    assert expert_in_loop.get_tag("assumption:low_noise") is True
    assert expert_in_loop.get_tag("data_types") == ["continuous"]


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
