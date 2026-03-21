import pytest

from pgmpy.structure_score import (
    AIC,
    BIC,
    K2,
    AICCondGauss,
    AICGauss,
    BDeu,
    BDs,
    BICCondGauss,
    BICGauss,
    LogLikeliHood,
    LogLikelihoodCondGauss,
    LogLikelihoodGauss,
    get_scoring_method,
)
from pgmpy.structure_score._base import get_scoring_method as get_scoring_method_base


@pytest.mark.parametrize(
    ("cls", "expected_default_for"),
    [
        (K2, None),
        (BDeu, None),
        (BDs, None),
        (LogLikeliHood, None),
        (AIC, None),
        (BIC, "discrete"),
        (LogLikelihoodGauss, None),
        (AICGauss, None),
        (BICGauss, "continuous"),
        (LogLikelihoodCondGauss, None),
        (AICCondGauss, None),
        (BICCondGauss, "mixed"),
    ],
)
def test_default_for_tags(cls, expected_default_for):
    assert cls.get_class_tag("default_for") == expected_default_for


@pytest.mark.parametrize(
    ("cls", "expected_name"),
    [
        (K2, "k2"),
        (BDeu, "bdeu"),
        (BDs, "bds"),
        (LogLikeliHood, "ll-d"),
        (AIC, "aic-d"),
        (BIC, "bic-d"),
        (LogLikelihoodGauss, "ll-g"),
        (AICGauss, "aic-g"),
        (BICGauss, "bic-g"),
        (LogLikelihoodCondGauss, "ll-cg"),
        (AICCondGauss, "aic-cg"),
        (BICCondGauss, "bic-cg"),
    ],
)
def test_name_tags(cls, expected_name):
    assert cls.get_class_tag("name") == expected_name


@pytest.mark.parametrize(
    ("cls", "expected_supported_datatype"),
    [
        (K2, "discrete"),
        (BDeu, "discrete"),
        (BDs, "discrete"),
        (LogLikeliHood, "discrete"),
        (AIC, "discrete"),
        (BIC, "discrete"),
        (LogLikelihoodGauss, "continuous"),
        (AICGauss, "continuous"),
        (BICGauss, "continuous"),
        (LogLikelihoodCondGauss, "mixed"),
        (AICCondGauss, "mixed"),
        (BICCondGauss, "mixed"),
    ],
)
def test_supported_datatype_tags(cls, expected_supported_datatype):
    assert cls.get_class_tag("supported_datatype") == expected_supported_datatype


def test_get_scoring_method_export():
    assert get_scoring_method is get_scoring_method_base
