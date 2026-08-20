import pandas as pd
import pytest
from skpro.distributions.normal import Normal
from skpro.distributions.poisson import Poisson

from pgmpy.distributions.nominal import NominalDistribution
from pgmpy.parameter.adapter.DistributionAdapter import DistributionAdapter


class TestDistributionAdapter:
    def test_base_parameter_default(self):
        distribution = Normal(mu=1.5, sigma=2.0)
        parameter = DistributionAdapter(distribution=distribution)

        assert parameter.__class__.__name__ == "DistributionAdapter"
        assert parameter.distribution is distribution
        assert parameter.get_params(deep=False)["distribution"] is distribution

        assert parameter.get_class_tag("parameter_type") == "distribution"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("can_be_root") is True

    def test_fit(self):
        distribution = Normal(mu=1.5, sigma=2.0)
        parameter = DistributionAdapter(distribution=distribution)
        X = pd.DataFrame({"A": [0.0, 1.0, 2.0]}, index=pd.Index(["a", "b", "c"]))
        with pytest.raises(NotImplementedError):
            parameter.fit(X)

    @pytest.mark.parametrize(
        ("distribution", "expected_type"),
        [
            (Normal(mu=1.5, sigma=2.0, columns=["y"]), Normal),
            (Poisson(mu=3.0, columns=["y"]), Poisson),
            (NominalDistribution([[0.7, 0.3]], ["A", "B"], columns=["y"]), NominalDistribution),
        ],
    )
    def test_predict_proba(self, distribution, expected_type):
        X = pd.DataFrame({"A": [0.0, 1.0, 2.0]})
        parameter = DistributionAdapter(distribution=distribution)

        result = parameter.predict_proba(X)

        assert parameter.is_fitted is True
        assert isinstance(result, expected_type)
        assert result.index.equals(X.index)
        assert result.columns.equals(pd.Index(["y"]))

    def test_rejects_non_skpro_distribution(self):
        with pytest.raises(TypeError, match="skpro distribution"):
            DistributionAdapter(distribution="not a distribution")
