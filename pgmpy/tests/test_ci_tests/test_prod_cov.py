from pgmpy.ci_tests import ProdCov

from pgmpy.tests.test_ci_tests._multivariate_fixtures import pillai_data


def test_prod_cov_approx(pillai_data):
    for df in pillai_data["indep"]:
        test = ProdCov(data=df, n_permutations=500, random_state=0)
        test("X", "Y", [])
        assert test.p_value_ <= 0.05

    for df in pillai_data["indep"][:3] + pillai_data["indep"][4:]:
        test = ProdCov(data=df, n_permutations=500, random_state=0)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.p_value_ >= 0.05

    for df in pillai_data["dep"]:
        test = ProdCov(data=df, n_permutations=500, random_state=0)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.p_value_ <= 0.05
