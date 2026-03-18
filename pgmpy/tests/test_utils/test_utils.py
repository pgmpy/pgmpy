import os
import random

import numpy as np
import pandas as pd
import pytest
from tqdm.auto import tqdm

from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.utils import (
    discretize,
    get_example_model,
    llm_pairwise_orient,
    preprocess_data,
)


class TestDiscretization:
    def setup_method(self, method):
        rng = np.random.default_rng(42)
        X = rng.standard_normal(1000)
        Y = 0.2 * X + rng.standard_normal(1000)
        Z = 0.4 * X + 0.5 * Y + rng.standard_normal(1000)

        self.data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_rounding_disc(self):
        df_disc = discretize(
            data=self.data, cardinality={"X": 5, "Y": 4, "Z": 3}, method="rounding"
        )
        assert df_disc["X"].nunique() == 5
        assert df_disc["Y"].nunique() == 4
        assert df_disc["Z"].nunique() == 3

        df_disc = discretize(
            data=self.data, cardinality={"X": 5, "Y": 4, "Z": 3}, method="quantile"
        )
        assert df_disc["X"].nunique() == 5
        assert df_disc["Y"].nunique() == 4
        assert df_disc["Z"].nunique() == 3


class TestPairwiseOrientation:
    @pytest.mark.skipif(
        "GEMINI_API_KEY" not in os.environ, reason="Gemini API key is not set"
    )
    def test_llm(self):
        descriptions = {
            "Age": "The age of a person",
            "Workclass": "The workplace where the person is "
            "employed such as Private industry, or self employed",
            "Education": "The highest level of education the " "person has finished",
            "MaritalStatus": "The marital status of the person",
            "Occupation": "The kind of job the person does. "
            "For example, sales, craft repair, clerical",
            "Relationship": "The relationship status of the person",
            "Race": "The ethnicity of the person",
            "Sex": "The sex or gender of the person",
            "HoursPerWeek": "The number of hours per week the person works",
            "NativeCountry": "The native country of the person",
            "Income": "The income i.e. amount of money the person makes",
        }

        assert llm_pairwise_orient(
            x="Age", y="Income", descriptions=descriptions, domain="Social Sciences"
        ) == ("Age", "Income")
        assert llm_pairwise_orient(
            x="Income", y="Age", descriptions=descriptions, domain="Social Sciences"
        ) == ("Age", "Income")


class TestPreprocessData:
    def setup_method(self, method):
        self.data_raw = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0
        )

        self.data_proc = self.data_raw.copy()
        self.data_proc["A_cat"] = self.data_proc.A_cat.astype("category")
        self.data_proc["B_cat"] = self.data_proc.C_cat.astype("category")
        self.data_proc["C_cat"] = self.data_proc.C_cat.astype("category")

        self.data_proc_proc = self.data_proc.copy()
        cat_type = pd.CategoricalDtype(
            categories=np.array(sorted(self.data_proc_proc.B_int.unique())),
            ordered=True,
        )

        self.data_proc_proc["B_int"] = self.data_proc_proc.B_int.astype(cat_type)

    def test_preprocess_data(self):
        df, dtypes = preprocess_data(self.data_raw)
        assert dtypes == {
            "A": "N",
            "B": "N",
            "C": "N",
            "A_cat": "C",
            "B_cat": "C",
            "C_cat": "C",
            "B_int": "N",
        }

        df, dtypes = preprocess_data(self.data_proc)
        assert dtypes == {
            "A": "N",
            "B": "N",
            "C": "N",
            "A_cat": "C",
            "B_cat": "C",
            "C_cat": "C",
            "B_int": "N",
        }

        df, dtypes = preprocess_data(self.data_proc_proc)
        assert dtypes == {
            "A": "N",
            "B": "N",
            "C": "N",
            "A_cat": "C",
            "B_cat": "C",
            "C_cat": "C",
            "B_int": "O",
        }


class TestGetExampleModel:
    def test_get_categorical_models(self):
        """Test loading of categorical Bayesian network models."""
        cat_models = {
            "asia",
            "cancer",
            "earthquake",
            "sachs",
            "survey",
            "alarm",
            "barley",
            "child",
            "insurance",
            "mildew",
            "water",
            "hailfinder",
            "hepar2",
            "win95pts",
            "andes",
            "diabetes",
            "link",
            "munin1",
            "munin2",
            "munin3",
            "munin4",
            "pathfinder",
            "pigs",
            "munin",
        }

        # Randomly select 5 categorical models to test
        choices = random.sample(list(cat_models), k=5)
        for model in tqdm(choices, desc="Testing categorical models"):
            m = get_example_model(model=model)
            # Basic model validation
            assert m is not None
            assert hasattr(m, "nodes")
            assert hasattr(m, "edges")
            del m

    def test_get_continuous_models(self):
        # Test ecoli70 model specifically as we have its structure
        model = get_example_model("ecoli70")
        assert isinstance(model, LinearGaussianBayesianNetwork)
        assert len(model.nodes()) == 46  # Number of nodes in ecoli70

        # Verify some known relationships from the provided structure
        assert ("asnA", "icdA") in model.edges()
        assert ("asnA", "lacA") in model.edges()
        assert ("sucA", "atpD") in model.edges()

        # Verify CPD structure for a known node
        cpd = model.get_cpds("aceB")
        assert cpd is not None
        assert cpd.variable == "aceB"
        assert len(cpd.evidence) == 1
        assert "icdA" in cpd.evidence

    def test_get_example_model_dagitty(self):
        dag_models = [
            "M-bias",
            "confounding",
            "mediator",
            "paths",
            "Sebastiani_2005",
            "Polzer_2012",
            "Schipf_2010",
            "Shrier_2008",
            "Acid_1996",
            "Thoemmes_2013",
            "Kampen_2014",
            "Didelez_2010",
        ]
        # Would take too much time to load all the models. Hence, randomly select
        # 3 and try to load them.
        choices = random.sample(dag_models, k=3)
        for model in tqdm(choices):
            print(model)
            m = get_example_model(model=model)
            assert m is not None
            assert hasattr(m, "nodes")
            assert hasattr(m, "edges")
            del m

    def test_invalid_model_name(self):
        """Test handling of invalid model names."""
        with pytest.raises(ValueError):
            get_example_model("nonexistent_model")

    def test_model_categorization(self):
        """Test that all models are properly categorized."""
        # Test a model from each category
        cat_model = get_example_model("asia")
        assert not isinstance(cat_model, LinearGaussianBayesianNetwork)

        cont_model = get_example_model("magic-irri")
        assert isinstance(cont_model, LinearGaussianBayesianNetwork)
