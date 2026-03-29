#!/usr/bin/env python3
from collections import defaultdict

import numpy as np
import pytest

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import Inference
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork


@pytest.fixture
def inference_data():
    bayesian = DiscreteBayesianNetwork([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])
    a_cpd = TabularCPD("a", 2, [[0.4], [0.6]])
    b_cpd = TabularCPD("b", 2, [[0.2, 0.4], [0.8, 0.6]], evidence=["a"], evidence_card=[2])
    c_cpd = TabularCPD("c", 2, [[0.1, 0.2], [0.9, 0.8]], evidence=["b"], evidence_card=[2])
    d_cpd = TabularCPD("d", 2, [[0.4, 0.3], [0.6, 0.7]], evidence=["c"], evidence_card=[2])
    e_cpd = TabularCPD("e", 2, [[0.3, 0.2], [0.7, 0.8]], evidence=["d"], evidence_card=[2])
    bayesian.add_cpds(a_cpd, b_cpd, c_cpd, d_cpd, e_cpd)

    markov = DiscreteMarkovNetwork([("a", "b"), ("b", "d"), ("a", "c"), ("c", "d")])
    factor_1 = DiscreteFactor(["a", "b"], [2, 2], np.array([100, 1, 1, 100]))
    factor_2 = DiscreteFactor(["a", "c"], [2, 2], np.array([40, 30, 100, 20]))
    factor_3 = DiscreteFactor(["b", "d"], [2, 2], np.array([1, 100, 100, 1]))
    factor_4 = DiscreteFactor(["c", "d"], [2, 2], np.array([60, 60, 40, 40]))
    markov.add_factors(factor_1, factor_2, factor_3, factor_4)
    return bayesian, markov


class TestInferenceBase:
    def test_bayesian_inference_init(self, inference_data):
        bayesian, _ = inference_data
        infer_bayesian = Inference(bayesian)
        infer_bayesian._initialize_structures()
        assert set(infer_bayesian.variables) == {"a", "b", "c", "d", "e"}
        assert infer_bayesian.cardinality == {"a": 2, "b": 2, "c": 2, "d": 2, "e": 2}
        assert isinstance(infer_bayesian.factors, defaultdict)
        assert set(infer_bayesian.factors["a"]) == {
            bayesian.get_cpds("a").to_factor(),
            bayesian.get_cpds("b").to_factor(),
        }
        assert set(infer_bayesian.factors["b"]) == {
            bayesian.get_cpds("b").to_factor(),
            bayesian.get_cpds("c").to_factor(),
        }
        assert set(infer_bayesian.factors["c"]) == {
            bayesian.get_cpds("c").to_factor(),
            bayesian.get_cpds("d").to_factor(),
        }
        assert set(infer_bayesian.factors["d"]) == {
            bayesian.get_cpds("d").to_factor(),
            bayesian.get_cpds("e").to_factor(),
        }
        assert set(infer_bayesian.factors["e"]) == {bayesian.get_cpds("e").to_factor()}

    def test_markov_inference_init(self, inference_data):
        _, markov = inference_data
        infer_markov = Inference(markov)
        infer_markov._initialize_structures()
        assert set(infer_markov.variables) == {"a", "b", "c", "d"}
        assert infer_markov.cardinality == {"a": 2, "b": 2, "c": 2, "d": 2}
        assert infer_markov.factors == {
            "a": [
                DiscreteFactor(["a", "b"], [2, 2], np.array([100, 1, 1, 100])),
                DiscreteFactor(["a", "c"], [2, 2], np.array([40, 30, 100, 20])),
            ],
            "b": [
                DiscreteFactor(["a", "b"], [2, 2], np.array([100, 1, 1, 100])),
                DiscreteFactor(["b", "d"], [2, 2], np.array([1, 100, 100, 1])),
            ],
            "c": [
                DiscreteFactor(["a", "c"], [2, 2], np.array([40, 30, 100, 20])),
                DiscreteFactor(["c", "d"], [2, 2], np.array([60, 60, 40, 40])),
            ],
            "d": [
                DiscreteFactor(["b", "d"], [2, 2], np.array([1, 100, 100, 1])),
                DiscreteFactor(["c", "d"], [2, 2], np.array([60, 60, 40, 40])),
            ],
        }
