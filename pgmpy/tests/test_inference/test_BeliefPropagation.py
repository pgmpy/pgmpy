import itertools

import numpy as np
import numpy.testing as np_test
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.example_models import load_model
from pgmpy.factors import factor_product
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.inference.ExactInference import BeliefPropagationWithMessagePassing
from pgmpy.models import DiscreteBayesianNetwork, FactorGraph, JunctionTree
from pgmpy.utils import compat_fns


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("torch not installed")
        config.set_backend("torch")
    yield request.param
    config.set_backend("numpy")


def build_junction_tree():
    junction_tree = JunctionTree([(("A", "B"), ("B", "C")), (("B", "C"), ("C", "D"))])
    phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
    phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
    phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))
    junction_tree.add_factors(phi1, phi2, phi3)
    return junction_tree


def build_bayesian_model():
    model = DiscreteBayesianNetwork([("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")])
    cpd_a = TabularCPD("A", 2, values=[[0.2], [0.8]])
    cpd_r = TabularCPD("R", 2, values=[[0.4], [0.6]])
    cpd_j = TabularCPD(
        "J",
        2,
        values=[[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
        evidence=["A", "R"],
        evidence_card=[2, 2],
    )
    cpd_q = TabularCPD("Q", 2, values=[[0.9, 0.2], [0.1, 0.8]], evidence=["J"], evidence_card=[2])
    cpd_l = TabularCPD(
        "L",
        2,
        values=[[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        evidence=["J", "G"],
        evidence_card=[2, 2],
    )
    cpd_g = TabularCPD("G", 2, values=[[0.6], [0.4]])
    model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
    return model


def build_factor_graph():
    factor_graph = FactorGraph()
    factor_graph.add_nodes_from(["A", "B", "C", "D"])

    phi1 = DiscreteFactor(["A"], [2], [0.4, 0.6])
    phi2 = DiscreteFactor(["B", "A"], [3, 2], [[0.2, 0.05], [0.3, 0.15], [0.5, 0.8]])
    phi3 = DiscreteFactor(["C", "B"], [2, 3], [[0.4, 0.5, 0.1], [0.6, 0.5, 0.9]])
    phi4 = DiscreteFactor(["D", "B"], [3, 3], [[0.1, 0.1, 0.2], [0.3, 0.2, 0.1], [0.6, 0.7, 0.7]])

    factor_graph.add_factors(phi1, phi2, phi3, phi4)
    factor_graph.add_edges_from(
        [
            (phi1, "A"),
            ("A", phi2),
            (phi2, "B"),
            ("B", phi3),
            (phi3, "C"),
            ("B", phi4),
            (phi4, "D"),
        ]
    )
    return factor_graph


@pytest.fixture
def bp_data(backend):
    return build_junction_tree(), build_bayesian_model()


@pytest.fixture
def factor_graph_bp():
    factor_graph = build_factor_graph()
    return factor_graph, BeliefPropagationWithMessagePassing(factor_graph)


class TestBeliefPropagation:
    def test_calibrate_clique_belief(self, bp_data, backend):
        junction_tree, _ = bp_data
        belief_propagation = BeliefPropagation(junction_tree)
        belief_propagation.calibrate()
        clique_belief = belief_propagation.get_clique_beliefs()

        phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
        phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
        phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))

        b_A_B = phi1 * (phi3.marginalize(["D"], inplace=False) * phi2).marginalize(["C"], inplace=False)
        b_B_C = phi2 * (phi1.marginalize(["A"], inplace=False) * phi3.marginalize(["D"], inplace=False))
        b_C_D = phi3 * (phi1.marginalize(["A"], inplace=False) * phi2).marginalize(["B"], inplace=False)

        assert clique_belief[("A", "B")] == b_A_B
        assert clique_belief[("B", "C")] == b_B_C
        assert clique_belief[("C", "D")] == b_C_D

    def test_calibrate_deeper_tree_single_pass(self, backend):
        junction_tree = JunctionTree(
            [(("C", "D"), ("B", "C")), (("B", "C"), ("A", "B")), (("C", "D"), ("D", "E")), (("D", "E"), ("E", "F"))]
        )
        factors = [
            DiscreteFactor(["A", "B"], [2, 2], [0.1, 0.9, 0.6, 0.4]),
            DiscreteFactor(["B", "C"], [2, 2], [0.7, 0.3, 0.2, 0.8]),
            DiscreteFactor(["C", "D"], [2, 2], [0.5, 0.5, 0.9, 0.1]),
            DiscreteFactor(["D", "E"], [2, 2], [0.3, 0.7, 0.4, 0.6]),
            DiscreteFactor(["E", "F"], [2, 2], [0.8, 0.2, 0.1, 0.9]),
        ]
        junction_tree.add_factors(*factors)
        joint = factor_product(*factors)
        for operation, calibrate in [("marginalize", "calibrate"), ("maximize", "max_calibrate")]:
            belief_propagation = BeliefPropagation(junction_tree)
            messages = []
            update_beliefs = belief_propagation._update_beliefs
            belief_propagation._update_beliefs = lambda *args, **kwargs: (
                messages.append(args),
                update_beliefs(*args, **kwargs),
            )[1]
            getattr(belief_propagation, calibrate)()
            assert len(messages) == 2 * (len(junction_tree.nodes()) - 1)
            assert belief_propagation._is_converged(operation=operation)
            for clique, belief in belief_propagation.get_clique_beliefs().items():
                assert belief == getattr(joint, operation)(list(set(joint.scope()) - set(clique)), inplace=False)

    def test_calibrate_sepset_belief(self, bp_data, backend):
        junction_tree, _ = bp_data
        belief_propagation = BeliefPropagation(junction_tree)
        belief_propagation.calibrate()
        sepset_belief = belief_propagation.get_sepset_beliefs()

        phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
        phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
        phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))

        b_B = (phi1 * (phi3.marginalize(["D"], inplace=False) * phi2).marginalize(["C"], inplace=False)).marginalize(
            ["A"], inplace=False
        )

        b_C = (phi2 * (phi1.marginalize(["A"], inplace=False) * phi3.marginalize(["D"], inplace=False))).marginalize(
            ["B"], inplace=False
        )

        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(sepset_belief[frozenset((("A", "B"), ("B", "C")))].values),
            compat_fns.to_numpy(b_B.values),
        )
        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(sepset_belief[frozenset((("B", "C"), ("C", "D")))].values),
            compat_fns.to_numpy(b_C.values),
        )

    def test_max_calibrate_clique_belief(self, bp_data, backend):
        junction_tree, _ = bp_data
        belief_propagation = BeliefPropagation(junction_tree)
        belief_propagation.max_calibrate()
        clique_belief = belief_propagation.get_clique_beliefs()

        phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
        phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
        phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))

        b_A_B = phi1 * (phi3.maximize(["D"], inplace=False) * phi2).maximize(["C"], inplace=False)
        b_B_C = phi2 * (phi1.maximize(["A"], inplace=False) * phi3.maximize(["D"], inplace=False))
        b_C_D = phi3 * (phi1.maximize(["A"], inplace=False) * phi2).maximize(["B"], inplace=False)

        assert clique_belief[("A", "B")] == b_A_B
        assert clique_belief[("B", "C")] == b_B_C
        assert clique_belief[("C", "D")] == b_C_D

    def test_max_calibrate_sepset_belief(self, bp_data, backend):
        junction_tree, _ = bp_data
        belief_propagation = BeliefPropagation(junction_tree)
        belief_propagation.max_calibrate()
        sepset_belief = belief_propagation.get_sepset_beliefs()

        phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
        phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
        phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))

        b_B = (phi1 * (phi3.maximize(["D"], inplace=False) * phi2).maximize(["C"], inplace=False)).maximize(
            ["A"], inplace=False
        )

        b_C = (phi2 * (phi1.maximize(["A"], inplace=False) * phi3.maximize(["D"], inplace=False))).maximize(
            ["B"], inplace=False
        )

        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(sepset_belief[frozenset((("A", "B"), ("B", "C")))].values),
            compat_fns.to_numpy(b_B.values),
        )
        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(sepset_belief[frozenset((("B", "C"), ("C", "D")))].values),
            compat_fns.to_numpy(b_C.values),
        )

    def test_query_single_variable(self, bp_data, backend):
        _, bayesian_model = bp_data
        belief_propagation = BeliefPropagation(bayesian_model)
        query_result = belief_propagation.query(["J"], show_progress=False)
        assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=[0.416, 0.584])

    def test_query_multiple_variable(self, bp_data, backend):
        _, bayesian_model = bp_data
        belief_propagation = BeliefPropagation(bayesian_model)
        query_result = belief_propagation.query(["Q", "J"], show_progress=False)
        assert query_result == DiscreteFactor(
            variables=["J", "Q"],
            cardinality=[2, 2],
            values=np.array([[0.3744, 0.0416], [0.1168, 0.4672]]),
        )

    def test_query_single_variable_with_evidence(self, bp_data, backend):
        _, bayesian_model = bp_data
        belief_propagation = BeliefPropagation(bayesian_model)
        query_result = belief_propagation.query(variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False)
        assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=np.array([0.6, 0.4]))

    def test_query_multiple_variable_with_evidence(self, bp_data, backend):
        _, bayesian_model = bp_data
        belief_propagation = BeliefPropagation(bayesian_model)
        query_result = belief_propagation.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        assert query_result == DiscreteFactor(
            variables=["J", "Q"],
            cardinality=[2, 2],
            values=np.array([[0.73636364, 0.08181818], [0.03636364, 0.14545455]]),
        )

    def test_query_common_var(self, bp_data, backend):
        _, bayesian_model = bp_data
        belief_propagation = BeliefPropagation(bayesian_model)
        with pytest.raises(ValueError):
            belief_propagation.query(variables=["J"], evidence=["J"])

    def test_map_query(self, bp_data, backend):
        _, bayesian_model = bp_data
        belief_propagation = BeliefPropagation(bayesian_model)
        map_query = belief_propagation.map_query(show_progress=False)
        assert map_query == {"A": 1, "R": 1, "J": 1, "Q": 1, "G": 0, "L": 0}

    def test_map_query_with_evidence(self, bp_data, backend):
        _, bayesian_model = bp_data
        belief_propagation = BeliefPropagation(bayesian_model)
        map_query = belief_propagation.map_query(["A", "R", "L"], {"J": 0, "Q": 1, "G": 0}, show_progress=False)
        assert map_query == {"A": 1, "R": 0, "L": 0}

    def test_map_query_common_var(self, bp_data, backend):
        _, bayesian_model = bp_data
        belief_propagation = BeliefPropagation(bayesian_model)
        with pytest.raises(ValueError):
            belief_propagation.map_query(variables=["J"], evidence=["J"])

    def test_query_keeps_calibration_and_model(self, bp_data, backend):
        junction_tree, bayesian_model = bp_data
        belief_propagation = BeliefPropagation(bayesian_model)
        junction_tree_ref = belief_propagation.junction_tree
        assert belief_propagation._is_converged(operation="marginalize")
        calibrations = []
        calibrate = belief_propagation._calibrate_junction_tree
        belief_propagation._calibrate_junction_tree = lambda *args, **kwargs: (
            calibrations.append(args),
            calibrate(*args, **kwargs),
        )[1]

        result = belief_propagation.query(["J"], evidence={"A": 0}, show_progress=False)
        assert result == belief_propagation.query(["J"], evidence={"A": 0}, show_progress=False)
        assert belief_propagation.map_query(["J", "Q"], evidence={"A": 0}, show_progress=False) == {"J": 0, "Q": 0}
        for kwargs in [{"variables": ["Z"]}, {"variables": ["J"], "evidence": {"Z": 0}}]:
            with pytest.raises(ValueError):
                belief_propagation.query(show_progress=False, **kwargs)
            with pytest.raises(ValueError):
                belief_propagation.map_query(show_progress=False, **kwargs)
        with pytest.raises(KeyError):
            belief_propagation.query(["J"], evidence={"A": 5}, show_progress=False)
        assert belief_propagation.query(["J"], show_progress=False) == BeliefPropagation(bayesian_model).query(
            ["J"], show_progress=False
        )
        assert belief_propagation.model is bayesian_model
        assert belief_propagation.junction_tree is junction_tree_ref
        assert calibrations == []
        assert sorted(bayesian_model.nodes()) == ["A", "G", "J", "L", "Q", "R"]
        assert BeliefPropagation(junction_tree).map_query(show_progress=False) == VariableElimination(
            junction_tree
        ).map_query(show_progress=False)

    def test_query_matches_variable_elimination(self, backend):
        alarm, asia = load_model("bnlearn/alarm"), load_model("bnlearn/asia")
        virtual_evidence = [
            TabularCPD("smoke", 2, [[0.9], [0.1]], state_names={"smoke": asia.states["smoke"]}),
            TabularCPD("xray", 2, [[0.3], [0.7]], state_names={"xray": asia.states["xray"]}),
        ]
        cases = [
            (alarm, alarm, ["HR"], {}, None),
            (alarm, alarm, ["HR", "BP"], {"CVP": "LOW"}, None),
            (alarm, alarm, ["HISTORY"], {"MINVOLSET": "HIGH", "HRBP": "LOW"}, None),
            (alarm, alarm, ["CVP", "PCWP"], {"LVFAILURE": "TRUE"}, None),
            (asia, asia, ["dysp", "lung"], {}, virtual_evidence),
            (asia, asia, ["dysp", "lung"], {"asia": "yes"}, virtual_evidence),
            (asia.to_junction_tree(), asia, ["dysp", "lung"], {"asia": "yes"}, virtual_evidence),
        ]
        for model, reference, variables, evidence, virt in cases:
            bp, ve = BeliefPropagation(model), VariableElimination(reference)
            kwargs = {"evidence": evidence, "virtual_evidence": virt, "show_progress": False}
            assert bp.query(variables, **kwargs) == ve.query(variables, **kwargs)
            bp_marginals = bp.query(variables, joint=False, **kwargs)
            ve_marginals = ve.query(variables, joint=False, **kwargs)
            assert [bp_marginals[var] for var in variables] == [ve_marginals[var] for var in variables]
            assert bp.map_query(variables, **kwargs) == ve.map_query(variables, **kwargs)
            assert sorted(bp.model.nodes()) == sorted(model.nodes())

        bp = BeliefPropagation(asia)
        for bad_cpd in [
            TabularCPD("smoke", 2, [[0.9], [0.1]]),
            TabularCPD("smoke", 3, [[0.5], [0.3], [0.2]]),
            TabularCPD("nonexistent", 2, [[0.9], [0.1]]),
            TabularCPD("smoke", 2, [[0.9, 0.2], [0.1, 0.8]], evidence=["asia"], evidence_card=[2]),
        ]:
            with pytest.raises(ValueError):
                bp.query(["dysp"], virtual_evidence=[bad_cpd], show_progress=False)

    def test_issue_1048(self, backend):
        model = DiscreteBayesianNetwork()

        parents = ["parent"]
        children = [f"child_{i}" for i in range(10)]

        model.add_nodes_from(parents + children)
        model.add_edges_from(itertools.product(parents, children))

        model.add_cpds(TabularCPD(parents[0], 2, [[0.5], [0.5]]))
        for c in children:
            model.add_cpds(TabularCPD(c, 2, [[0.9, 0.1], [0.1, 0.9]], evidence=parents, evidence_card=[2]))

        inf = BeliefPropagation(model)
        inf.calibrate()
        evidence = {}

        expected_evidences = [
            {},
            {"child_0": 1},
            {"child_0": 1, "child_1": 1},
            {"child_0": 1, "child_1": 1, "child_2": 1},
        ]
        expected_values = [
            np.array([0.5, 0.5]),
            np.array([0.1, 0.9]),
            np.array([0.0122, 0.9878]),
            np.array([0.0014, 0.9987]),
        ]
        for i, c in enumerate(children[:4]):
            assert evidence == expected_evidences[i]
            np_test.assert_almost_equal(
                compat_fns.to_numpy(
                    inf.query(["parent"], evidence, show_progress=False).normalize(inplace=False).values
                ),
                expected_values[i],
                decimal=2,
            )
            evidence.update({c: 1})


class TestBeliefPropagationWithMessagePassing:
    def test_query_single_variable(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        res = belief_propagation.query(["C"])
        assert np.allclose(res["C"].values, np.array([0.217, 0.783]), atol=1e-20)

    def test_query_multiple_variable(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        res = belief_propagation.query(["A", "B", "C", "D"])
        assert np.allclose(res["A"].values, np.array([0.4, 0.6]), atol=1e-20)
        assert np.allclose(res["B"].values, np.array([0.11, 0.21, 0.68]), atol=1e-20)
        assert np.allclose(res["C"].values, np.array([0.217, 0.783]), atol=1e-20)
        assert np.allclose(res["D"].values, np.array([0.168, 0.143, 0.689]), atol=1e-20)

    def test_query_single_variable_with_evidence(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        res = belief_propagation.query(["B", "C"], {"A": 1, "D": 0})
        assert np.allclose(res["B"].values, np.array([0.02777778, 0.08333333, 0.88888889]), atol=1e-20)
        assert np.allclose(res["C"].values, np.array([0.14166667, 0.85833333]), atol=1e-20)

    def test_query_multiple_variable_with_evidence(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        res = belief_propagation.query(["B", "C"], {"A": 1, "D": 0})
        assert np.allclose(res["B"].values, np.array([0.02777778, 0.08333333, 0.88888889]), atol=1e-20)
        assert np.allclose(res["C"].values, np.array([0.14166667, 0.85833333]), atol=1e-20)

    def test_query_single_variable_with_virtual_evidence(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        ve = [TabularCPD("A", 2, [[0.1], [0.9]])]
        res = belief_propagation.query(["B"], virtual_evidence=ve)
        assert np.allclose(res["B"].values, np.array([0.06034483, 0.16034483, 0.77931034]), atol=1e-20)

    def test_query_multiple_variable_with_multiple_evidence_and_virtual_evidence(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        ve = [
            TabularCPD("A", 2, [[0.027], [0.972]]),
            TabularCPD("B", 3, [[0.3], [0.6], [0.1]]),
        ]
        res = belief_propagation.query(["B", "C"], evidence={"D": 0}, virtual_evidence=ve)
        assert np.allclose(res["B"].values, np.array([0.05938567, 0.3440273, 0.59658703]), atol=1e-20)
        assert np.allclose(res["C"].values, np.array([0.25542662, 0.74457338]), atol=1e-20)

    def test_query_allows_multiple_virtual_evidence_per_variable(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        ve1 = [
            TabularCPD("A", 2, [[0.1], [0.9]]),
            TabularCPD("A", 2, [[0.3], [0.7]]),
        ]
        res1 = belief_propagation.query(["B"], virtual_evidence=ve1)
        cpd = TabularCPD("A", 2, [[0.1 * 0.3], [0.9 * 0.7]])
        cpd.normalize()
        res2 = belief_propagation.query(["B"], virtual_evidence=[cpd])
        assert np.allclose(res1["B"].values, res2["B"].values, atol=1e-20)
        assert np.allclose(res2["B"].values, np.array([0.05461538, 0.15461538, 0.79076923]), atol=1e-20)

    def test_query_error_obs_var_has_evidence(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        with pytest.raises(
            ValueError,
            match="Can't have the same variables in both `evidence` and `virtual_evidence`",
        ):
            belief_propagation.query(["B"], evidence={"A": 1}, virtual_evidence=[TabularCPD("A", 2, [[0.1], [0.9]])])

    def test_query_single_variable_can_return_all_computed_messages(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        res, messages = belief_propagation.query(["B"], get_messages=True)
        assert np.allclose(res["B"].values, np.array([0.11, 0.21, 0.68]), atol=1e-20)
        assert np.allclose(messages["['A'] -> A"], np.array([0.4, 0.6]), atol=1e-20)
        assert np.allclose(messages["['B', 'A'] -> B"], np.array([0.11, 0.21, 0.68]), atol=1e-20)
        assert np.allclose(
            messages["['C', 'B'] -> B"],
            np.array([0.33333333, 0.33333333, 0.33333333]),
            atol=1e-20,
        )
        assert np.allclose(
            messages["['D', 'B'] -> B"],
            np.array([0.33333333, 0.33333333, 0.33333333]),
            atol=1e-20,
        )

    def test_query_multiple_variable_returns_each_message_once(self, factor_graph_bp):
        _, belief_propagation = factor_graph_bp
        res, messages = belief_propagation.query(["C", "B"], get_messages=True)
        assert np.allclose(res["B"].values, np.array([0.11, 0.21, 0.68]), atol=1e-20)
        assert np.allclose(res["C"].values, np.array([0.217, 0.783]), atol=1e-20)

        assert np.allclose(messages["['A'] -> A"], np.array([0.4, 0.6]), atol=1e-20)
        assert np.allclose(messages["['B', 'A'] -> B"], np.array([0.11, 0.21, 0.68]), atol=1e-20)

        assert np.allclose(
            messages["['C', 'B'] -> B"],
            np.array([0.33333333, 0.33333333, 0.33333333]),
            atol=1e-20,
        )
        assert np.allclose(
            messages["['D', 'B'] -> B"],
            np.array([0.33333333, 0.33333333, 0.33333333]),
            atol=1e-20,
        )

        assert np.allclose(messages["['C', 'B'] -> C"], np.array([0.217, 0.783]), atol=1e-20)
