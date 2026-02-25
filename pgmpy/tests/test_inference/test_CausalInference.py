import numpy as np
import numpy.testing as np_test
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork, SEMGraph

np.random.seed(42)


@pytest.fixture
def inference():
    game = DiscreteBayesianNetwork(
        [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
    )
    inference = CausalInference(game)
    return inference


@pytest.fixture
def inference_bd():
    dag_bd1 = DiscreteBayesianNetwork([("X", "Y"), ("Z1", "X"), ("Z1", "Y")])
    inference_bd = CausalInference(dag_bd1)
    return inference_bd


@pytest.fixture
def inference_bd2():
    dag_bd2 = DiscreteBayesianNetwork(
        [("X", "Y"), ("Z1", "X"), ("Z1", "Z2"), ("Z2", "Y")]
    )
    inference_bd2 = CausalInference(dag_bd2)
    return inference_bd2


@pytest.fixture
def infer_dag():
    # Model example taken from Constructing Separators and Adjustment Sets
    # in Ancestral Graphs UAI 2014.
    model_dag = DAG(
        [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]
    )
    infer_dag = CausalInference(model_dag)
    return infer_dag


@pytest.fixture
def infer_sem():
    model_sem = SEMGraph(
        [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]
    )
    infer_sem = CausalInference(model_sem)
    return infer_sem


@pytest.fixture
def demo_inference():
    demo = SEMGraph(
        ebunch=[
            ("xi1", "x1"),
            ("xi1", "x2"),
            ("xi1", "x3"),
            ("xi1", "eta1"),
            ("eta1", "y1"),
            ("eta1", "y2"),
            ("eta1", "y3"),
            ("eta1", "y4"),
            ("eta1", "eta2"),
            ("xi1", "eta2"),
            ("eta2", "y5"),
            ("eta2", "y6"),
            ("eta2", "y7"),
            ("eta2", "y8"),
        ],
        latents=["xi1", "eta1", "eta2"],
        err_corr=[
            ("y1", "y5"),
            ("y2", "y6"),
            ("y2", "y4"),
            ("y3", "y7"),
            ("y4", "y8"),
            ("y6", "y8"),
        ],
    )

    demo_inference = CausalInference(demo)
    return demo_inference


@pytest.fixture
def union_inference():
    union = SEMGraph(
        ebunch=[
            ("yrsmill", "unionsen"),
            ("age", "laboract"),
            ("age", "deferenc"),
            ("deferenc", "laboract"),
            ("deferenc", "unionsen"),
            ("laboract", "unionsen"),
        ],
        latents=[],
        err_corr=[("yrsmill", "age")],
    )

    union_inference = CausalInference(union)
    return union_inference


@pytest.fixture
def demo_params_inference():
    demo_params = SEMGraph(
        ebunch=[
            ("xi1", "x1", 0.4),
            ("xi1", "x2", 0.5),
            ("xi1", "x3", 0.6),
            ("xi1", "eta1", 0.3),
            ("eta1", "y1", 1.1),
            ("eta1", "y2", 1.2),
            ("eta1", "y3", 1.3),
            ("eta1", "y4", 1.4),
            ("eta1", "eta2", 0.1),
            ("xi1", "eta2", 0.2),
            ("eta2", "y5", 0.7),
            ("eta2", "y6", 0.8),
            ("eta2", "y7", 0.9),
            ("eta2", "y8", 1.0),
        ],
        latents=["xi1", "eta1", "eta2"],
        err_corr=[
            ("y1", "y5", 1.5),
            ("y2", "y6", 1.6),
            ("y2", "y4", 1.9),
            ("y3", "y7", 1.7),
            ("y4", "y8", 1.8),
            ("y6", "y8", 2.0),
        ],
        err_var={
            "y1": 2.1,
            "y2": 2.2,
            "y3": 2.3,
            "y4": 2.4,
            "y5": 2.5,
            "y6": 2.6,
            "y7": 2.7,
            "y8": 2.8,
            "x1": 3.1,
            "x2": 3.2,
            "x3": 3.3,
            "eta1": 2.9,
            "eta2": 3.0,
            "xi1": 3.4,
        },
    )

    demo_params_inference = CausalInference(demo_params)
    return demo_params_inference


@pytest.fixture
def custom_inference():
    custom = SEMGraph(
        ebunch=[
            ("xi1", "eta1"),
            ("xi1", "y1"),
            ("xi1", "y4"),
            ("xi1", "x1"),
            ("xi1", "x2"),
            ("y4", "y1"),
            ("y1", "eta2"),
            ("eta2", "y5"),
            ("y1", "eta1"),
            ("eta1", "y2"),
            ("eta1", "y3"),
        ],
        latents=["xi1", "eta1", "eta2"],
        err_corr=[("y1", "y2"), ("y2", "y3")],
        err_var={},
    )

    custom_inference = CausalInference(custom)
    return custom_inference


@pytest.fixture
def custom_inference2():
    model = DiscreteBayesianNetwork(
        ebunch=[("Z", "X"), ("X", "Y"), ("U", "Y"), ("U", "X")], latents=["U"]
    )

    custom_inference2 = CausalInference(model)
    return custom_inference2


def get_simpson_model():
    simpson_model = DiscreteBayesianNetwork([("S", "T"), ("T", "C"), ("S", "C")])
    cpd_s = TabularCPD(
        variable="S",
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={"S": ["m", "f"]},
    )
    cpd_t = TabularCPD(
        variable="T",
        variable_card=2,
        values=[[0.25, 0.75], [0.75, 0.25]],
        evidence=["S"],
        evidence_card=[2],
        state_names={"S": ["m", "f"], "T": [0, 1]},
    )
    cpd_c = TabularCPD(
        variable="C",
        variable_card=2,
        values=[[0.3, 0.4, 0.7, 0.8], [0.7, 0.6, 0.3, 0.2]],
        evidence=["S", "T"],
        evidence_card=[2, 2],
        state_names={"S": ["m", "f"], "T": [0, 1], "C": [0, 1]},
    )
    simpson_model.add_cpds(cpd_s, cpd_t, cpd_c)

    return simpson_model


@pytest.fixture
def simp_inference():
    simpson_model = get_simpson_model()
    simp_inference = CausalInference(simpson_model)
    return simp_inference


def get_example_model():
    # Model structure: Z -> X -> Y; Z -> W -> Y
    example_model = DiscreteBayesianNetwork(
        [("X", "Y"), ("Z", "X"), ("Z", "W"), ("W", "Y")]
    )
    cpd_z = TabularCPD(variable="Z", variable_card=2, values=[[0.2], [0.8]])

    cpd_x = TabularCPD(
        variable="X",
        variable_card=2,
        values=[[0.1, 0.3], [0.9, 0.7]],
        evidence=["Z"],
        evidence_card=[2],
    )

    cpd_w = TabularCPD(
        variable="W",
        variable_card=2,
        values=[[0.2, 0.9], [0.8, 0.1]],
        evidence=["Z"],
        evidence_card=[2],
    )

    cpd_y = TabularCPD(
        variable="Y",
        variable_card=2,
        values=[[0.3, 0.4, 0.7, 0.8], [0.7, 0.6, 0.3, 0.2]],
        evidence=["X", "W"],
        evidence_card=[2, 2],
    )

    example_model.add_cpds(cpd_z, cpd_x, cpd_w, cpd_y)

    return example_model


@pytest.fixture
def example_inference():
    example_model = get_example_model()
    example_inference = CausalInference(example_model)
    return example_inference


def get_iv_model():
    # Model structure: Z -> X -> Y; X <- U -> Y
    example_model = DiscreteBayesianNetwork(
        [("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")]
    )
    cpd_z = TabularCPD(variable="Z", variable_card=2, values=[[0.2], [0.8]])
    cpd_u = TabularCPD(variable="U", variable_card=2, values=[[0.7], [0.3]])
    cpd_x = TabularCPD(
        variable="X",
        variable_card=2,
        values=[[0.1, 0.3, 0.2, 0.9], [0.9, 0.7, 0.8, 0.1]],
        evidence=["U", "Z"],
        evidence_card=[2, 2],
    )
    cpd_y = TabularCPD(
        variable="Y",
        variable_card=2,
        values=[[0.5, 0.8, 0.2, 0.7], [0.5, 0.2, 0.8, 0.3]],
        evidence=["U", "X"],
        evidence_card=[2, 2],
    )

    example_model.add_cpds(cpd_z, cpd_u, cpd_x, cpd_y)

    return example_model


@pytest.fixture
def iv_inference():
    iv_model = get_iv_model()
    iv_inference = CausalInference(iv_model)
    return iv_inference


class TestCausalGraphMethods:
    def test_is_d_separated(self, inference):
        assert inference.model.is_dconnected("X", "Y", observed=None)
        assert not inference.model.is_dconnected("B", "Y", observed=("C", "X"))

    def test_backdoor_validation(self, inference, inference_bd, inference_bd2):
        assert inference.is_valid_backdoor_adjustment_set("X", "Y", Z="C")

        # Z accepts str or set[str]
        assert inference_bd.is_valid_backdoor_adjustment_set("X", "Y", Z="Z1")
        assert inference_bd2.is_valid_backdoor_adjustment_set("X", "Y", Z={"Z1", "Z2"})


class TestCausalInferenceInit:
    def test_integer_variable_name(self):
        df = pd.DataFrame([[0, 1], [0, 0]])
        self.model = DiscreteBayesianNetwork(df)
        with pytest.raises(NotImplementedError):
            CausalInference(self.model)


class TestAdjustmentSet:
    def test_proper_backdoor_graph_error(self, infer_dag, infer_sem):
        # DAG
        with pytest.raises(ValueError):
            infer_dag.get_proper_backdoor_graph(
                X=["x3"],
                Y=["y1", "y2"],
            )
        with pytest.raises(ValueError):
            infer_dag.get_proper_backdoor_graph(
                X=["x2"],
                Y=["y1", "y3"],
            )
        with pytest.raises(ValueError):
            infer_dag.get_proper_backdoor_graph(
                X=["x3", "x2"],
                Y=["y1", "y3"],
            )

        # SEMGraph
        with pytest.raises(ValueError):
            infer_sem.get_proper_backdoor_graph(
                X=["x3"],
                Y=["y1", "y2"],
            )
        with pytest.raises(ValueError):
            infer_sem.get_proper_backdoor_graph(
                X=["x2"],
                Y=["y1", "y3"],
            )
        with pytest.raises(ValueError):
            infer_sem.get_proper_backdoor_graph(
                X=["x3", "x2"],
                Y=["y1", "y3"],
            )

    def test_proper_backdoor_graph(self, infer_dag, infer_sem):
        # DAG
        bd_graph = infer_dag.get_proper_backdoor_graph(X=["x1", "x2"], Y=["y1", "y2"])
        assert ("x1", "y1") not in bd_graph.edges()
        assert len(bd_graph.edges()) == 4
        assert set(bd_graph.edges()) == set(
            [("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]
        )

        # SEMGraph
        bd_graph = infer_sem.get_proper_backdoor_graph(X=["x1", "x2"], Y=["y1", "y2"])
        assert ("x1", "y1") not in bd_graph.edges()
        assert len(bd_graph.edges()) == 10
        assert set(bd_graph.edges()) == set(
            [
                ("x1", "z1"),
                ("z1", "z2"),
                ("z2", "x2"),
                ("y2", "z2"),
                (".x1", "x1"),
                (".y1", "y1"),
                (".z1", "z1"),
                (".z2", "z2"),
                (".x2", "x2"),
                (".y2", "y2"),
            ]
        )

    def test_proper_backdoor_graph_not_list(self, infer_dag, infer_sem):
        # DAG
        bd_graph = infer_dag.get_proper_backdoor_graph(X="x1", Y="y1")
        assert ("x1", "y1") not in bd_graph.edges()
        assert len(bd_graph.edges()) == 4
        assert set(bd_graph.edges()) == set(
            [("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]
        )

        # SEMGraph
        bd_graph = infer_sem.get_proper_backdoor_graph(X="x1", Y="y1")
        assert ("x1", "y1") not in bd_graph.edges()
        assert len(bd_graph.edges()) == 10
        assert set(bd_graph.edges()) == set(
            [
                ("x1", "z1"),
                ("z1", "z2"),
                ("z2", "x2"),
                ("y2", "z2"),
                (".x1", "x1"),
                (".y1", "y1"),
                (".z1", "z1"),
                (".z2", "z2"),
                (".x2", "x2"),
                (".y2", "y2"),
            ]
        )

    def test_is_valid_adjustment_set(self, infer_dag, infer_sem):
        # DAG
        assert infer_dag.is_valid_adjustment_set(
            X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1", "z2"]
        )

        assert infer_dag.is_valid_adjustment_set(
            X="x1", Y="y1", adjustment_set=["z1", "z2"]
        )

        assert not infer_dag.is_valid_adjustment_set(
            X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1"]
        )

        assert infer_dag.is_valid_adjustment_set(
            X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z2"]
        )

        # SEMGraph
        assert infer_sem.is_valid_adjustment_set(
            X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1", "z2"]
        )

        assert infer_sem.is_valid_adjustment_set(
            X="x1", Y="y1", adjustment_set=["z1", "z2"]
        )

        assert not infer_sem.is_valid_adjustment_set(
            X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1"]
        )

        assert infer_sem.is_valid_adjustment_set(
            X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z2"]
        )

    def test_get_minimal_adjustment_set(self):
        # Without latent variables
        dag1 = DAG([("X", "Y"), ("Z", "X"), ("Z", "Y")])
        infer = CausalInference(dag1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        assert adj_set == {"Z"}

        with pytest.raises(ValueError):
            infer.get_minimal_adjustment_set(X="W", Y="Y")

        # M graph
        dag2 = DAG([("X", "Y"), ("Z1", "X"), ("Z1", "Z3"), ("Z2", "Z3"), ("Z2", "Y")])
        infer = CausalInference(dag2)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        assert adj_set == set()

        # With latents
        dag_lat1 = DAG([("X", "Y"), ("Z", "X"), ("Z", "Y")], latents={"Z"})
        infer = CausalInference(dag_lat1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        assert adj_set is None

        # Pearl's Simpson machine
        dag_lat2 = DAG(
            [
                ("X", "Y"),
                ("Z1", "U"),
                ("U", "X"),
                ("Z1", "Z3"),
                ("Z3", "Y"),
                ("U", "Z2"),
                ("Z3", "Z2"),
            ],
            latents={"U"},
        )
        infer = CausalInference(dag_lat2)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        assert (adj_set == {"Z1"}) or (adj_set == {"Z3"})

    def test_get_minimal_adjustment_set_sem(self):
        # Without latent variables
        dag1 = SEMGraph([("X", "Y"), ("Z", "X"), ("Z", "Y")])
        infer = CausalInference(dag1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        assert adj_set == {"Z"}

        with pytest.raises(ValueError):
            infer.get_minimal_adjustment_set(X="W", Y="Y")

        # M graph
        dag2 = SEMGraph(
            [("X", "Y"), ("Z1", "X"), ("Z1", "Z3"), ("Z2", "Z3"), ("Z2", "Y")]
        )
        infer = CausalInference(dag2)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        assert adj_set == set()

        # With latents
        dag_lat1 = SEMGraph([("X", "Y"), ("Z", "X"), ("Z", "Y")], latents={"Z"})
        infer = CausalInference(dag_lat1)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        assert adj_set is None

        # Pearl's Simpson machine
        dag_lat2 = SEMGraph(
            [
                ("X", "Y"),
                ("Z1", "U"),
                ("U", "X"),
                ("Z1", "Z3"),
                ("Z3", "Y"),
                ("U", "Z2"),
                ("Z3", "Z2"),
            ],
            latents={"U"},
        )
        infer = CausalInference(dag_lat2)
        adj_set = infer.get_minimal_adjustment_set(X="X", Y="Y")
        assert (adj_set == {"Z1"}) or (adj_set == {"Z3"})

    def test_issue_1710(self):
        # DAG
        dag = DAG([("X_1", "X_2"), ("Z", "X_1"), ("Z", "X_2")])
        infer = CausalInference(dag)
        adj_set = infer.get_minimal_adjustment_set("X_1", "X_2")

        assert adj_set == {"Z"}
        with pytest.raises(ValueError):
            infer.get_minimal_adjustment_set(X="X_3", Y="Y")

        # SEM
        sem = SEMGraph([("X_1", "X_2"), ("Z", "X_1"), ("Z", "X_2")])
        infer = CausalInference(sem)
        adj_set = infer.get_minimal_adjustment_set("X_1", "X_2")

        assert adj_set == {"Z"}
        with pytest.raises(ValueError):
            infer.get_minimal_adjustment_set(X="X_3", Y="Y")


class TestBackdoorPaths:
    """
    These tests are drawn from games presented in The Book of Why by Judea Pearl. See the Jupyter Notebook called
    Causal Games in the examples folder for further explanation about each of these.
    """

    def test_game1_bn(self):
        game1 = DiscreteBayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])
        inference = CausalInference(game1)
        assert inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset()

    def test_game1_sem(self):
        game1 = SEMGraph(ebunch=[("X", "A"), ("A", "Y"), ("A", "B")])
        inference = CausalInference(game1)
        assert inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset()

    def test_game2_bn(self):
        game2 = DiscreteBayesianNetwork(
            [
                ("X", "E"),
                ("E", "Y"),
                ("A", "B"),
                ("A", "X"),
                ("B", "C"),
                ("D", "B"),
                ("D", "E"),
            ]
        )
        inference = CausalInference(game2)
        assert inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset()

    def test_game2_sem(self):
        game2 = SEMGraph(
            [
                ("X", "E"),
                ("E", "Y"),
                ("A", "B"),
                ("A", "X"),
                ("B", "C"),
                ("D", "B"),
                ("D", "E"),
            ]
        )
        inference = CausalInference(game2)
        assert inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset()

    def test_game3_bn(self):
        game3 = DiscreteBayesianNetwork(
            [("X", "Y"), ("X", "A"), ("B", "A"), ("B", "Y"), ("B", "X")]
        )
        inference = CausalInference(game3)
        assert not inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset({frozenset({"B"})})

    def test_game3_sem(self):
        game3 = SEMGraph([("X", "Y"), ("X", "A"), ("B", "A"), ("B", "Y"), ("B", "X")])
        inference = CausalInference(game3)
        assert not inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset({frozenset({"B"})})

    def test_game4_bn(self):
        game4 = DiscreteBayesianNetwork(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y")]
        )
        inference = CausalInference(game4)
        assert inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset()

    def test_game4_sem(self):
        game4 = SEMGraph([("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y")])
        inference = CausalInference(game4)
        assert inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset()

    def test_game5_bn(self):
        game5 = DiscreteBayesianNetwork(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        inference = CausalInference(game5)
        assert not inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset({frozenset({"C"}), frozenset({"A", "B"})})

    def test_game5_sem(self):
        game5 = SEMGraph(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")]
        )
        inference = CausalInference(game5)
        assert not inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset({frozenset({"C"}), frozenset({"A", "B"})})

    def test_game6_bn(self):
        game6 = DiscreteBayesianNetwork(
            [
                ("X", "F"),
                ("C", "X"),
                ("A", "C"),
                ("A", "D"),
                ("B", "D"),
                ("B", "E"),
                ("D", "X"),
                ("D", "Y"),
                ("E", "Y"),
                ("F", "Y"),
            ]
        )
        inference = CausalInference(game6)
        assert not inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset(
            {
                frozenset({"C", "D"}),
                frozenset({"A", "D"}),
                frozenset({"D", "E"}),
                frozenset({"B", "D"}),
            }
        )

    def test_game6_sem(self):
        game6 = SEMGraph(
            [
                ("X", "F"),
                ("C", "X"),
                ("A", "C"),
                ("A", "D"),
                ("B", "D"),
                ("B", "E"),
                ("D", "X"),
                ("D", "Y"),
                ("E", "Y"),
                ("F", "Y"),
            ]
        )
        inference = CausalInference(game6)
        assert not inference.is_valid_backdoor_adjustment_set("X", "Y")
        deconfounders = inference.get_all_backdoor_adjustment_sets("X", "Y")
        assert deconfounders == frozenset(
            {
                frozenset({"C", "D"}),
                frozenset({"A", "D"}),
                frozenset({"D", "E"}),
                frozenset({"B", "D"}),
            }
        )


class TestSEMIdentification:
    def test_get_scaling_indicators(
        self, demo_inference, union_inference, custom_inference
    ):
        demo_scaling_indicators = demo_inference.get_scaling_indicators()
        assert demo_scaling_indicators["eta1"] in ["y1", "y2", "y3", "y4"]
        assert demo_scaling_indicators["eta2"] in ["y5", "y6", "y7", "y8"]
        assert demo_scaling_indicators["xi1"] in ["x1", "x2", "x3"]

        union_scaling_indicators = union_inference.get_scaling_indicators()
        assert union_scaling_indicators == dict()

        custom_scaling_indicators = custom_inference.get_scaling_indicators()
        assert custom_scaling_indicators["xi1"] in ["x1", "x2", "y1", "y4"]
        assert custom_scaling_indicators["eta1"] in ["y2", "y3"]
        assert custom_scaling_indicators["eta2"] in ["y5"]

    def test_iv_transformations_demo(self, demo_inference):
        scale = {"eta1": "y1", "eta2": "y5", "xi1": "x1"}

        with pytest.raises(ValueError):
            demo_inference._iv_transformations("x1", "y1", scale)

        for y in ["y2", "y3", "y4"]:
            full_graph, dependent_var = demo_inference._iv_transformations(
                X="eta1", Y=y, scaling_indicators=scale
            )
            assert dependent_var == y
            assert (".y1", y) in full_graph.edges
            assert ("eta1", y) not in full_graph.edges

        for y in ["y6", "y7", "y8"]:
            full_graph, dependent_var = demo_inference._iv_transformations(
                X="eta2", Y=y, scaling_indicators=scale
            )
            assert dependent_var == y
            assert (".y5", y) in full_graph.edges
            assert ("eta2", y) not in full_graph.edges

        full_graph, dependent_var = demo_inference._iv_transformations(
            X="xi1", Y="eta1", scaling_indicators=scale
        )
        assert dependent_var == "y1"
        assert (".eta1", "y1") in full_graph.edges()
        assert (".x1", "y1") in full_graph.edges()
        assert ("xi1", "eta1") not in full_graph.edges()

        full_graph, dependent_var = demo_inference._iv_transformations(
            X="xi1", Y="eta2", scaling_indicators=scale
        )
        assert dependent_var == "y5"
        assert (".y1", "y5") in full_graph.edges()
        assert (".eta2", "y5") in full_graph.edges()
        assert (".x1", "y5") in full_graph.edges()
        assert ("eta1", "eta2") not in full_graph.edges()
        assert ("xi1", "eta2") not in full_graph.edges()

        full_graph, dependent_var = demo_inference._iv_transformations(
            X="eta1", Y="eta2", scaling_indicators=scale
        )
        assert dependent_var == "y5"
        assert (".y1", "y5") in full_graph.edges()
        assert (".eta2", "y5") in full_graph.edges()
        assert (".x1", "y5") in full_graph.edges()
        assert ("eta1", "eta2") not in full_graph.edges()
        assert ("xi1", "eta2") not in full_graph.edges()

    def test_iv_transformations_union(self, union_inference):
        scale = {}
        for u, v in [
            ("yrsmill", "unionsen"),
            ("age", "laboract"),
            ("age", "deferenc"),
            ("deferenc", "laboract"),
            ("deferenc", "unionsen"),
            ("laboract", "unionsen"),
        ]:
            full_graph, dependent_var = union_inference._iv_transformations(
                u, v, scaling_indicators=scale
            )
            assert (u, v) not in full_graph.edges()
            assert dependent_var == v

    def test_get_ivs_demo(self, demo_inference):
        scale = {"eta1": "y1", "eta2": "y5", "xi1": "x1"}

        assert demo_inference.get_ivs("eta1", "y2", scaling_indicators=scale) == {
            "x1",
            "x2",
            "x3",
            "y3",
            "y7",
            "y8",
        }
        assert demo_inference.get_ivs("eta1", "y3", scaling_indicators=scale) == {
            "x1",
            "x2",
            "x3",
            "y2",
            "y4",
            "y6",
            "y8",
        }
        assert demo_inference.get_ivs("eta1", "y4", scaling_indicators=scale) == {
            "x1",
            "x2",
            "x3",
            "y3",
            "y6",
            "y7",
        }

        assert demo_inference.get_ivs("eta2", "y6", scaling_indicators=scale) == {
            "x1",
            "x2",
            "x3",
            "y3",
            "y4",
            "y7",
        }
        assert demo_inference.get_ivs("eta2", "y7", scaling_indicators=scale) == {
            "x1",
            "x2",
            "x3",
            "y2",
            "y4",
            "y6",
            "y8",
        }
        assert demo_inference.get_ivs("eta2", "y8", scaling_indicators=scale) == {
            "x1",
            "x2",
            "x3",
            "y2",
            "y3",
            "y7",
        }

        assert demo_inference.get_ivs("xi1", "x2", scaling_indicators=scale) == {
            "x3",
            "y1",
            "y2",
            "y3",
            "y4",
            "y5",
            "y6",
            "y7",
            "y8",
        }
        assert demo_inference.get_ivs("xi1", "x3", scaling_indicators=scale) == {
            "x2",
            "y1",
            "y2",
            "y3",
            "y4",
            "y5",
            "y6",
            "y7",
            "y8",
        }

        assert demo_inference.get_ivs("xi1", "eta1", scaling_indicators=scale) == {
            "x2",
            "x3",
        }
        assert demo_inference.get_ivs("xi1", "eta2", scaling_indicators=scale) == {
            "x2",
            "x3",
            "y2",
            "y3",
            "y4",
        }
        assert demo_inference.get_ivs("eta1", "eta2", scaling_indicators=scale) == {
            "x2",
            "x3",
            "y2",
            "y3",
            "y4",
        }

    def test_get_conditional_ivs_demo(self, demo_inference):
        scale = {"eta1": "y1", "eta2": "y5", "xi1": "x1"}

        assert (
            demo_inference.get_conditional_ivs("eta1", "y2", scaling_indicators=scale)
            == []
        )
        assert (
            demo_inference.get_conditional_ivs("eta1", "y3", scaling_indicators=scale)
            == []
        )
        assert (
            demo_inference.get_conditional_ivs("eta1", "y4", scaling_indicators=scale)
            == []
        )

        assert (
            demo_inference.get_conditional_ivs("eta2", "y6", scaling_indicators=scale)
            == []
        )
        assert (
            demo_inference.get_conditional_ivs("eta2", "y7", scaling_indicators=scale)
            == []
        )
        assert (
            demo_inference.get_conditional_ivs("eta2", "y8", scaling_indicators=scale)
            == []
        )

        assert (
            demo_inference.get_conditional_ivs("xi1", "x2", scaling_indicators=scale)
            == []
        )
        assert (
            demo_inference.get_conditional_ivs("xi1", "x3", scaling_indicators=scale)
            == []
        )

        assert (
            demo_inference.get_conditional_ivs("xi1", "eta1", scaling_indicators=scale)
            == []
        )
        assert (
            demo_inference.get_conditional_ivs("xi1", "eta2", scaling_indicators=scale)
            == []
        )
        assert (
            demo_inference.get_conditional_ivs("eta1", "eta2", scaling_indicators=scale)
            == []
        )

    def test_get_ivs_union(self, union_inference):
        scale = {}
        assert (
            union_inference.get_ivs("yrsmill", "unionsen", scaling_indicators=scale)
            == set()
        )
        assert (
            union_inference.get_ivs("deferenc", "unionsen", scaling_indicators=scale)
            == set()
        )
        assert (
            union_inference.get_ivs("laboract", "unionsen", scaling_indicators=scale)
            == set()
        )
        assert (
            union_inference.get_ivs("deferenc", "laboract", scaling_indicators=scale)
            == set()
        )
        assert union_inference.get_ivs("age", "laboract", scaling_indicators=scale) == {
            "yrsmill"
        }
        assert union_inference.get_ivs("age", "deferenc", scaling_indicators=scale) == {
            "yrsmill"
        }

    def test_get_conditional_ivs_union(self, union_inference):
        assert union_inference.get_conditional_ivs("yrsmill", "unionsen") == [
            ("age", {"laboract", "deferenc"})
        ]
        # This case wouldn't have conditonal IV if the Total effect between `deferenc` and
        # `unionsen` needs to be computed because one of the conditional variable lies on the
        # effect path.
        assert union_inference.get_conditional_ivs("deferenc", "unionsen") == [
            ("age", {"yrsmill", "laboract"})
        ]
        assert union_inference.get_conditional_ivs("laboract", "unionsen") == [
            ("age", {"yrsmill", "deferenc"})
        ]
        assert union_inference.get_conditional_ivs("deferenc", "laboract") == []

        assert union_inference.get_conditional_ivs("age", "laboract") == [
            ("yrsmill", {"deferenc"})
        ]
        assert union_inference.get_conditional_ivs("age", "deferenc") == []

    def test_total_conditional_ivs_union(self, union_inference):
        assert union_inference.get_total_conditional_ivs("deferenc", "unionsen") == []

    def test_iv_transformations_custom(self, custom_inference):
        scale_custom = {"eta1": "y2", "eta2": "y5", "xi1": "x1"}

        full_graph, var = custom_inference._iv_transformations(
            "xi1", "x2", scaling_indicators=scale_custom
        )
        assert var == "x2"
        assert (".x1", "x2") in full_graph.edges()
        assert ("xi1", "x2") not in full_graph.edges()

        full_graph, var = custom_inference._iv_transformations(
            "xi1", "y4", scaling_indicators=scale_custom
        )
        assert var == "y4"
        assert (".x1", "y4") in full_graph.edges()
        assert ("xi1", "y4") not in full_graph.edges()

        full_graph, var = custom_inference._iv_transformations(
            "xi1", "y1", scaling_indicators=scale_custom
        )
        assert var == "y1"
        assert (".x1", "y1") in full_graph.edges()
        assert ("xi1", "y1") not in full_graph.edges()
        assert ("y4", "y1") not in full_graph.edges()

        full_graph, var = custom_inference._iv_transformations(
            "xi1", "eta1", scaling_indicators=scale_custom
        )
        assert var == "y2"
        assert (".eta1", "y2") in full_graph.edges()
        assert (".x1", "y2") in full_graph.edges()
        assert ("y1", "eta1") not in full_graph.edges()
        assert ("xi1", "eta1") not in full_graph.edges()

        full_graph, var = custom_inference._iv_transformations(
            "y1", "eta1", scaling_indicators=scale_custom
        )
        assert var == "y2"
        assert (".eta1", "y2") in full_graph.edges()
        assert (".x1", "y2") in full_graph.edges()
        assert ("y1", "eta1") not in full_graph.edges()
        assert ("xi1", "eta1") not in full_graph.edges()

        full_graph, var = custom_inference._iv_transformations(
            "y1", "eta2", scaling_indicators=scale_custom
        )
        assert var == "y5"
        assert (".eta2", "y5") in full_graph.edges()
        assert ("y1", "eta2") not in full_graph.edges()

        full_graph, var = custom_inference._iv_transformations(
            "y4", "y1", scaling_indicators=scale_custom
        )
        assert var == "y1"
        assert ("y4", "y1") not in full_graph.edges()

        full_graph, var = custom_inference._iv_transformations(
            "eta1", "y3", scaling_indicators=scale_custom
        )
        assert var == "y3"
        assert (".y2", "y3") in full_graph.edges()
        assert ("eta1", "y3") not in full_graph.edges()

    def test_get_ivs_custom(self, custom_inference):
        scale_custom = {"eta1": "y2", "eta2": "y5", "xi1": "x1"}

        assert custom_inference.get_ivs(
            "xi1", "x2", scaling_indicators=scale_custom
        ) == {
            "y1",
            "y2",
            "y3",
            "y4",
            "y5",
        }
        assert custom_inference.get_ivs(
            "xi1", "y4", scaling_indicators=scale_custom
        ) == {"x2"}
        assert custom_inference.get_ivs(
            "xi1", "y1", scaling_indicators=scale_custom
        ) == {
            "x2",
            "y4",
        }
        assert custom_inference.get_ivs(
            "xi1", "eta1", scaling_indicators=scale_custom
        ) == {
            "x2",
            "y4",
        }
        # TODO: Test this and fix.
        assert custom_inference.get_ivs(
            "y1", "eta1", scaling_indicators=scale_custom
        ) == {
            "x2",
            "y4",
            "y5",
        }
        assert custom_inference.get_ivs(
            "y1", "eta2", scaling_indicators=scale_custom
        ) == {
            "x1",
            "x2",
            "y2",
            "y3",
            "y4",
        }
        assert (
            custom_inference.get_ivs("y4", "y1", scaling_indicators=scale_custom)
            == set()
        )
        assert custom_inference.get_ivs(
            "eta1", "y3", scaling_indicators=scale_custom
        ) == {
            "x1",
            "x2",
            "y4",
        }

    def test_small_model_ivs(self):
        model1 = SEMGraph(
            ebunch=[("X", "Y"), ("I", "X"), ("W", "I")],
            latents=[],
            err_corr=[("W", "Y")],
            err_var={},
        )
        inference1 = CausalInference(model1)
        assert inference1.get_conditional_ivs("X", "Y") == [("I", {"W"})]

        model2 = SEMGraph(
            ebunch=[
                ("x", "y"),
                ("z", "x"),
                ("w", "z"),
                ("w", "u"),
                ("u", "x"),
                ("u", "y"),
            ],
            latents=["u"],
        )
        inference2 = CausalInference(model2)
        assert inference2.get_conditional_ivs("x", "y") == [("z", {"w"})]

        model3 = SEMGraph(
            ebunch=[("x", "y"), ("u", "x"), ("u", "y"), ("z", "x")], latents=["u"]
        )
        inference3 = CausalInference(model3)
        assert inference3.get_ivs("x", "y") == {"z"}

        model4 = SEMGraph(ebunch=[("x", "y"), ("z", "x"), ("u", "x"), ("u", "y")])
        inference4 = CausalInference(model4)
        assert inference4.get_conditional_ivs("x", "y") == [("z", {"u"})]


class TestBayesianIV:
    def test_get_ivs(self, custom_inference2):
        ivs = custom_inference2.get_ivs("X", "Y")
        assert "Z" in ivs

    def test_get_conditional_ivs(self, custom_inference2):
        custom_inference2.model.add_edge("I", "X")
        custom_inference2.model.add_edge("W", "I")
        custom_inference2.model.add_edge("W", "Y")
        causal_inf = CausalInference(custom_inference2.model)
        cond_ivs = causal_inf.get_conditional_ivs("X", "Y")
        assert ("I", {"W"}) in cond_ivs

    def test_identification_method(self):
        backdoor_model = DiscreteBayesianNetwork(
            ebunch=[("X", "Y"), ("M", "Y"), ("M", "X")]
        )
        causal_inf = CausalInference(backdoor_model)
        methods = causal_inf.identification_method("X", "Y")
        expected_backdoor = {"backdoor set": {frozenset({"M"})}}
        assert methods == expected_backdoor

        frontdoor_model = DiscreteBayesianNetwork(ebunch=[("X", "M"), ("M", "Y")])
        causal_inf = CausalInference(frontdoor_model)
        methods = causal_inf.identification_method("X", "Y")
        expected_frontdoor = {"frontdoor set": {frozenset({"M"})}}
        assert methods == expected_frontdoor

        iv_model = DiscreteBayesianNetwork(
            ebunch=[("Z", "X"), ("X", "Y"), ("U", "Y"), ("U", "X")], latents=["U"]
        )
        causal_inf = CausalInference(iv_model)
        methods = causal_inf.identification_method("X", "Y")
        expected_iv = {"instrumental variables": {"Z"}}
        assert methods == expected_iv


class TestDoQuery:
    def test_query(self, simp_inference, iv_inference):
        for algo in ["ve", "bp"]:
            # Simpson model queries
            query_nodo1 = simp_inference.query(
                variables=["C"], do=None, evidence={"T": 1}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query_nodo1.values, np.array([0.5, 0.5]))

            query_nodo2 = simp_inference.query(
                variables=["C"], do=None, evidence={"T": 0}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query_nodo2.values, np.array([0.6, 0.4]))

            query1 = simp_inference.query(
                variables=["C"], do={"T": 1}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query1.values, np.array([0.6, 0.4]))

            query2 = simp_inference.query(
                variables=["C"], do={"T": 0}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query2.values, np.array([0.5, 0.5]))

            query3 = simp_inference.query(["C"], adjustment_set=["S"])
            np_test.assert_array_almost_equal(query3.values, np.array([0.55, 0.45]))

            # IV model queries
            query_nodo1 = iv_inference.query(["Z"], do=None, inference_algo=algo)
            np_test.assert_array_almost_equal(query_nodo1.values, np.array([0.2, 0.8]))

            query_nodo2 = iv_inference.query(["X"], do=None, evidence={"Z": 1})
            np_test.assert_array_almost_equal(
                query_nodo2.values, np.array([0.48, 0.52])
            )

            query1 = iv_inference.query(["X"], do={"Z": 1})
            np_test.assert_array_almost_equal(query1.values, np.array([0.48, 0.52]))

            query2 = iv_inference.query(["Y"], do={"X": 1})
            np_test.assert_array_almost_equal(query2.values, np.array([0.77, 0.23]))

            query3 = iv_inference.query(["Y"], do={"X": 1}, adjustment_set={"U"})
            np_test.assert_array_almost_equal(query3.values, np.array([0.77, 0.23]))

    def test_adjustment_query(self, example_inference):
        for algo in ["ve", "bp"]:
            # Test adjustment with do operation.
            query1 = example_inference.query(
                variables=["Y"], do={"X": 1}, adjustment_set={"Z"}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query1.values, np.array([0.7240, 0.2760]))

            query2 = example_inference.query(
                variables=["Y"], do={"X": 1}, adjustment_set={"W"}, inference_algo=algo
            )
            np_test.assert_array_almost_equal(query2.values, np.array([0.7240, 0.2760]))

            # Test adjustment without do operation.
            query3 = example_inference.query(["Y"], adjustment_set=["W"])
            np_test.assert_array_almost_equal(query3.values, np.array([0.62, 0.38]))

            query4 = example_inference.query(["Y"], adjustment_set=["Z"])
            np_test.assert_array_almost_equal(query4.values, np.array([0.62, 0.38]))

            query5 = example_inference.query(["Y"], adjustment_set=["W", "Z"])
            np_test.assert_array_almost_equal(query5.values, np.array([0.62, 0.38]))

    def test_issue_1459(self):
        bn = DiscreteBayesianNetwork([("X", "Y"), ("W", "X"), ("W", "Y")])
        cpd_w = TabularCPD(variable="W", variable_card=2, values=[[0.7], [0.3]])
        cpd_x = TabularCPD(
            variable="X",
            variable_card=2,
            values=[[0.7, 0.4], [0.3, 0.6]],
            evidence=["W"],
            evidence_card=[2],
        )
        cpd_y = TabularCPD(
            variable="Y",
            variable_card=2,
            values=[[0.7, 0.7, 0.5, 0.1], [0.3, 0.3, 0.5, 0.9]],
            evidence=["W", "X"],
            evidence_card=[2, 2],
        )

        bn.add_cpds(cpd_w, cpd_x, cpd_y)
        causal_infer = CausalInference(bn)
        query = causal_infer.query(["Y"], do={"X": 1}, evidence={"W": 1})
        np_test.assert_array_almost_equal(query.values, np.array([0.1, 0.9]))

        # A slight modified version of the above model where only some of the adjustment
        # set variables are in evidence.
        bn = DiscreteBayesianNetwork(
            [("X", "Y"), ("W1", "X"), ("W1", "Y"), ("W2", "X"), ("W2", "Y")]
        )
        cpd_w1 = TabularCPD(variable="W1", variable_card=2, values=[[0.7], [0.3]])
        cpd_w2 = TabularCPD(variable="W2", variable_card=2, values=[[0.3], [0.7]])
        cpd_x = TabularCPD(
            variable="X",
            variable_card=2,
            values=[[0.7, 0.4, 0.3, 0.8], [0.3, 0.6, 0.7, 0.2]],
            evidence=["W1", "W2"],
            evidence_card=[2, 2],
        )
        cpd_y = TabularCPD(
            variable="Y",
            variable_card=2,
            values=[
                [0.7, 0.7, 0.5, 0.1, 0.9, 0.2, 0.4, 0.6],
                [0.3, 0.3, 0.5, 0.9, 0.1, 0.8, 0.6, 0.4],
            ],
            evidence=["W1", "W2", "X"],
            evidence_card=[2, 2, 2],
        )
        bn.add_cpds(cpd_w1, cpd_w2, cpd_x, cpd_y)
        causal_infer = CausalInference(bn)
        query = causal_infer.query(["Y"], do={"X": 1}, evidence={"W1": 1})
        np_test.assert_array_almost_equal(query.values, np.array([0.48, 0.52]))

    def test_query_error(self, simp_inference):
        with pytest.raises(ValueError):
            simp_inference.query(variables="C", do={"T": 1})
        with pytest.raises(ValueError):
            simp_inference.query(variables=["E"], do={"T": 1})
        with pytest.raises(ValueError):
            simp_inference.query(variables=["C"], do="T")
        with pytest.raises(ValueError):
            simp_inference.query(
                variables=["C"],
                do={"T": 1},
                evidence="S",
            )
        with pytest.raises(ValueError):
            simp_inference.query(
                variables=["C"],
                do={"T": 1},
                inference_algo="random",
            )

    def test_invalid_causal_query_direct_descendant_intervention(self):
        # Model: R -> S -> W & R -> W. We intervene on S and query R.
        model = DiscreteBayesianNetwork([("R", "W"), ("S", "W"), ("R", "S")])
        cpd_rain = TabularCPD(
            variable="R",
            variable_card=2,
            values=[[0.6], [0.4]],
            state_names={"R": ["True", "False"]},
        )
        cpd_sprinkler = TabularCPD(
            variable="S",
            variable_card=2,
            values=[[0.1, 0.5], [0.9, 0.5]],
            evidence=["R"],
            evidence_card=[2],
            state_names={"S": ["True", "False"], "R": ["True", "False"]},
        )
        cpd_wet_grass = TabularCPD(
            variable="W",
            variable_card=2,
            values=[[0.99, 0.9, 0.9, 0.01], [0.01, 0.1, 0.1, 0.99]],
            evidence=["R", "S"],
            evidence_card=[2, 2],
            state_names={
                "W": ["True", "False"],
                "R": ["True", "False"],
                "S": ["True", "False"],
            },
        )
        model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet_grass)
        causal_inference = CausalInference(model)

        evidence = {"W": "True"}
        counterfactual_intervention = {"S": "False"}
        with pytest.raises(ValueError) as cm:
            causal_inference.query(
                variables=["R"], evidence=evidence, do=counterfactual_intervention
            )
        assert (
            "Invalid causal query: There is a direct edge from the query variable 'R' to the intervention variable 'S'."
            in str(cm.value)
        )


class TestEstimator:
    def test_create_estimator(self):
        game1 = DiscreteBayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])
        data = pd.DataFrame(
            np.random.randint(2, size=(1000, 4)), columns=["X", "A", "B", "Y"]
        )
        inference = CausalInference(model=game1)
        ate = inference.estimate_ate("X", "Y", data=data, estimator_type="linear")
        assert ate == pytest.approx(0, abs=0.1)

    def test_estimate_frontdoor(self):
        model = DiscreteBayesianNetwork(
            [("X", "Z"), ("Z", "Y"), ("U", "X"), ("U", "Y")], latents=["U"]
        )
        U = np.random.randn(10000)
        X = 0.3 * U + np.random.randn(10000)
        Z = 0.8 * X + 0.3 * np.random.randn(10000)
        Y = 0.5 * U + 0.9 * Z + 0.4 * np.random.randn(10000)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        infer = CausalInference(model=model)
        ate = infer.estimate_ate("X", "Y", data=data, estimator_type="linear")
        assert ate == pytest.approx(0.8 * 0.9, abs=0.1)

    def test_estimate_fail_no_adjustment(self):
        model = DiscreteBayesianNetwork(
            [("X", "Y"), ("U", "X"), ("U", "Y")], latents=["U"]
        )

        U = np.random.randn(10000)
        X = 0.3 * U + np.random.randn(10000)
        Z = 0.8 * X + 0.3 * np.random.randn(10000)
        Y = 0.5 * U + 0.9 * Z + 0.4 * np.random.randn(10000)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        infer = CausalInference(model=model)
        with pytest.raises(ValueError):
            infer.estimate_ate("X", "Y", data)

    def test_estimate_multiple_paths(self):
        model = DiscreteBayesianNetwork(
            [("X", "Z"), ("U", "X"), ("U", "Y"), ("Z", "Y"), ("X", "P1"), ("P1", "Y")],
            latents=["U"],
        )

        U = np.random.randn(10000)
        X = 0.3 * U + np.random.randn(10000)
        P1 = 0.9 * X + np.random.randn(10000)
        Z = 0.8 * X + 0.3 * np.random.randn(10000)
        Y = 0.5 * U + 0.9 * Z + 0.1 * P1 + 0.4 * np.random.randn(10000)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z, "P1": P1})

        infer = CausalInference(model=model)
        assert infer.estimate_ate("X", "Y", data) == pytest.approx(
            (0.8 * 0.9) + (0.9 * 0.1), abs=0.1
        )
