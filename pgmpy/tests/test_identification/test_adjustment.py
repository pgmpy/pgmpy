import pytest

from pgmpy.base import DAG
from pgmpy.identification import Adjustment


@pytest.fixture
def dag_multiple_exposure_outcome():
    # Model example taken from "Constructing Separators and Adjustment Sets in
    # Ancestral Graphs" (UAI 2014).
    return DAG(
        [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")],
        roles={"exposure": {"x1", "x2"}, "outcome": {"y1", "y2"}},
    )


@pytest.fixture
def dag_single_exposure_outcome():
    return DAG(
        [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")],
        roles={"exposure": "x1", "outcome": "y1"},
    )


def test_proper_backdoor_graph(
    dag_single_exposure_outcome, dag_multiple_exposure_outcome
):
    backdoor_dag = Adjustment(variant="minimal")._get_proper_backdoor_graph(
        dag_single_exposure_outcome
    )

    assert ("x1", "y1") not in backdoor_dag.edges()
    assert set(backdoor_dag.edges()) == {
        ("x1", "z1"),
        ("z1", "z2"),
        ("z2", "x2"),
        ("y2", "z2"),
    }

    backdoor_dag = Adjustment(variant="minimal")._get_proper_backdoor_graph(
        dag_multiple_exposure_outcome
    )

    assert set(backdoor_dag.edges()) == {
        ("x1", "z1"),
        ("z1", "z2"),
        ("z2", "x2"),
        ("y2", "z2"),
    }
    assert set(backdoor_dag.nodes()) == {"x1", "x2", "y1", "y2", "z1", "z2"}
    assert ("x1", "y1") not in backdoor_dag.edges()


def test_is_valid_adjustment_set():
    dag = DAG(
        [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")],
        roles={
            "exposure": {"x1", "x2"},
            "outcome": {"y1", "y2"},
            "adjustment": {"z1", "z2"},
        },
    )
    assert Adjustment(variant="minimal").validate(dag)

    dag = DAG(
        [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")],
        roles={"exposure": "x1", "outcome": "y1", "adjustment": {"z1", "z2"}},
    )
    assert Adjustment(variant="minimal").validate(dag)

    dag = DAG(
        [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")],
        roles={"exposure": {"x1", "x2"}, "outcome": {"y1", "y2"}, "adjustment": {"z1"}},
    )
    assert not Adjustment(variant="minimal").validate(dag)

    dag = DAG(
        [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")],
        roles={"exposure": {"x1", "x2"}, "outcome": {"y1", "y2"}, "adjustment": {"z2"}},
    )
    assert Adjustment(variant="minimal").validate(dag)


def test_get_minimal_adjustment_set():
    # Without latent variables
    dag1 = DAG(
        [("X", "Y"), ("Z", "X"), ("Z", "Y")], roles={"exposure": "X", "outcome": "Y"}
    )
    dag1_iden, success = Adjustment(variant="minimal").identify(dag1)
    assert success
    assert dag1_iden.get_role("adjustment") == ["Z"]

    # M graph
    dag2 = DAG(
        [("X", "Y"), ("Z1", "X"), ("Z1", "Z3"), ("Z2", "Z3"), ("Z2", "Y")],
        roles={"exposure": "X", "outcome": "Y"},
    )
    dag2_iden, success = Adjustment(variant="minimal").identify(dag2)
    assert success
    assert dag2_iden.get_role("adjustment") == []

    # With latents
    dag_lat1 = DAG(
        [("X", "Y"), ("Z", "X"), ("Z", "Y")],
        latents={"Z"},
        roles={"exposure": "X", "outcome": "Y"},
    )
    dag_lat1_iden, success = Adjustment(variant="minimal").identify(dag_lat1)
    assert not success
    assert dag_lat1_iden == dag_lat1

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
        roles={"exposure": "X", "outcome": "Y"},
    )
    dag_lat2_iden, success = Adjustment(variant="minimal").identify(dag_lat2)
    assert success
    assert set(dag_lat2_iden.get_role("adjustment")) in ({"Z1"}, {"Z3"})


class TestBackdoorPaths:
    """
    These tests are drawn from games presented in The Book of Why by Judea Pearl. See the Jupyter Notebook called
    Causal Games in the examples folder for further explanation about each of these.
    """

    def test_game1_bn(self):
        game1 = DAG(
            [("X", "A"), ("A", "Y"), ("A", "B")],
            roles={"exposure": "X", "outcome": "Y"},
        )
        game1_adj, success = Adjustment(variant="minimal").identify(game1)
        assert success
        assert game1_adj.get_role_dict() == {"exposure": ["X"], "outcome": ["Y"]}

    def test_game2_bn(self):
        game2 = DAG(
            [
                ("X", "E"),
                ("E", "Y"),
                ("A", "B"),
                ("A", "X"),
                ("B", "C"),
                ("D", "B"),
                ("D", "E"),
            ],
            roles={"exposure": "X", "outcome": "Y"},
        )

        success = Adjustment(variant="all").validate(game2)
        assert success

        game2_adjs, success = Adjustment(variant="minimal").identify(game2)
        assert success
        assert game2_adjs.get_role_dict() == {"exposure": ["X"], "outcome": ["Y"]}

    def test_game3_bn(self):
        game3 = DAG(
            [("X", "Y"), ("X", "A"), ("B", "A"), ("B", "Y"), ("B", "X")],
            roles={"exposure": "X", "outcome": "Y"},
        )
        game3_adj, success = Adjustment(variant="minimal").identify(game3)
        assert success
        assert game3_adj.get_role_dict() == {
            "exposure": ["X"],
            "outcome": ["Y"],
            "adjustment": ["B"],
        }

    def test_game4_bn(self):
        game4 = DAG(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y")],
            roles={"exposure": "X", "outcome": "Y"},
        )
        game4_adj, success = Adjustment(variant="minimal").identify(game4)
        assert success
        assert game4_adj.get_role_dict() == {"exposure": ["X"], "outcome": ["Y"]}

    def test_game5_bn(self):
        game5 = DAG(
            [("A", "X"), ("A", "B"), ("C", "B"), ("C", "Y"), ("X", "Y"), ("B", "X")],
            roles={"exposure": "X", "outcome": "Y"},
        )
        game5_adj, success = Adjustment(variant="minimal").identify(game5)
        assert success
        assert game5_adj.get_role("adjustment") == ["C"] or set(
            game5_adj.get_role("adjustment")
        ) == {"A", "B"}

        game5_adj_all, success = Adjustment(variant="all").identify(game5)
        assert success
        assert len(game5_adj_all) == 5
        potential_adjustment_sets = [
            {"C"},
            {"A", "B"},
            {"A", "C"},
            {"B", "C"},
            {"A", "B", "C"},
        ]
        for adj_model in game5_adj_all:
            assert set(adj_model.get_role("adjustment")) in potential_adjustment_sets

    def test_game6_bn(self):
        game6 = DAG(
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
            ],
            roles={"exposure": "X", "outcome": "Y"},
        )
        game6_adj, success = Adjustment(variant="minimal").identify(game6)
        assert success
        adjustment_vars = game6_adj.get_role("adjustment")
        assert "D" in adjustment_vars
        assert (
            "C" in adjustment_vars
            or "A" in adjustment_vars
            or "B" in adjustment_vars
            or "E" in adjustment_vars
        )
