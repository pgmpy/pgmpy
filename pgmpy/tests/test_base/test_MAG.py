import pytest

from pgmpy.base import MAG


@pytest.fixture
def mag():
    """MAG with nodes A, B, C, D, L and mixed edges."""
    edges = [
        ("A", "B", ">", ">"),
        ("C", "D", ">", ">"),
        ("A", "C", ">", ">"),
        ("B", "D", ">", ">"),
        ("A", "D", "-", ">"),
        ("B", "C", "-", ">"),
        ("A", "L", "-", "-"),
        ("C", "L", "-", "-"),
    ]
    latents = {"L"}
    return MAG(ebunch=edges, latents=latents)


def test_init_empty():
    empty = MAG()
    assert len(empty.nodes()) == 0
    assert empty.latents == set()


def test_roles_assignment_and_equality():
    e = [("X", "Z", "-", ">"), ("Y", "Z", "-", ">")]
    roles = {"exposure": "X", "outcome": "Z", "adjustment": {"Y"}}
    m1 = MAG(ebunch=e, latents={"L"}, roles=roles)
    m2 = MAG(
        ebunch=e,
        latents={"L"},
        roles={"exposure": "X", "outcome": "Z", "adjustment": {"Y"}},
    )
    assert m1 == m2

    m3 = MAG(ebunch=e, latents={"L"}, roles={"exposure": "X"})
    assert m1 != m3

    m4 = MAG(
        ebunch=[("X", "Z", ">", ">"), ("Y", "Z", "-", ">")], latents={"L"}, roles=roles
    )
    assert m1 != m4

    m5 = MAG(ebunch=e, latents={"L", "U"}, roles=roles)
    assert m1 != m5


def test_is_collider_on_fixture(mag):
    assert mag._is_collider("A", "D", "B")

    assert not mag._is_collider("A", "B", "C")


def test_is_collider_simple():
    m = MAG(ebunch=[("X", "Z", "-", ">"), ("Y", "Z", "-", ">")])
    assert m._is_collider("X", "Z", "Y")


def test_has_inducing_path_on_fixture_is_false_for_pairs_without_valid_colliders(mag):
    assert not mag.has_inducing_path("A", "B", {"L"})
    assert not mag.has_inducing_path("C", "D", {"L"})
    assert not mag.has_inducing_path("A", "D", {"L"})

    assert not mag.has_inducing_path("X", "A", {"L"})


def test_has_inducing_path_latent_collider_true_when_in_W():
    m = MAG(ebunch=[("X", "L", "-", ">"), ("Y", "L", "-", ">")], latents={"L"})
    assert m.has_inducing_path("X", "Y", {"L"})
    assert not m.has_inducing_path("X", "Y", set())


def test_is_visible_edge_on_fixture(mag):
    assert mag.is_visible_edge("A", "B")
    assert mag.is_visible_edge("A", "D")
    assert mag.is_visible_edge("B", "C")

    assert not mag.is_visible_edge("A", "Z")
    assert not mag.is_visible_edge("Z", "Z")


def test_is_visible_edge_invisible_due_to_latent_inducing_path():
    m = MAG(
        ebunch=[("X", "Y", "-", ">"), ("X", "L", "-", ">"), ("Y", "L", "-", ">")],
        latents={"L"},
    )

    assert not m.is_visible_edge("X", "Y")


def test_lower_manipulation_on_fixture_removes_L_and_preserves_others(mag):
    new_mag = mag.lower_manipulation({"L"})
    assert "L" not in new_mag.nodes()
    assert new_mag.has_edge("A", "B")
    assert new_mag.has_edge("A", "C")
    assert new_mag.has_edge("B", "D")
    assert new_mag.has_edge("C", "D")
    assert new_mag.has_edge("A", "D")
    assert new_mag.has_edge("B", "C")


def test_lower_manipulation_adds_bidirected_when_node_is_collider():
    m = MAG(ebunch=[("X", "L", "-", ">"), ("Y", "L", "-", ">")], latents={"L"})
    new_m = m.lower_manipulation({"L"})
    assert "L" not in new_m.nodes()
    assert new_m.has_edge("X", "Y")

    marks = new_m.edges["X", "Y"]["marks"]
    assert marks["X"] == ">" and marks["Y"] == ">"


def test_upper_manipulation_removes_outgoing_edges_only(mag):
    new_mag = mag.upper_manipulation({"A"})
    assert not new_mag.has_edge("A", "D")
    assert new_mag.has_edge("A", "B")
    assert new_mag.has_edge("A", "C")
    assert new_mag.has_edge("A", "L")

    new_mag_b = mag.upper_manipulation({"B"})
    assert not new_mag_b.has_edge("B", "C")
    assert new_mag_b.has_edge("B", "D")


def test_manipulations_do_not_modify_original(mag):
    original_nodes = set(mag.nodes())
    original_edges = {
        (u, v, frozenset(data["marks"].items())) for u, v, data in mag.edges(data=True)
    }

    _ = mag.lower_manipulation({"L"})
    _ = mag.upper_manipulation({"A"})

    assert set(mag.nodes()) == original_nodes
    now_edges = {
        (u, v, frozenset(data["marks"].items())) for u, v, data in mag.edges(data=True)
    }
    assert now_edges == original_edges


def test_graph_properties_fixture(mag):
    assert len(mag.nodes()) == 5
    assert mag.latents == {"L"}
    assert mag.has_edge("A", "B")
    assert mag.has_edge("C", "D")
    assert mag.has_edge("A", "D")
    assert mag.has_edge("B", "C")
    assert mag.has_edge("A", "L")
    assert mag.has_edge("C", "L")
