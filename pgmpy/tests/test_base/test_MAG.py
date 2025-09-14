import pytest

from pgmpy.base import MAG


# graph has been taken from the zhang 2008 paper (figure 1)
@pytest.fixture
def mag():
    edges = [
        ("A", "B", ">", ">"),
        ("C", "D", ">", ">"),
        ("A", "C", ">", ">"),
        ("B", "D", ">", ">"),
        ("A", "D", "-", ">"),
        ("B", "C", "-", ">"),
    ]
    roles = {"exposure": {"A"}, "outcome": {"D"}, "adjustment": {"B", "C"}}
    return MAG(ebunch=edges, roles=roles)


@pytest.fixture
def mag2():
    edges = [
        ("P", "Q", ">", ">"),
        ("Q", "R", "-", ">"),
        ("P", "R", "-", ">"),
        ("P", "L", "-", ">"),
    ]
    return MAG(ebunch=edges, latents={"L"})


# mag3 and mag4 are taken from Maathuis 2018 JMLR Figure 2
@pytest.fixture
def mag3():
    edges = [("V", "X", "-", ">"), ("X", "Y", "-", ">")]
    return MAG(ebunch=edges)


@pytest.fixture
def mag4():
    edges = [
        ("V1", "V2", ">", ">"),
        ("V2", "V3", ">", ">"),
        ("V3", "V4", ">", ">"),
        ("V3", "V4", ">", ">"),
        ("V4", "X", ">", ">"),
        ("X", "Y", "-", ">"),
        ("V2", "Y", "-", ">"),
        ("V3", "Y", "-", ">"),
        ("V4", "Y", "-", ">"),
    ]
    return MAG(ebunch=edges)


class TestMAG:
    def test_empty_init(self):
        empty = MAG()
        assert len(empty.nodes()) == 0
        assert empty.latents == set()

    def test_roles_and_equality(self):
        e = [
            ("X", "Z", "-", ">"),
            ("Y", "Z", "-", ">"),
            ("L", "X", "-", ">"),
            ("L", "Z", "-", ">"),
            ("U", "X", "-", ">"),
        ]
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
            ebunch=[
                ("X", "Z", ">", ">"),
                ("Y", "Z", "-", ">"),
                ("L", "X", "-", ">"),
                ("L", "Z", "-", ">"),
                ("U", "X", "-", ">"),
            ],
            latents={"L"},
            roles=roles,
        )
        assert m1 != m4

        m5 = MAG(ebunch=e, latents={"L", "U"}, roles=roles)
        assert m1 != m5

    def test_is_collider(self, mag):
        assert mag._is_collider("A", "D", "B")
        assert not mag._is_collider("A", "B", "C")

    def test_has_inducing_path(self, mag):
        assert not mag.has_inducing_path("A", "B", {"L"})
        assert not mag.has_inducing_path("C", "D", {"L"})
        assert not mag.has_inducing_path("A", "D", {"L"})

        edges = [
            ("X", "L", "-", ">"),
            ("Y", "L", "-", ">"),
        ]
        new_mag = MAG(ebunch=edges, latents={"L"})
        assert new_mag.has_inducing_path("X", "Y", {"L"})

    def test_is_visible_edge(self, mag, mag3, mag4):
        assert not mag.is_visible_edge("A", "D")
        assert not mag.is_visible_edge("B", "C")
        assert not mag.is_visible_edge("A", "B")
        assert not mag.is_visible_edge("C", "D")
        assert not mag.is_visible_edge("A", "C")
        assert not mag.is_visible_edge("B", "D")

        assert mag3.is_visible_edge("X", "Y")
        assert mag4.is_visible_edge("X", "Y")

    def test_lower_manipulation(self, mag, mag2):
        new_mag = mag.lower_manipulation({"A"})
        assert not new_mag.has_edge("A", "D")
        assert new_mag.has_edge("A", "B")
        assert new_mag.has_edge("A", "C")
        assert new_mag.has_edge("B", "C")

        new_mag = mag.lower_manipulation({"B"})
        assert not new_mag.has_edge("B", "C")
        assert new_mag.has_edge("A", "B")
        assert new_mag.has_edge("A", "C")
        assert new_mag.has_edge("A", "D")

        new_mag2 = mag2.lower_manipulation({"P"})
        assert not new_mag2.has_edge("P", "R")
        assert new_mag2.has_edge("Q", "R")

    def test_upper_manipulation(self, mag, mag2):
        new_mag = mag.upper_manipulation({"D"})
        assert not new_mag.has_edge("A", "D")
        assert not new_mag.has_edge("C", "D")
        assert not new_mag.has_edge("B", "D")
        assert new_mag.has_edge("A", "B")

        new_mag = mag.upper_manipulation({"C"})
        assert not new_mag.has_edge("B", "C")
        assert not new_mag.has_edge("A", "C")
        assert new_mag.has_edge("A", "B")

        new_mag2 = mag2.upper_manipulation({"R"})
        assert not new_mag2.has_edge("P", "R")
        assert not new_mag2.has_edge("Q", "R")
        assert new_mag2.has_edge("P", "Q")

    def test_graph_properties(self, mag):
        assert len(mag.nodes()) == 4
        assert mag.has_edge("A", "B")
        assert mag.has_edge("C", "D")
        assert mag.has_edge("A", "D")
        assert mag.has_edge("B", "C")
