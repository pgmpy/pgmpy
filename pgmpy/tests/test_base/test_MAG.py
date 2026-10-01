import pytest

from pgmpy.base import MAG


# graph has been taken from the zhang 2008 paper (figure 1)
@pytest.fixture
def mag():
    edges = [
        ("A", "B", "<>"),
        ("C", "D", "<>"),
        ("A", "C", "<>"),
        ("B", "D", "<>"),
        ("A", "D", "->"),
        ("B", "C", "->"),
    ]
    roles = {"exposures": {"A"}, "outcomes": {"D"}, "adjustment": {"B", "C"}}
    return MAG(edge_list=edges, roles=roles)


@pytest.fixture
def mag2():
    edges = [
        ("P", "Q", "<>"),
        ("Q", "R", "->"),
        ("P", "R", "->"),
        ("P", "L", "->"),
    ]
    return MAG(edge_list=edges, latents={"L"})


# mag3 and mag4 are taken from Maathuis 2018 JMLR Figure 2
@pytest.fixture
def mag3():
    edges = [("V", "X", "->"), ("X", "Y", "->")]
    return MAG(edge_list=edges)


@pytest.fixture
def mag4():
    edges = [
        ("V1", "V2", "<>"),
        ("V2", "V3", "<>"),
        ("V3", "V4", "<>"),
        ("V4", "X", "<>"),
        ("X", "Y", "->"),
        ("V2", "Y", "->"),
        ("V3", "Y", "->"),
        ("V4", "Y", "->"),
    ]
    return MAG(edge_list=edges)


class TestMAG:
    def test_empty_init(self):
        empty = MAG()
        assert len(empty.nodes()) == 0
        assert empty.latents == set()

    def test_roles_and_equality(self):
        e = [
            ("X", "Z", "->"),
            ("Y", "Z", "->"),
            ("L", "X", "->"),
            ("L", "Z", "->"),
            ("U", "X", "->"),
        ]
        roles = {"exposures": "X", "outcomes": "Z", "adjustment": {"Y"}}
        m1 = MAG(edge_list=e, latents={"L"}, roles=roles)
        m2 = MAG(
            edge_list=e,
            latents={"L"},
            roles={"exposures": "X", "outcomes": "Z", "adjustment": {"Y"}},
        )
        assert m1 == m2

        m3 = MAG(edge_list=e, latents={"L"}, roles={"exposures": "X"})
        assert m1 != m3

        m4 = MAG(
            edge_list=[
                ("X", "Z", "<>"),
                ("Y", "Z", "->"),
                ("L", "X", "->"),
                ("L", "Z", "->"),
                ("U", "X", "->"),
            ],
            latents={"L"},
            roles=roles,
        )
        assert m1 != m4

        m5 = MAG(edge_list=e, latents={"L", "U"}, roles=roles)
        assert m1 != m5

    @pytest.mark.skip(reason="Refactoring: Skip now. I implement this When Refactoring DAG(Related: #2384, #2385)")
    def test_is_valid_mag(self):
        """Test code for `is_valid_mag` method"""
        # TODO(@daehyun99): [#2384] Implement code logic and test code
        ...

    def test_add_directed_edge(self):
        """Test adding directed edges."""
        mag = MAG()
        mag.add_edge("A", "B", "->")
        mag.add_edge("B", "C", "->")

        assert mag.has_edge("A", "B")
        assert set(mag.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

    def test_add_undirected_edge(self):
        """Test adding undirected edges."""
        mag = MAG()
        mag.add_edge("A", "B", "--")
        mag.add_edge("B", "C", "--")

        assert mag.has_edge("A", "B")
        assert set(mag.get_edges(data=True)) == {("A", "B", "--"), ("B", "C", "--")}

    def test_add_bidirected_edge(self):
        """Test adding bidirected edges."""
        mag = MAG()
        mag.add_edge("X", "Y", "<>")

        assert mag.has_edge("X", "Y")
        assert set(mag.get_edges(data=True)) == {("X", "Y", "<>")}

    def test_add_directed_edges(self):
        """Test adding multiple directed edges at once."""
        mag = MAG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "->")]
        mag.add_edges_from(edges)

        for u, v, _ in edges:
            assert mag.has_edge(u, v)

        assert set(mag.get_edges(data=True)) == set(edges)

    def test_add_undirected_edges(self):
        """Test adding multiple undirected edges at once."""
        mag = MAG()
        edges = [("A", "B", "--"), ("B", "C", "--"), ("C", "D", "--")]
        mag.add_edges_from(edges)

        for u, v, _ in edges:
            assert mag.has_edge(u, v)

        assert set(mag.get_edges(data=True)) == set(edges)

    def test_add_bidirected_edges(self):
        """Test adding multiple bidirected edges at once."""
        mag = MAG()
        edges = [("X", "Y", "<>"), ("Y", "Z", "<>")]
        mag.add_edges_from(edges)

        for u, v, _ in edges:
            assert mag.has_edge(u, v)

        assert set(mag.get_edges(data=True)) == set(edges)

    # MAG-specific algorithm methods (migrated to the _CoreGraph marker API)

    def test_is_collider(self):
        m = MAG(edge_list=[("X", "Z", "->"), ("Y", "Z", "->")])
        assert m.is_collider("X", "Z", "Y") is True
        # Z is not a collider when an edge has its tail at Z
        m2 = MAG(edge_list=[("X", "Z", "->"), ("Z", "Y", "->")])
        assert m2.is_collider("X", "Z", "Y") is False
        # a shielded collider counts by default; shielded=False makes it a v-structure test
        m3 = MAG(edge_list=[("X", "Z", "->"), ("Y", "Z", "->"), ("X", "Y", "->")])
        assert m3.is_collider("X", "Z", "Y") is True
        assert m3.is_collider("X", "Z", "Y", shielded=False) is False

    def test_is_maximal(self):
        """A MAG is maximal iff no primitive inducing path joins a non-adjacent pair.

        References
        ----------
        [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
        Journal of Machine Learning Research 9 (2008): 1437-1474. Figure 1.
        """
        # Zhang (2008) Fig. 1-(a): C and D are non-adjacent but joined by the primitive inducing
        # path C <> A <> B <> D (colliders A, B are ancestors of the endpoints) -> not maximal
        edges = [("A", "B", "<>"), ("A", "C", "<>"), ("B", "D", "<>"), ("A", "D", "->"), ("B", "C", "->")]
        assert MAG(edge_list=edges).is_maximal() is False
        # Fig. 1-(b): adding the C <> D edge makes it maximal
        assert MAG(edge_list=[*edges, ("C", "D", "<>")]).is_maximal() is True

        # maximality is a property of the graph itself: the latents role must not affect it
        assert MAG(edge_list=edges, latents={"A"}).is_maximal() is False

        # chain and collider: the non-adjacent pair has no inducing path (Y not a collider /
        # Y a collider but not an ancestor of an endpoint) -> maximal
        assert MAG(edge_list=[("X", "Y", "->"), ("Y", "Z", "->")]).is_maximal() is True
        assert MAG(edge_list=[("X", "Y", "->"), ("Z", "Y", "->")]).is_maximal() is True

    def test_is_visible_edge(self):
        m = MAG(edge_list=[("A", "D", "->"), ("B", "C", "->"), ("X", "A", "->")])
        assert m.is_visible_edge("A", "D") is True
        assert m.is_visible_edge("B", "C") is False

    def test_lower_manipulation(self):
        m = MAG(edge_list=[("A", "B", "->"), ("C", "B", "->")])
        new = m.lower_manipulation({"A"})
        assert new.has_edge("B", "C", "<>")
        assert len(new.get_edges(data=False)) == 1
        assert m.has_edge("A", "B", "->")  # original is unchanged

    def test_upper_manipulation(self):
        m = MAG(edge_list=[("Y", "X", "->"), ("X", "Z", "->"), ("A", "X", "->")])
        new = m.upper_manipulation({"X"})
        assert new.has_edge("X", "Z")
        assert not new.has_edge("A", "X")
        assert not new.has_edge("X", "Y")
        assert m.has_edge("A", "X")  # original is unchanged


class TestMAGDagitty:
    def test_from_dagitty_edge_types(self):
        mag = MAG.from_dagitty("mag { X -> Y Y <-> Z Z -- W }")
        assert sorted(mag.get_edges(data=True)) == [
            ("X", "Y", "->"),
            ("Y", "Z", "<>"),
            ("Z", "W", "--"),
        ]

    def test_from_dagitty_roles(self):
        mag = MAG.from_dagitty("mag { X [exposure] Y [outcome] L [latent] X -> Y }")
        assert mag.exposures == {"X"}
        assert mag.outcomes == {"Y"}
        assert mag.latents == {"L"}
        assert set(mag.nodes()) == {"X", "Y", "L"}

    def test_from_dagitty_isolated_nodes(self):
        mag = MAG.from_dagitty("mag { A -> B C }")
        assert set(mag.nodes()) == {"A", "B", "C"}
        assert sorted(mag.get_edges(data=True)) == [("A", "B", "->")]

    def test_from_dagitty_empty(self):
        mag = MAG.from_dagitty("mag { }")
        assert len(mag.nodes()) == 0
        assert len(mag.get_edges(data=True)) == 0

    def test_from_dagitty_no_input(self):
        with pytest.raises(ValueError, match="Either `filename` or `string`"):
            MAG.from_dagitty()

    def test_from_dagitty_file(self, tmp_path):
        dagitty_file = tmp_path / "model.dagitty"
        dagitty_file.write_text("mag { X [exposure] X -> Y Y <-> Z }")
        mag = MAG.from_dagitty(filename=str(dagitty_file))
        assert sorted(mag.get_edges(data=True)) == [("X", "Y", "->"), ("Y", "Z", "<>")]
        assert mag.exposures == {"X"}

    def test_from_dagitty_multiline(self):
        mag = MAG.from_dagitty(
            """
            mag {
                X [exposure]
                Y [outcome]
                X -> Y
                Y <-> Z
            }
            """
        )
        assert sorted(mag.get_edges(data=True)) == [("X", "Y", "->"), ("Y", "Z", "<>")]
        assert mag.exposures == {"X"}
        assert mag.outcomes == {"Y"}

    def test_to_dagitty(self):
        mag = MAG(edge_list=[("X", "Y", "->"), ("Y", "Z", "<>"), ("Z", "W", "--")])
        assert mag.to_dagitty() == "mag {\nX -> Y\nY <-> Z\nZ -- W\n}"

    def test_to_dagitty_with_roles_and_isolated_node(self):
        mag = MAG(edge_list=[("A", "B", "->")], exposures={"A"}, outcomes={"B"})
        mag.add_node("C")
        assert mag.to_dagitty() == "mag {\nA -> B\nA [exposure]\nB [outcome]\nC\n}"

    def test_to_dagitty_empty(self):
        assert MAG().to_dagitty() == "mag {\n}"

    def test_dagitty_roundtrip(self):
        mag = MAG(
            edge_list=[("X", "Y", "->"), ("Y", "Z", "<>"), ("Z", "W", "--")],
            exposures={"X"},
            outcomes={"Y"},
        )
        mag.add_node("L")
        mag.latents = {"L"}
        mag.add_node("Q")
        restored = MAG.from_dagitty(mag.to_dagitty())
        assert sorted(restored.get_edges(data=True)) == sorted(mag.get_edges(data=True))
        assert set(restored.nodes()) == set(mag.nodes())
        assert restored.exposures == mag.exposures
        assert restored.outcomes == mag.outcomes
        assert restored.latents == mag.latents
