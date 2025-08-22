import pytest

from pgmpy.base.ADMG import ADMG
from pgmpy.base.DAG import DAG


class TestADMGInitialization:
    """Test ADMG initialization and basic setup."""

    def test_empty_initialization(self):
        """Test creating an empty ADMG."""
        admg = ADMG()
        assert len(admg.nodes) == 0
        assert len(admg.edges) == 0
        assert len(admg.latents) == 0
        assert len(admg.get_roles()) == 0

    def test_initialization_with_directed_edges(self):
        """Test initialization with directed edges."""
        directed_edges = [("A", "B"), ("B", "C")]
        admg = ADMG(directed_ebunch=directed_edges)

        assert "A" in admg.nodes
        assert "B" in admg.nodes
        assert "C" in admg.nodes
        assert admg.has_edge("A", "B")
        assert admg.has_edge("B", "C")

    def test_initialization_with_bidirected_edges(self):
        """Test initialization with bidirected edges."""
        bidirected_edges = [("X", "Y"), ("Y", "Z")]
        admg = ADMG(bidirected_ebunch=bidirected_edges)

        assert "X" in admg.nodes
        assert "Y" in admg.nodes
        assert "Z" in admg.nodes
        # Bidirected edges create edges in both directions
        assert admg.has_edge("X", "Y")
        assert admg.has_edge("Y", "X")

    def test_initialization_with_latents(self):
        """Test initialization with latent variables."""
        latents = ["L1", "L2"]
        admg = ADMG(latents=latents)

        assert admg.latents == {"L1", "L2"}

    def test_initialization_with_roles(self):
        """Test initialization with roles variables."""
        directed_edges = [("A", "C"), ("B", "C")]
        roles = {"exposure": ("A", "B"), "outcome": ["C"]}
        admg = ADMG(directed_ebunch=directed_edges, roles=roles)

        assert set(admg.get_role("exposure")) == set(["A", "B"])
        assert admg.get_role("outcome") == ["C"]
        assert set(admg.get_roles()) == set(["exposure", "outcome"])
        assert admg.get_role_dict() == {"exposure": ["A", "B"], "outcome": ["C"]}


class TestADMGNodeOperations:
    """Test node addition and validation."""

    def test_add_node(self):
        """Test adding a single node."""
        admg = ADMG()
        admg.add_node("A")
        admg.add_node("B")
        admg.add_node("C", latent=True)
        admg.add_node("D", latent=True)

        assert set(admg.nodes()) == {"A", "B", "C", "D"}
        assert set(admg.latents) == {"C", "D"}

    def test_add_nodes_from(self):
        """Test adding multiple nodes at once."""
        admg = ADMG()
        admg.add_nodes_from(["A", "B"])
        admg.add_nodes_from(set(["C", "D"]))
        admg.add_nodes_from(["E", "F"], latent=[False, True])
        admg.add_nodes_from(["G", "H"], latent=True)
        admg.add_nodes_from(set(["I", "J"]), latent=True)

        assert set(admg.nodes()) == {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}
        assert set(admg.latents) == {"F", "G", "H", "I", "J"}


class TestADMGEdgeOperations:
    """Test edge addition and validation."""

    def test_add_directed_edges(self):
        """Test adding directed edges."""
        admg = ADMG()
        egdes = [("A", "B"), ("B", "C")]
        admg.add_directed_edges(egdes)

        assert admg.has_edge("A", "B")
        assert admg.get_edge_data("A", "B", 0)["type"] == "directed"

    def test_add_bidirected_edges(self):
        """Test adding bidirected edges."""
        admg = ADMG()
        admg.add_bidirected_edges([("X", "Y")])

        assert admg.has_edge("X", "Y")
        assert admg.has_edge("Y", "X")
        assert admg.get_edge_data("X", "Y", 0)["type"] == "bidirected"
        assert admg.get_edge_data("Y", "X", 0)["type"] == "bidirected"

    def testadd_directed_edgess_batch(self):
        """Test adding multiple directed edges at once."""
        admg = ADMG()
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        admg.add_directed_edges(edges)

        for u, v in edges:
            assert admg.has_edge(u, v)
            assert admg.get_edge_data(u, v, 0)["type"] == "directed"

    def test_add_bidirected_edgess_batch(self):
        """Test adding multiple bidirected edges at once."""
        admg = ADMG()
        edges = [("X", "Y"), ("Y", "Z")]
        admg.add_bidirected_edges(edges)

        for u, v in edges:
            assert admg.has_edge(u, v)
            assert admg.has_edge(v, u)

    def test_cycle_detection(self):
        """Test that cycles are prevented in directed edges."""
        admg = ADMG()
        admg.add_directed_edges([("A", "B")])
        admg.add_directed_edges([("B", "C")])

        # This should raise an error as it creates a cycle
        with pytest.raises(ValueError, match="Adding this edge would create a cycle"):
            admg.add_directed_edges([("C", "A")])

    def test_none_node_rejection(self):
        """Test that None nodes are rejected."""
        admg = ADMG()

        with pytest.raises(ValueError, match="Can't add since one of nodes is None"):
            admg.add_directed_edges([(None, "B")])

        with pytest.raises(ValueError, match="Can't add since one of"):
            admg.add_bidirected_edges([("A", None)])

    def test_self_bidirected_edge_rejection(self):
        """Test that self-loops in bidirected edges are rejected."""
        admg = ADMG()

        with pytest.raises(
            ValueError, match="Cannot add a bidirected edge from a node to itself"
        ):
            admg.add_bidirected_edges([("A", "A")])

    def test_add_edge_not_implemented(self):
        """Test that generic add_edge raises NotImplementedError."""
        admg = ADMG()

        with pytest.raises(NotImplementedError):
            admg.add_edge("A", "B")


class TestADMGRelationships:
    """Test getting parents, children, spouses, etc."""

    def setup_method(self):
        """Set up a test graph for relationship tests."""
        self.admg = ADMG()
        # Directed edges: A -> B -> C, D -> B
        self.admg.add_directed_edges([("A", "B"), ("B", "C"), ("D", "B")])
        # Bidirected edges: A <-> D, B <-> E
        self.admg.add_bidirected_edges([("A", "D"), ("B", "E")])

    def test_get_directed_parents(self):
        """Test getting parents of nodes."""
        parents = self.admg.get_directed_parents("B")

        assert "A" in parents
        assert "D" in parents
        assert len(parents) == 2  # A and D are parents of B

    def test_get_bidirected_parents(self):
        """Test getting bidirected parents of nodes."""
        bidirected_parents = self.admg.get_bidirected_parents("B")

        assert "A" not in bidirected_parents
        assert "E" in bidirected_parents
        assert len(bidirected_parents) == 1  # Only E is a bidirected parent of B

    def test_get_children(self):
        """Test getting children of nodes."""
        children = self.admg.get_children("B")

        assert "C" in children
        assert len(children) == 1

    def test_get_spouses(self):
        """Test getting spouses (bidirected connections)."""
        spouses_a = self.admg.get_spouses("A")
        spouses_b = self.admg.get_spouses("B")

        assert "D" in spouses_a
        assert "E" in spouses_b

    def test_get_ancestors(self):
        """Test getting ancestors."""
        ancestors_c = self.admg.get_ancestors("C")

        assert "A" in ancestors_c
        assert "B" in ancestors_c
        assert "D" in ancestors_c
        assert "C" in ancestors_c  # Node includes itself

    def test_get_descendants(self):
        """Test getting descendants."""
        descendants_a = self.admg.get_descendants("A")

        assert "B" in descendants_a
        assert "C" in descendants_a
        assert "A" in descendants_a  # Node includes itself

    def test_get_district(self):
        """Test getting district (bidirected connected components)."""
        district_a = self.admg.get_district("A")

        # A and D are connected by bidirected edge
        assert "A" in district_a
        assert "D" in district_a

    def test_nonexistent_node_error(self):
        """Test that operations on nonexistent nodes raise errors."""
        with pytest.raises(ValueError, match="Node .* is not in the graph"):
            self.admg.get_directed_parents("Z")

        with pytest.raises(ValueError, match="Node .* is not in the graph"):
            self.admg.get_children("Z")


class TestADMGGraphOperations:
    """Test advanced graph operations."""

    def setup_method(self):
        """Set up a test graph."""
        self.admg = ADMG()
        self.admg.add_directed_edges([("A", "B"), ("B", "C"), ("D", "E")])
        self.admg.add_bidirected_edges([("A", "D"), ("B", "E")])
        self.admg.add_node("F", latent=True)
        self.admg.with_role(role="exposure", variables={"A"}, inplace=True)
        self.admg.with_role(role="outcome", variables={"C"}, inplace=True)

    def test_get_ancestral_graph(self):
        """Test getting ancestral graph of a subset of nodes."""
        ancestral = self.admg.get_ancestral_graph(["A", "B", "D"])

        assert "A" in ancestral.nodes
        assert "B" in ancestral.nodes
        assert "D" in ancestral.nodes
        assert "C" not in ancestral.nodes
        assert "E" not in ancestral.nodes

        # Should have directed edge A -> B
        assert ancestral.has_edge("A", "B")
        # Should have bidirected edge A <-> D
        assert ancestral.has_edge("A", "D")
        assert ancestral.has_edge("D", "A")

    def test_get_ancestral_graph_invalid_nodes(self):
        """Test ancestral graph with invalid nodes."""
        with pytest.raises(ValueError, match="Input nodes must be subset"):
            self.admg.get_ancestral_graph(["A", "Z"])

    def test_get_markov_blanket(self):
        """Test getting Markov blanket."""
        mb_b = self.admg.get_markov_blanket("B")

        # B's Markov blanket should include its parents, children, and spouses
        assert "A" in mb_b  # parent
        assert "C" in mb_b  # child
        assert "E" in mb_b  # spouse

    def test_to_dag(self):
        """Test conversion to DAG."""
        dag = self.admg.to_dag()

        # Should return a pgmpy DAG instance
        from pgmpy.base.DAG import DAG as pgmpy_DAG

        assert isinstance(dag, pgmpy_DAG)

    def test_admg_equality(self):
        """
        Test the `__eq__` method
        which compares both graph structure and variable-role mappings to allow comparison of two models.
        """
        # ToDo:
        # If issue #2306 is resolved,
        # `admg` should be deleted.
        # issue_url: https://github.com/pgmpy/pgmpy/issues/2306
        admg = ADMG(
            directed_ebunch=[("A", "B"), ("B", "C"), ("D", "E")],
            bidirected_ebunch=[("A", "D"), ("B", "E")],
            latents=["F"],
            roles={"exposure": ["A"], "outcome": ["C"]},
        )

        # Case1: When the models are the same
        other1 = ADMG(
            directed_ebunch=[("A", "B"), ("B", "C"), ("D", "E")],
            bidirected_ebunch=[("A", "D"), ("B", "E")],
            latents=["F"],
            roles={"exposure": ["A"], "outcome": ["C"]},
        )
        # Case2: When the models differ
        other2 = DAG(
            ebunch=[("A", "C"), ("D", "C")],
            latents=["B"],
            roles={"exposure": "A", "adjustment": "D", "outcome": "C"},
        )
        # Case3: When the directed_ebunch variables differ between models
        other3 = ADMG(
            directed_ebunch=[("A", "C"), ("B", "C"), ("D", "E")],
            bidirected_ebunch=[("A", "D"), ("B", "E")],
            latents=["F"],
            roles={"exposure": ["A"], "outcome": ["C"]},
        )
        # Case4: When the bidirected_ebunch variables differ between models
        other4 = ADMG(
            directed_ebunch=[("A", "B"), ("B", "C"), ("D", "E")],
            bidirected_ebunch=[("A", "E"), ("B", "E")],
            latents=["F"],
            roles={"exposure": ["A"], "outcome": ["C"]},
        )
        # Case5: When the latents variables differ between models
        other5 = ADMG(
            directed_ebunch=[("A", "B"), ("B", "C"), ("D", "E")],
            bidirected_ebunch=[("A", "D"), ("B", "E")],
            latents=["G"],
            roles={"exposure": ["A"], "outcome": ["C"]},
        )
        # Case6: When the roles variables differ between models
        other6 = ADMG(
            directed_ebunch=[("A", "B"), ("B", "C"), ("D", "E")],
            bidirected_ebunch=[("A", "D"), ("B", "E")],
            latents=["F"],
            roles={"exposure": ["A"], "adjustment": "D", "outcome": ["C"]},
        )

        # ToDo:
        # If issue #2306 is resolved,
        # `admg.__eq__(other_number)` should be changed to `self.admg.__eq__(other_number)`.
        # issue_url: https://github.com/pgmpy/pgmpy/issues/2306
        assert admg.__eq__(other1) is True
        assert admg.__eq__(other2) is False
        assert admg.__eq__(other3) is False
        assert admg.__eq__(other4) is False
        assert admg.__eq__(other5) is False
        assert admg.__eq__(other6) is False


class TestADMGSeparation:
    """Test m-separation and m-connection."""

    def setup_method(self):
        """Set up a test graph for separation tests."""
        self.admg = ADMG()
        self.admg.add_directed_edges([("A", "C"), ("B", "C"), ("C", "D")])
        self.admg.add_bidirected_edges([("A", "B")])

    def test_is_m_separated(self):
        """Test m-separation check."""
        # A and B should not be m-separated (they have bidirected edge)
        assert not self.admg.is_mseparated("A", "B")

        # Test with conditional set
        assert self.admg.is_mseparated("A", "D", conditional_set={"C"}) is True
        assert self.admg.is_mseparated("A", "D", conditional_set=set()) is False
        # This depends on the specific graph structure and d-separation rules

    def test_is_m_connected(self):
        """Test m-connection check."""
        # This should be the opposite of m-separation
        connected = self.admg.is_mconnected("A", "B")
        separated = self.admg.is_mseparated("A", "B")
        assert connected != separated
