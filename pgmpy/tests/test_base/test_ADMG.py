import pytest
import networkx as nx

from pgmpy.base.ADMG import ADMG


class TestADMGInitialization:
    """Test ADMG initialization and basic setup."""

    def test_empty_initialization(self):
        """Test creating an empty ADMG."""
        admg = ADMG()
        assert len(admg.nodes) == 0
        assert len(admg.edges) == 0
        assert len(admg.latents) == 0

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


class TestADMGEdgeOperations:
    """Test edge addition and validation."""

    def test_add_directed_edge(self):
        """Test adding directed edges."""
        admg = ADMG()
        admg._add_directed_edge("A", "B")

        assert admg.has_edge("A", "B")
        assert admg.get_edge_data("A", "B", 0)["type"] == "directed"

    def test_add_bidirected_edge(self):
        """Test adding bidirected edges."""
        admg = ADMG()
        admg._add_bidirected_edge("X", "Y")

        assert admg.has_edge("X", "Y")
        assert admg.has_edge("Y", "X")
        assert admg.get_edge_data("X", "Y", 0)["type"] == "bidirected"
        assert admg.get_edge_data("Y", "X", 0)["type"] == "bidirected"

    def test_add_directed_edges_batch(self):
        """Test adding multiple directed edges at once."""
        admg = ADMG()
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        admg.add_directed_edges(edges)

        for u, v in edges:
            assert admg.has_edge(u, v)
            assert admg.get_edge_data(u, v, 0)["type"] == "directed"

    def test_add_bidirected_edges_batch(self):
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
        admg._add_directed_edge("A", "B")
        admg._add_directed_edge("B", "C")

        # This should raise an error as it creates a cycle
        with pytest.raises(ValueError, match="Adding this edge would create a cycle"):
            admg._add_directed_edge("C", "A")

    def test_none_node_rejection(self):
        """Test that None nodes are rejected."""
        admg = ADMG()

        with pytest.raises(ValueError, match="Can't add since one of nodes is None"):
            admg._add_directed_edge(None, "B")

        with pytest.raises(ValueError, match="Can't add since one of"):
            admg._add_bidirected_edge("A", None)

    def test_self_bidirected_edge_rejection(self):
        """Test that self-loops in bidirected edges are rejected."""
        admg = ADMG()

        with pytest.raises(
            ValueError, match="Cannot add a bidirected edge from a node to itself"
        ):
            admg._add_bidirected_edge("A", "A")

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

    def test_get_parents(self):
        """Test getting parents of nodes."""
        parents, district_parents = self.admg.get_parents("B")

        assert "A" in parents
        assert "D" in parents
        # A and D are also connected by bidirected edge, so they're district parents too
        assert "A" in district_parents
        assert "D" in district_parents

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
            self.admg.get_parents("Z")

        with pytest.raises(ValueError, match="Node .* is not in the graph"):
            self.admg.get_children("Z")


class TestADMGGraphOperations:
    """Test advanced graph operations."""

    def setup_method(self):
        """Set up a test graph."""
        self.admg = ADMG()
        self.admg.add_directed_edges([("A", "B"), ("B", "C"), ("D", "E")])
        self.admg.add_bidirected_edges([("A", "D"), ("B", "E")])

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


# class TestADMGSeparation:
#     """Test m-separation and m-connection."""

#     def setup_method(self):
#         """Set up a test graph for separation tests."""
#         self.admg = ADMG()
#         self.admg.add_directed_edges([("A", "C"), ("B", "C"), ("C", "D")])
#         self.admg.add_bidirected_edges([("A", "B")])

#     def test_is_m_separated(self):
#         """Test m-separation check."""
#         # A and B should not be m-separated (they have bidirected edge)
#         assert not self.admg.is_m_separated("A", "B")

#         # Test with conditional set
#         separated = self.admg.is_m_separated("A", "D", conditional_set={"C"})
#         # This depends on the specific graph structure and d-separation rules

#     def test_is_m_connected(self):
#         """Test m-connection check."""
#         # This should be the opposite of m-separation
#         connected = self.admg.is_m_connected("A", "B")
#         separated = self.admg.is_m_separated("A", "B")
#         assert connected != separated
