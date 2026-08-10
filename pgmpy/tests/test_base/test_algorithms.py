from itertools import combinations

import pytest

from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.base._base import _CoreGraph


@pytest.fixture
def AncestralGraph():
    """
    References
    ----------
    [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
    Journal of Machine Learning Research 9 (2008): 1437-1474. Figure 1-(a).
    """
    edges1 = [  # an ancestral graph that is not maximal.
        ("A", "B", "<>"),
        ("A", "C", "<>"),
        ("B", "D", "<>"),
        ("A", "D", "->"),
        ("B", "C", "->"),
    ]
    graph = _CoreGraph(edge_list=edges1)
    return graph


@pytest.fixture
def MaximalAncestralGraph():
    """
    References
    ----------
    [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
    Journal of Machine Learning Research 9 (2008): 1437-1474. Figure 1-(b).
    """
    edges1 = [  # an ancestral graph that is not maximal.
        ("A", "B", "<>"),
        ("A", "C", "<>"),
        ("B", "D", "<>"),
        ("A", "D", "->"),
        ("B", "C", "->"),
        ("C", "D", "<>"),
    ]
    graph = _CoreGraph(edge_list=edges1)
    return graph


class TestGraphAlgorithmMixin:
    def test_is_collider(self):
        """`is_collider(u, w, v)`: arrowheads at `w` on the edges from both `u` and `v`;
        `shielded=False` additionally requires `u` and `v` to be non-adjacent (v-structure test).

        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 3.
        """
        # arrowhead at M from T (T -> M); the edge type at the other flank decides
        graph = _CoreGraph(
            edge_list=[
                ("T", "M", "->"),
                ("M", "O", "->"),
                ("M", "I", "<-"),
                ("M", "B", "<>"),
                ("M", "U", "--"),
                ("Z", "M", "o>"),
            ]
        )
        assert graph.is_collider("T", "M", "I") is True  # I -> M
        assert graph.is_collider("T", "M", "B") is True  # B <> M
        assert graph.is_collider("T", "M", "Z") is True  # Z o> M: a circle-origin arrowhead counts
        assert graph.is_collider("T", "M", "O") is False  # M -> O: tail at M
        assert graph.is_collider("T", "M", "U") is False  # M -- U: tail at M

        # no arrowhead at M from T's side -> never a collider
        for t_edge in ["<-", "--", "-o"]:
            g = _CoreGraph(edge_list=[("T", "M", t_edge), ("M", "I", "<-")])
            assert g.is_collider("T", "M", "I") is False

        # shielded=False requires u and v to be non-adjacent (v-structure)
        graph.add_edge("T", "I", "->")
        assert graph.is_collider("T", "M", "I") is True  # the plain collider test is unaffected
        assert graph.is_collider("T", "M", "I", shielded=False) is False  # shielded by T -> I
        assert graph.is_collider("T", "M", "B", shielded=False) is True  # still unshielded

        # fails: node not in the graph
        with pytest.raises(ValueError):
            _CoreGraph().is_collider("T", "M", "O")

    def has_directed_cycle(self):
        """
        Testing `has_directed_cycle` method of Graph class(_CoreGraph, DAG, ADMG)
        """
        # TODO(@daehyun99): [#2385] Consider implement `has_directed_cycle` method.

    def has_almost_directed_cycle(self):
        """
        Testing `has_almost_directed_cycle` method of Graph class(_CoreGraph, DAG, ADMG)
        """
        # TODO(@daehyun99): [#2385] Consider implement `has_almost_directed_cycle` method.

    def test_get_ancestral_graph(self):
        """
        Testing `_get_ancestral_graph` method of All graph class
        """
        # _CoreGraph
        graph1 = _CoreGraph()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")]
        graph1.add_edges_from(edges)

        new_graph1 = graph1.get_ancestral_graph(["C"])

        assert isinstance(new_graph1, _CoreGraph)
        assert set(new_graph1.nodes()) == {"A", "B", "C"}
        assert set(new_graph1.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

        # ADMG
        graph2 = ADMG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "->")]
        graph2.add_edges_from(edges)

        new_graph2 = graph2.get_ancestral_graph(["C"])

        assert isinstance(new_graph2, ADMG)
        assert set(new_graph2.nodes()) == {"A", "B", "C"}
        assert set(new_graph2.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

        # MAG
        graph3 = MAG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")]
        graph3.add_edges_from(edges)

        new_graph3 = graph3.get_ancestral_graph(["C"])

        assert isinstance(new_graph3, MAG)
        assert set(new_graph3.nodes()) == {"A", "B", "C"}
        assert set(new_graph3.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

        # TODO(@daehyun99): [#2384] Expand DAG
        ...

        # TODO(@daehyun99): [#2384] Expand PAG
        ...

        # TODO(@daehyun99): [#2384] Expand UndirectedGraph
        ...

        # _CoreGraph with rols
        graph = _CoreGraph()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")]
        graph.add_edges_from(edges)
        graph.exposures = "A"
        graph.outcomes = {"C", "D"}
        graph.latents = {"D", "E"}

        new_graph = graph.get_ancestral_graph(["C"])

        assert isinstance(new_graph, _CoreGraph)
        assert set(new_graph.nodes()) == {"A", "B", "C"}
        assert new_graph.exposures == {"A"}
        assert new_graph.outcomes == {"C"}
        assert new_graph.latents == set()
        assert set(new_graph.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

    def test_get_markov_blanket(self):
        """Test getting Markov blanket."""
        edges = [
            ("A", "B", "->"),
            ("B", "C", "->"),
            ("D", "E", "->"),
            ("A", "D", "<>"),
            ("B", "E", "<>"),
        ]
        self.admg = ADMG()

        self.admg.add_edges_from(edges)
        self.admg.add_node("F", latent=True)
        self.admg.with_role(role="exposures", variables={"A"}, inplace=True)
        self.admg.with_role(role="outcomes", variables={"C"}, inplace=True)

        # parents(B)={A}, children(B)={C}, district(B)={B,E} (spouse E), pa(E)={D}.
        # D enters because conditioning on E (B <> E) opens the collider B <> E <- D.
        assert self.admg.get_markov_blanket("B") == {"A", "C", "D", "E"}

        # Co-parents: a child's other parents belong to the blanket (collider A -> C <- X).
        collider = ADMG(edge_list=[("A", "C", "->"), ("X", "C", "->")])
        assert collider.get_markov_blanket("A") == {"C", "X"}

        # The district is the full (transitive) bidirected component and its parents are included:
        # B <> E <> G with D -> E  =>  MB(B) = {E, G} (district) ∪ {D} (parent of E).
        chain = ADMG(edge_list=[("B", "E", "<>"), ("E", "G", "<>"), ("D", "E", "->")])
        assert chain.get_markov_blanket("B") == {"E", "G", "D"}

    def test_get_markov_blanket_fails(self):
        """Test getting Markov blanket."""
        graph = _CoreGraph()
        with pytest.raises(TypeError):
            graph.get_markov_blanket("B")

        graph = MAG()
        with pytest.raises(TypeError):
            graph.get_markov_blanket("B")

        graph = PDAG()
        with pytest.raises(TypeError):
            graph.get_markov_blanket("B")

    def test_has_inducing_path(self, AncestralGraph, MaximalAncestralGraph):
        """`has_inducing_path(u, v, w)`: a path on which every intermediate in `w` (the pool of
        variables available for conditioning; default: the observed nodes) is a collider, and every
        collider is an ancestor of `u` or `v` -- i.e. a dependence no subset of `w` can block."""
        # Zhang (2008) Fig. 1-(a): on C <> A <> B <> D the colliders A, B are ancestors of the
        # endpoints (A -> D, B -> C), so no subset of the observed nodes m-separates C and D
        assert AncestralGraph.has_inducing_path("C", "D") is True
        assert MaximalAncestralGraph.has_inducing_path("C", "D") is True

        # a latent chain X -> L -> Y is unblockable (L cannot be conditioned on) ...
        g = MAG(edge_list=[("X", "L", "->"), ("L", "Y", "->")], latents={"L"})
        assert g.has_inducing_path("X", "Y") is True
        # ... but allowing L into the conditioning pool makes the path blockable
        assert g.has_inducing_path("X", "Y", w={"L"}) is False

        # a latent collider X -> L <- Y is NOT inducing: conditioning on nothing blocks it, since
        # L is neither conditionable-irrelevant (it is a collider) nor an ancestor of an endpoint
        g = MAG(edge_list=[("X", "L", "->"), ("Y", "L", "->")], latents={"L"})
        assert g.has_inducing_path("X", "Y") is False

        # an observed chain is blocked by conditioning on the middle node
        g = _CoreGraph(edge_list=[("X", "M", "->"), ("M", "Y", "->")])
        assert g.has_inducing_path("X", "Y") is False

        # exists-semantics: one inducing path (X <> C <> Y with C -> X) plus an unrelated
        # non-inducing path (X -> M -> Y); the parallel C -> X edge also exercises multi-edges
        g = _CoreGraph(
            edge_list=[("X", "C", "<>"), ("C", "Y", "<>"), ("C", "X", "->"), ("X", "M", "->"), ("M", "Y", "->")]
        )
        assert g.has_inducing_path("X", "Y") is True

        # a direct edge is a trivial inducing path (no intermediate node can block it), any edge type
        for et in ("->", "<>", "--"):
            assert _CoreGraph(edge_list=[("X", "Y", et)]).has_inducing_path("X", "Y") is True
        # disconnected, non-adjacent nodes -> False
        g = _CoreGraph(edge_list=[("X", "A", "->"), ("B", "Y", "->")])
        assert g.has_inducing_path("X", "Y") is False

        # fails: endpoint not in the graph
        with pytest.raises(ValueError, match="not in graph"):
            _CoreGraph(edge_list=[("X", "Y", "->")]).has_inducing_path("X", "Q")

    def test_get_mconnected_nodes(self):
        """`get_mconnected_nodes(nodes, conditioning_set)`: all nodes reachable from `nodes` by an
        m-connecting walk -- colliders on the walk must be in the conditioning set, non-colliders
        must not be."""
        # hub motif: T -> M with M -> O, M <- I, M <> B
        g = _CoreGraph(edge_list=[("T", "M", "->"), ("M", "O", "->"), ("M", "I", "<-"), ("M", "B", "<>")])
        # unconditioned: M is a non-collider towards O (open), a collider towards I and B (blocked)
        assert g.get_mconnected_nodes("T") == {"T", "M", "O"}
        # conditioning on M flips it: collider paths open, non-collider paths block
        assert g.get_mconnected_nodes("T", {"M"}) == {"T", "M", "I", "B"}

        # undirected (selection) edges are non-colliders: open unconditioned, blocked when conditioned
        g = _CoreGraph(edge_list=[("X", "Z", "--"), ("Z", "Y", "--")])
        assert g.get_mconnected_nodes("X") == {"X", "Z", "Y"}
        assert g.get_mconnected_nodes("X", {"Z"}) == {"X", "Z"}

        # conditioning on a collider's DESCENDANT opens the collider (walk goes down to D and back)
        g = _CoreGraph(edge_list=[("X", "C", "->"), ("Y", "C", "->"), ("C", "D", "->")])
        assert g.get_mconnected_nodes("X") == {"X", "C", "D"}
        assert "Y" in g.get_mconnected_nodes("X", {"D"})

        # multiple sources: the union of reachable sets
        g = _CoreGraph(edge_list=[("A", "B", "->"), ("X", "Y", "->")])
        assert g.get_mconnected_nodes(["A", "X"]) == {"A", "B", "X", "Y"}

        # fails: missing node / source inside the conditioning set / circle marks (PAG) /
        # non-ancestral graph (an undirected edge meeting an arrowhead, as in a PDAG)
        with pytest.raises(ValueError, match="not in graph"):
            g.get_mconnected_nodes("Q")
        with pytest.raises(ValueError, match="disjoint"):
            g.get_mconnected_nodes("A", {"A"})
        pag = _CoreGraph(edge_list=[("A", "B", "o>")])
        with pytest.raises(ValueError, match="circle"):
            pag.get_mconnected_nodes("A")
        non_ancestral = _CoreGraph(edge_list=[("T", "M", "->"), ("M", "U", "--")])
        with pytest.raises(ValueError, match="ancestral"):
            non_ancestral.get_mconnected_nodes("T")

    def test_is_mseparated(self):
        """`is_mseparated(u, v, conditioning_set)`: True iff no m-connecting walk joins u and v.

        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 3.
        """
        # hub motif (restores the dev-era is_m_connected corpus): T -> M with M -> O, M <- I, M <> B
        g = _CoreGraph(edge_list=[("T", "M", "->"), ("M", "O", "->"), ("M", "I", "<-"), ("M", "B", "<>")])
        # given {M}: the non-collider route (O) blocks, collider routes (I, B) open
        assert g.is_mseparated("T", "O", {"M"}) is True
        assert g.is_mseparated("T", "I", {"M"}) is False
        assert g.is_mseparated("T", "B", {"M"}) is False
        # given the empty set: exactly reversed
        assert g.is_mseparated("T", "O") is False
        assert g.is_mseparated("T", "I") is True
        assert g.is_mseparated("T", "B") is True

        # conditioning on a collider's descendant opens the collider
        g = _CoreGraph(edge_list=[("X", "C", "->"), ("Y", "C", "->"), ("C", "D", "->")])
        assert g.is_mseparated("X", "Y") is True
        assert g.is_mseparated("X", "Y", {"C"}) is False
        assert g.is_mseparated("X", "Y", {"D"}) is False

        # MAG selection (undirected) edges are non-colliders; ADMG bidirected chains are colliders
        mag = MAG(edge_list=[("X", "Z", "--"), ("Z", "Y", "--")])
        assert mag.is_mseparated("X", "Y") is False
        assert mag.is_mseparated("X", "Y", {"Z"}) is True
        admg = ADMG(edge_list=[("X", "C", "<>"), ("C", "Y", "<>")])
        assert admg.is_mseparated("X", "Y") is True
        assert admg.is_mseparated("X", "Y", {"C"}) is False

        # on a purely directed graph, m-separation == d-separation: exhaustive check against
        # DAG.is_dconnected over all pairs and all conditioning subsets
        edges = [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C"), ("C", "E")]
        dag = DAG(ebunch=edges)
        core = _CoreGraph(edge_list=[(u, v, "->") for u, v in edges])
        node_names = ["A", "B", "C", "D", "E"]
        for u, v in combinations(node_names, 2):
            others = [n for n in node_names if n not in (u, v)]
            for r in range(len(others) + 1):
                for z in combinations(others, r):
                    observed = list(z) if z else None
                    assert core.is_mseparated(u, v, set(z)) == (not dag.is_dconnected(u, v, observed=observed))

        # fails: an endpoint inside the conditioning set / missing endpoint
        with pytest.raises(ValueError, match="conditioning_set"):
            core.is_mseparated("A", "B", {"A"})
        with pytest.raises(ValueError, match="not in graph"):
            core.is_mseparated("A", "Q")

    def test_check_new_unshielded_collider(self):
        graph = _CoreGraph()

        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "--")

        assert graph._check_new_unshielded_collider(u="C", v="B") is True
        assert graph._check_new_unshielded_collider(u="B", v="C") is False

        graph.add_edge("A", "C", "--")

        assert graph._check_new_unshielded_collider(u="C", v="B") is False

    # def test_get_directed_subgraph(self):
    #     graph = _CoreGraph()

    #     graph.add_edge("A", "B", "->")
    #     graph.add_edge("B", "C", "->")
    #     graph.add_edge("C", "D", "<>")
    #     graph.add_edge("D", "E", "--")
    #     graph.latents = "E"

    #     subgraph = graph.get_directed_subgraph()

    #     assert isinstance(
    #         subgraph, _CoreGraph
    #     )  # TODO(@daehyun99): [#2385] Refactoring _CoreGraph -> DAG when Refactor DAG
    #     assert set(subgraph.nodes()) == {"A", "B", "C", "D", "E"}
    #     assert set(subgraph.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}
    #     assert subgraph.latents == {"E"}
