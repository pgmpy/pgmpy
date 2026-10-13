"""Tests for the ID algorithm of Shpitser & Pearl (AAAI 2006)."""

import pytest

from pgmpy.base import ADMG, DAG
from pgmpy.identification import ID
from pgmpy.identification.probability_expression import (
    DivisionNode,
    MarginalNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
)

# Fig. 1 (a) of the paper: W1, W2 are afflictions of a pregnant mother and her unborn child, X a toxin-lowering
# treatment, Y1, Y2 the survival of the two patients. The bidirected arcs are reconstructed from the worked example on
# p. 1224, which pins down every intermediate quantity the algorithm computes on this graph (see
# ``test_figure_1a_matches_the_papers_worked_example``).
FIGURE_1A = [
    ("W1", "X", "->"),
    ("X", "Y1", "->"),
    ("W2", "Y2", "->"),
    ("W1", "W2", "<>"),
    ("W1", "Y1", "<>"),
    ("W1", "Y2", "<>"),
]

# Fig. 1 (b): the same graph with X pulled into the confounded component. The paper describes the resulting hedge as
# "if e is the edge between W1 and X, then F = G \ {e}, and F' = F \ {X}", which is only possible when a second,
# bidirected edge joins W1 and X.
FIGURE_1B = FIGURE_1A + [("W1", "X", "<>")]

BOW_ARC = [("X", "Y", "->"), ("X", "Y", "<>")]
EXTENDED_BOW_ARC = [("X", "Z", "->"), ("Z", "Y", "->"), ("X", "Z", "<>")]
INSTRUMENTAL_VARIABLE = [("Z", "X", "->"), ("X", "Y", "->"), ("X", "Y", "<>")]

CHAIN = [("X", "M", "->"), ("M", "Y", "->")]
BACK_DOOR = [("Z", "X", "->"), ("Z", "Y", "->"), ("X", "Y", "->")]
FRONT_DOOR = [("X", "M", "->"), ("M", "Y", "->"), ("X", "Y", "<>")]
NAPKIN = [
    ("W", "Z", "->"),
    ("Z", "X", "->"),
    ("X", "Y", "->"),
    ("W", "X", "<>"),
    ("W", "Y", "<>"),
]


def graph(edge_list, exposures, outcomes):
    """Build an ADMG with the exposure and outcome roles assigned."""
    return ADMG(edge_list=edge_list, exposures=set(exposures), outcomes=set(outcomes))


class TestIDPaperExamples:
    """Cases whose expected output is stated in the paper itself."""

    def test_figure_1a_matches_the_papers_worked_example(self):
        r"""Fig. 1 (a), p. 1224.

        The paper concludes the worked example with

        .. math::

            P_x(y_1, y_2) = \sum_{w_2} P(y_2 | w_2) P(w_2)
                            \sum_{w_1} P(y_1 | x, w_1) P(w_1)
        """
        result = ID().identify(graph(FIGURE_1A, {"X"}, {"Y1", "Y2"}))
        assert result.to_latex() == r"\sum_{W2} P(W2) P(Y2 \mid W2) \left[ \sum_{W1} P(W1) P(Y1 \mid W1, X) \right]"

    def test_figure_1a_intermediate_quantities_match_the_paper(self):
        """The paper states G = An(Y), C(G \\ {X}) = {G \\ {X}} and W = {W1}."""
        admg = graph(FIGURE_1A, {"X"}, {"Y1", "Y2"})
        variables = set(admg.nodes())

        assert admg.get_ancestors({"Y1", "Y2"}) == variables
        assert admg.get_subgraph(variables - {"X"}).get_district() == {frozenset(variables - {"X"})}

        # W = (V \ X) \ An(Y) in G with the incoming edges of X removed.
        ancestors_after_do = admg.do({"X"}).get_ancestors({"Y1", "Y2"})
        assert (variables - {"X"}) - ancestors_after_do == {"W1"}

        # ... and after Step 3 the graph splits into three C-components.
        assert admg.get_subgraph(variables - {"X", "W1"}).get_district() == {
            frozenset({"Y1"}),
            frozenset({"W2"}),
            frozenset({"Y2"}),
        }

    def test_figure_1b_is_not_identifiable(self):
        """ "The very same effect in a very similar graph is not identifiable due to the presence of C-forests forming
        a hedge" (p. 1224)."""
        algorithm = ID()
        assert algorithm.identify(graph(FIGURE_1B, {"X"}, {"Y1", "Y2"})) is False

        # Theorem 6: the pair thrown by Step 5 witnesses a hedge for P_x'(y') for some X' in X, Y' in Y, so it is
        # the graph local to the failing call, not the whole of G. Here Steps 3 and 2 have already narrowed it to
        # P_{w1,x}(y1) on An(Y1) before the hedge is found.
        forest, subforest = algorithm.hedge_
        assert set(forest.nodes()) == {"W1", "X", "Y1"}
        assert sorted(forest.get_edges(data=True)) == [
            ("W1", "X", "->"),
            ("W1", "X", "<>"),
            ("W1", "Y1", "<>"),
            ("X", "Y1", "->"),
        ]
        # F is a single C-component rooted at Y1, and dropping the directed W1 -> X edge leaves the C-forest the paper
        # describes.
        assert forest.get_district() == {frozenset({"W1", "X", "Y1"})}
        assert set(subforest.nodes()) == {"Y1"}

    def test_bow_arc_is_not_identifiable(self):
        """Theorem 2: P_x(Y) is not identifiable in the bow arc graph."""
        algorithm = ID()
        assert algorithm.identify(graph(BOW_ARC, {"X"}, {"Y"})) is False

        forest, subforest = algorithm.hedge_
        assert set(forest.nodes()) == {"X", "Y"}
        assert sorted(forest.get_edges(data=True)) == [("X", "Y", "->"), ("X", "Y", "<>")]
        # F' is the root of the C-tree, exactly as the paper notes for C-trees: "F is the C-tree itself, and F' is the
        # singleton root Y".
        assert set(subforest.nodes()) == {"Y"}


class TestIDIdentifiableFormulas:
    """Identifiable effects, asserted against their exact closed-form expression."""

    def test_chain(self):
        """X -> M -> Y is Markovian: P_x(y) = sum_m P(m|x) P(y|m,x)."""
        result = ID().identify(graph(CHAIN, {"X"}, {"Y"}))
        assert result.to_latex() == r"\sum_{M} P(M \mid X) P(Y \mid M, X)"

    def test_back_door(self):
        """Z confounds X and Y: P_x(y) = sum_z P(z) P(y|x,z)."""
        result = ID().identify(graph(BACK_DOOR, {"X"}, {"Y"}))
        assert result.to_latex() == r"\sum_{Z} P(Z) P(Y \mid X, Z)"

    def test_front_door(self):
        r"""P_x(y) = \sum_m P(m|x) \sum_{x'} P(x') P(y|m,x').

        The inner ``\sum_{X}`` is produced by Step 7 of the algorithm. An implementation that never reaches Step 7
        returns the shorter -- and wrong -- ``\sum_{M} P(M|X) P(Y|M,X)``, which has the same tree shape.
        """
        result = ID().identify(graph(FRONT_DOOR, {"X"}, {"Y"}))
        assert result.to_latex() == (r"\sum_{M} P(M \mid X) \left[ \sum_{X} P(X) P(Y \mid M, X) \right]")

    def test_napkin(self):
        """The napkin graph. Its estimand is a ratio, which only appears if Steps 6 and 7 factorise the estimand
        carried into the call rather than the original observational joint."""
        result = ID().identify(graph(NAPKIN, {"X"}, {"Y"}))
        assert result.to_latex() == (
            r"\frac{\sum_{W} P(W) P(X \mid W, Z) P(Y \mid W, X, Z)}"
            r"{\sum_{W, Y} P(W) P(X \mid W, Z) P(Y \mid W, X, Z)}"
        )
        assert DivisionNode in result.collect_node_types()

    def test_mediator_confounded_with_outcome(self):
        """X -> M -> Y with M <> Y. {M, Y} is a C-component of G, so Step 6 returns the chain factorisation
        directly."""
        result = ID().identify(graph(FRONT_DOOR[:2] + [("M", "Y", "<>")], {"X"}, {"Y"}))
        assert result.to_latex() == r"\sum_{M} P(M \mid X) P(Y \mid M, X)"

    def test_descendants_of_the_outcome_are_dropped(self):
        """Step 2 restricts the problem to An(Y), so the extra descendant D leaves the front-door formula
        unchanged."""
        result = ID().identify(graph(FRONT_DOOR + [("Y", "D", "->")], {"X"}, {"Y"}))
        assert result.to_latex() == (r"\sum_{M} P(M \mid X) \left[ \sum_{X} P(X) P(Y \mid M, X) \right]")

    def test_multiple_exposures(self):
        result = ID().identify(graph([("X1", "Y", "->"), ("X2", "Y", "->")], {"X1", "X2"}, {"Y"}))
        assert result.to_latex() == r"P(Y \mid X1, X2)"

    def test_multiple_outcomes(self):
        """Y1 and Y2 are independent given X, so the joint effect factorises."""
        result = ID().identify(graph([("X", "Y1", "->"), ("X", "Y2", "->")], {"X"}, {"Y1", "Y2"}))
        assert result.to_latex() == r"P(Y1 \mid X) P(Y2 \mid X)"

    def test_no_intervention_reduces_to_marginalisation(self):
        """Step 1: with an empty x the effect is sum_{v \\ y} P(v)."""
        admg = ADMG(edge_list=CHAIN, exposures={"X"}, outcomes={"Y"})
        result = ID()._identify_recursive(
            outcomes=frozenset({"Y"}),
            exposures=frozenset(),
            variables=frozenset({"X", "M", "Y"}),
            causal_graph=admg,
            estimand=ProbabilityNode(frozenset({"X", "M", "Y"})),
            ordering=["X", "M", "Y"],
        )
        assert result == ProbabilityNode(frozenset({"Y"}))


class TestIDNonIdentifiable:
    """Effects blocked by a hedge. Every one returns False and a witness pair."""

    @pytest.mark.parametrize(
        ("name", "edge_list", "exposures", "outcomes"),
        [
            ("bow arc", BOW_ARC, {"X"}, {"Y"}),
            ("extended bow arc", EXTENDED_BOW_ARC, {"X"}, {"Y"}),
            ("instrumental variable", INSTRUMENTAL_VARIABLE, {"X"}, {"Y"}),
            ("figure 1 (b)", FIGURE_1B, {"X"}, {"Y1", "Y2"}),
        ],
    )
    def test_returns_false_with_a_hedge(self, name, edge_list, exposures, outcomes):
        algorithm = ID()
        assert algorithm.identify(graph(edge_list, exposures, outcomes)) is False

        forest, subforest = algorithm.hedge_
        # Definition 6: F' is a subset of F, F meets X, and F' does not.
        assert set(subforest.nodes()) < set(forest.nodes())
        assert set(forest.nodes()) & exposures
        assert not (set(subforest.nodes()) & exposures)

    def test_failure_propagates_out_of_the_step_4_decomposition(self):
        """In the extended bow arc, Step 4 splits G \\ X into {Z} and {Y}. Only the {Z} subproblem hedges, and that
        FAIL must abort the whole product rather than being swallowed."""
        algorithm = ID()
        assert algorithm.identify(graph(EXTENDED_BOW_ARC, {"X"}, {"Y"})) is False
        forest, subforest = algorithm.hedge_
        assert set(forest.nodes()) == {"X", "Z"}
        assert set(subforest.nodes()) == {"Z"}

    def test_hedge_is_reset_between_runs(self):
        algorithm = ID()
        assert algorithm.identify(graph(BOW_ARC, {"X"}, {"Y"})) is False
        assert algorithm.hedge_ is not None
        assert algorithm.identify(graph(CHAIN, {"X"}, {"Y"})) is not False
        assert algorithm.hedge_ is None


class TestIDInputHandling:
    def test_dag_input(self):
        dag = DAG(ebunch=[("X", "M"), ("M", "Y")], exposures={"X"}, outcomes={"Y"})
        result = ID().identify(dag)
        assert result.to_latex() == r"\sum_{M} P(M \mid X) P(Y \mid M, X)"

    def test_dag_input_keeps_isolated_nodes(self):
        """Converting a DAG through its edge list alone silently drops nodes that have no edges; an isolated outcome
        would then raise."""
        dag = DAG(ebunch=[("A", "B")], exposures={"A"})
        # The outcome role can only be assigned once the isolated node exists.
        dag.add_node("Y")
        dag = dag.with_role("outcomes", ["Y"])
        result = ID().identify(dag)
        assert result.to_latex() == r"P(Y)"

    def test_dag_latents_are_projected_to_bidirected_edges(self):
        """A DAG that names its confounder must behave like the bow arc, not like a back-door graph: U is unobserved,
        so no formula may condition on it."""
        dag = DAG(ebunch=[("U", "X"), ("U", "Y"), ("X", "Y")], latents={"U"})
        dag = dag.with_role("exposures", ["X"]).with_role("outcomes", ["Y"])
        assert ID().identify(dag) is False

    def test_dag_latent_projection_matches_the_hand_written_admg(self):
        """The front-door graph written with an explicit confounder projects onto the ADMG fixture."""
        dag = DAG(ebunch=[("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")], latents={"U"})
        dag = dag.with_role("exposures", ["X"]).with_role("outcomes", ["Y"])
        assert ID().identify(dag).to_latex() == ID().identify(graph(FRONT_DOOR, {"X"}, {"Y"})).to_latex()

    def test_dag_latent_projection_bridges_directed_paths_through_latents(self):
        """A latent lying on a directed path becomes a directed edge, not a bidirected one."""
        dag = DAG(ebunch=[("X", "U"), ("U", "Y")], latents={"U"})
        dag = dag.with_role("exposures", ["X"]).with_role("outcomes", ["Y"])
        assert ID().identify(dag).to_latex() == r"P(Y \mid X)"

    def test_dag_latent_mediator_is_not_treated_as_a_confounder(self):
        """A latent with a single observed descendant opens no back-door path, so the effect stays identifiable."""
        dag = DAG(ebunch=[("X", "U"), ("U", "Y"), ("X", "Y")], latents={"U"})
        dag = dag.with_role("exposures", ["X"]).with_role("outcomes", ["Y"])
        assert ID().identify(dag).to_latex() == r"P(Y \mid X)"

    @pytest.mark.parametrize("role", ["exposures", "outcomes"])
    def test_latent_exposures_and_outcomes_are_rejected(self, role):
        dag = DAG(ebunch=[("U", "X"), ("X", "Y")], latents={"U"})
        dag = dag.with_role(role, ["U"]).with_role("outcomes" if role == "exposures" else "exposures", ["Y"])
        with pytest.raises(ValueError, match="cannot be both latent"):
            ID().identify(dag)

    def test_exposures_and_outcomes_must_be_disjoint(self):
        """Definition 2 defines P_x(Y) only when X and Y are disjoint."""
        with pytest.raises(ValueError, match="disjoint"):
            ID().identify(graph(CHAIN, {"X", "Y"}, {"Y"}))

    def test_unsupported_graph_type(self):
        with pytest.raises(ValueError, match="must be an instance of"):
            ID().identify("not a graph")

    def test_returns_an_expression_tree(self):
        result = ID().identify(graph(CHAIN, {"X"}, {"Y"}))
        assert isinstance(result, ProbabilityExpressionTree)
        assert isinstance(result.root, MarginalNode)


class TestIDHelpers:
    """The rules that Steps 1-7 lean on, tested directly."""

    def test_district_product_of_a_singleton_district(self):
        """ProductNode rejects a single factor, but a district may well be a singleton -- the lone factor is then its
        own product."""
        result = ID()._district_product(
            estimand=ProbabilityNode(frozenset({"X", "Y"})),
            district=frozenset({"Y"}),
            ordered=["X", "Y"],
        )
        assert result == ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X"}))

    def test_district_product_of_a_plain_joint_is_atomic(self):
        """With the original P(v) still carried, the chain-rule conditionals are atomic terms and no division is
        needed."""
        result = ID()._district_product(
            estimand=ProbabilityNode(frozenset({"X", "M", "Y"})),
            district=frozenset({"X", "Y"}),
            ordered=["X", "M", "Y"],
        )
        assert result == ProductNode(
            [
                ProbabilityNode(frozenset({"X"})),
                ProbabilityNode(frozenset({"Y"}), cond=frozenset({"X", "M"})),
            ]
        )

    def test_district_product_of_a_derived_estimand_divides(self):
        """Once Step 7 has replaced P by Q[S'], a conditional of it can only be expressed as a ratio of two marginals
        of that Q[S']."""
        estimand = MarginalNode(ProbabilityNode(frozenset({"X", "Y"})), sumset=frozenset({"Z"}))
        result = ID()._district_product(estimand=estimand, district=frozenset({"Y"}), ordered=["X", "Y"])
        assert isinstance(result, DivisionNode)
        numerator, denominator = result.children
        assert numerator is estimand
        assert denominator == MarginalNode(estimand.children[0], sumset=frozenset({"Y", "Z"}))
