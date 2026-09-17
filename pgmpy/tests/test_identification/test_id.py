"""Tests for the ID algorithm of Shpitser & Pearl (AAAI 2006) and the IDC algorithm of Shpitser & Pearl (UAI 2006)."""

from itertools import permutations

import pytest

from pgmpy.base import ADMG, DAG
from pgmpy.identification import ID, IDC
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


# Fig. 1 of the conditional paper (p. 1220), read off the figure. The three graphs share X -> Z <> X and differ in
# how Y attaches to Z; (c) adds W. The paper's own claims about them are asserted in ``TestIDCPaperExamples``.
CONDITIONAL_FIGURE_1A = [("X", "Z", "->"), ("Z", "Y", "->"), ("X", "Z", "<>")]
CONDITIONAL_FIGURE_1B = [("X", "Z", "->"), ("Y", "Z", "->"), ("X", "Z", "<>")]
CONDITIONAL_FIGURE_1C = CONDITIONAL_FIGURE_1B + [("W", "Y", "->")]

# Z0 -> X1 -> Z -> X2 -> Y, the shape of the sequential decision problem of Section 2: the second treatment X2 is
# chosen after seeing the intermediate observation Z, which is confounded with the outcome.
SEQUENTIAL_PLAN = [
    ("X1", "Z", "->"),
    ("Z", "X2", "->"),
    ("X2", "Y", "->"),
    ("X1", "Y", "->"),
    ("Z", "Y", "<>"),
]

# Z is a collider on the path from the context variable to the outcome, so Y and Z are dependent given X in G, but
# independent in G with the incoming edges of X removed.
COLLIDER_AT_EXPOSURE = [("Z", "X", "->"), ("W", "X", "->"), ("W", "Y", "->"), ("X", "Y", "->")]


def graph(edge_list, exposures, outcomes):
    """Build an ADMG with the exposure and outcome roles assigned."""
    return ADMG(edge_list=edge_list, exposures=set(exposures), outcomes=set(outcomes))


def conditional_graph(edge_list, exposures, outcomes, conditioning):
    """Build an ADMG with the exposure, outcome and conditioning roles assigned."""
    return ADMG(
        edge_list=edge_list,
        exposures=set(exposures),
        outcomes=set(outcomes),
        roles={"conditioning": set(conditioning)},
    )


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


class TestIDCPaperExamples:
    """Cases whose expected output is stated in the conditional paper itself."""

    def test_figure_1a_matches_the_papers_worked_example(self):
        """Fig. 1 (a), p. 1223.

        "Because Y || Z | X, Z in G_{x_bar, z_under}, rule 2 applies and we call the algorithm again with the
        expression P_{x,z}(Y). This expression is an unconditional effect, so we call ID as a subroutine. ID, in turn,
        succeeds immediately on line 6, returning the expression P(y|x,z)."
        """
        algorithm = IDC()
        admg = conditional_graph(CONDITIONAL_FIGURE_1A, {"X"}, {"Y"}, {"Z"})

        assert algorithm._is_rule_2_applicable(
            "Z",
            outcomes=frozenset({"Y"}),
            exposures=frozenset({"X"}),
            conditioning=frozenset({"Z"}),
            causal_graph=admg,
        )
        assert algorithm.identify(admg).to_latex() == r"P(Y \mid X, Z)"

    def test_figure_1a_conditioning_helps(self):
        """ "Here P_x(y) is not identifiable in G in Fig. 1 (a)" (p. 1219), yet P_x(y|z) is: "conditioning on Z renders
        Y independent of any changes to X, making P_x(y|z) equal to P(y|z)"."""
        assert ID().identify(graph(CONDITIONAL_FIGURE_1A, {"X"}, {"Y"})) is False
        assert IDC().identify(conditional_graph(CONDITIONAL_FIGURE_1A, {"X"}, {"Y"}, {"Z"})) is not False

    def test_figure_1a_c_components(self):
        """ "the graph in Fig. 1 (a) has two C-components, {X, Z} and {Y}" (p. 1221)."""
        admg = graph(CONDITIONAL_FIGURE_1A, {"X"}, {"Y"})
        assert admg.get_district() == {frozenset({"X", "Z"}), frozenset({"Y"})}

    def test_figure_1b_conditioning_hinders(self):
        """ "in G', conditioning on Z makes X and Y dependent, resulting in P_x(y|z) becoming non-identifiable"
        (p. 1219), while P_x(y) on its own "is identifiable and equal to P(y)"."""
        assert ID().identify(graph(CONDITIONAL_FIGURE_1B, {"X"}, {"Y"})).to_latex() == r"P(Y)"
        assert IDC().identify(conditional_graph(CONDITIONAL_FIGURE_1B, {"X"}, {"Y"}, {"Z"})) is False

    def test_figure_1c_is_the_counterexample_to_tians_algorithm(self):
        """Lemma 3, p. 1224: "Consider the graph G'' in Fig. 1 (c). We will consider the conditional effect P_x(w|z)
        in this graph. Note that by the back-door hedge criterion this effect is not identifiable in G''." Tian's
        cond-identify wrongly succeeds on it; IDC must fail."""
        assert IDC().identify(conditional_graph(CONDITIONAL_FIGURE_1C, {"X"}, {"W"}, {"Z"})) is False

    def test_figure_1c_intermediate_quantities_match_the_paper(self):
        """The trace of Lemma 3 pins down the graph: "D = {Y, Z, W}, F = {Y}, C(G) = {{X, Z}, {Y}, {W}},
        C(D) = {{Z}, {Y}, {W}}", and W is reached from Y only through Z, since "{Y} is not a parent of any identifiable
        C-component"."""
        admg = graph(CONDITIONAL_FIGURE_1C, {"X"}, {"W"})

        # D = An(Y u Z) in G with X removed, for the query P_x(w|z), i.e. with the roles of the trace.
        district = admg.get_subgraph(set(admg.nodes()) - {"X"})
        assert district.get_ancestors({"W", "Z"}) == {"Y", "Z", "W"}
        assert admg.get_district() == {frozenset({"X", "Z"}), frozenset({"Y"}), frozenset({"W"})}
        assert district.get_district() == {frozenset({"Z"}), frozenset({"Y"}), frozenset({"W"})}
        assert admg.get_parents({"W"}) == set()


class TestIDCStep1:
    """Step 1: rule 2 of do-calculus moves context variables into ``do()``."""

    def test_every_removable_context_variable_is_absorbed(self):
        """Corollary 1: Step 1 recurses until the maximal set has been moved, so both Z1 and Z2 end up behind
        ``do()`` and the answer is a single atomic term rather than a ratio."""
        edge_list = [("Z1", "X", "->"), ("Z2", "X", "->"), ("Z1", "Y", "->"), ("X", "Y", "->")]
        result = IDC().identify(conditional_graph(edge_list, {"X"}, {"Y"}, {"Z1", "Z2"}))
        assert result.to_latex() == r"P(Y \mid X, Z1, Z2)"
        assert DivisionNode not in result.collect_node_types()

    def test_absorbing_one_context_variable_neither_unlocks_nor_blocks_another(self):
        """Lemma 2: "an application of rule 2 on any set does not influence future applications of the rule on other
        sets elsewhere in the graph"."""
        edge_list = [("Z1", "X", "->"), ("Z2", "X", "->"), ("Z1", "Y", "->"), ("X", "Y", "->")]
        admg = conditional_graph(edge_list, {"X"}, {"Y"}, {"Z1", "Z2"})
        algorithm = IDC()

        before = algorithm._is_rule_2_applicable(
            "Z2",
            outcomes=frozenset({"Y"}),
            exposures=frozenset({"X"}),
            conditioning=frozenset({"Z1", "Z2"}),
            causal_graph=admg,
        )
        after = algorithm._is_rule_2_applicable(
            "Z2",
            outcomes=frozenset({"Y"}),
            exposures=frozenset({"X", "Z1"}),
            conditioning=frozenset({"Z2"}),
            causal_graph=admg,
        )
        assert before is after is True

    def test_absorption_order_does_not_change_the_result(self):
        """Corollary 1: the maximal set rule 2 applies to is unique, so which qualifying variable Step 1 happens to
        pick first -- here decided by a topological order that varies with edge insertion order -- cannot matter."""
        edge_list = [("Z1", "X", "->"), ("Z2", "X", "->"), ("Z1", "Y", "->"), ("X", "Y", "->")]
        results = {
            IDC().identify(conditional_graph(list(order), {"X"}, {"Y"}, {"Z1", "Z2"})).to_latex()
            for order in permutations(edge_list)
        }
        assert results == {r"P(Y \mid X, Z1, Z2)"}

    def test_a_context_variable_with_a_back_door_path_stays_behind_the_bar(self):
        """Rule 2 does not apply to a descendant of the outcome, so Z survives Step 1 and Step 2 has to divide."""
        algorithm = IDC()
        admg = conditional_graph([("X", "Y", "->"), ("Y", "Z", "->")], {"X"}, {"Y"}, {"Z"})

        assert not algorithm._is_rule_2_applicable(
            "Z",
            outcomes=frozenset({"Y"}),
            exposures=frozenset({"X"}),
            conditioning=frozenset({"Z"}),
            causal_graph=admg,
        )
        assert DivisionNode in algorithm.identify(admg).collect_node_types()

    def test_a_bidirected_edge_at_the_context_variable_survives_the_underline(self):
        """G_{x_bar, z_under} deletes the arrows *out of* Z. Only directed edges leave a node, so the confounding arc
        Z <> Y stays and keeps Z behind the conditioning bar."""
        edge_list = [("Z", "X", "->"), ("X", "Y", "->"), ("Z", "Y", "<>")]
        algorithm = IDC()

        assert not algorithm._is_rule_2_applicable(
            "Z",
            outcomes=frozenset({"Y"}),
            exposures=frozenset({"X"}),
            conditioning=frozenset({"Z"}),
            causal_graph=graph(edge_list, {"X"}, {"Y"}),
        )
        # Had the arc been cut with the rest of Z's edges, Z would have been absorbed and the answer would have been
        # the sum over Z of the same product instead of a ratio.
        result = algorithm.identify(conditional_graph(edge_list, {"X"}, {"Y"}, {"Z"}))
        assert result.to_latex() == (r"\frac{P(Z) P(Y \mid X, Z)}{\sum_{Y} P(Z) P(Y \mid X, Z)}")

    def test_the_incoming_edges_of_the_exposures_are_cut(self):
        """G_{x_bar, z_under} also deletes the arrows *into* X. Without that, the collider at X -- which rule 2
        conditions on -- would open a path from Z to Y and Z would never be absorbed."""
        algorithm = IDC()
        admg = conditional_graph(COLLIDER_AT_EXPOSURE, {"X"}, {"Y"}, {"Z"})

        assert not admg.is_mseparated("Z", "Y", conditioning_set={"X"})
        assert algorithm._is_rule_2_applicable(
            "Z",
            outcomes=frozenset({"Y"}),
            exposures=frozenset({"X"}),
            conditioning=frozenset({"Z"}),
            causal_graph=admg,
        )
        assert algorithm.identify(admg).to_latex() == r"\sum_{W} P(W) P(Y \mid W, X, Z)"


class TestIDCStep2:
    """Step 2: the remaining problem is handed to ID as a joint effect and turned into a conditional by dividing."""

    def test_the_ratio_is_the_id_expression_over_its_outcome_marginal(self):
        """Step 2 returns P' / sum_y P', where P' is exactly what ID returns for the joint effect P_x(y, z)."""
        edge_list = [("X", "Y", "->"), ("Y", "Z", "->")]
        result = IDC().identify(conditional_graph(edge_list, {"X"}, {"Y"}, {"Z"}))
        joint = ID().identify(graph(edge_list, {"X"}, {"Y", "Z"}))

        assert isinstance(result.root, DivisionNode)
        numerator, denominator = result.root.children
        assert numerator == joint.root
        assert denominator == joint.root.marginalize({"Y"})

    def test_an_empty_context_reduces_to_id(self):
        """P_x(y) already sums to one over y, so with nothing left behind the bar Step 2 is just a call to ID. The
        public entry point requires a conditioning role, so the step is exercised directly."""
        admg = graph(FRONT_DOOR, {"X"}, {"Y"})
        result = IDC()._identify_conditional(
            outcomes=frozenset({"Y"}),
            exposures=frozenset({"X"}),
            conditioning=frozenset(),
            causal_graph=admg,
        )
        assert result == ID().identify(admg).root

    def test_failure_of_the_id_subcall_propagates(self):
        """Theorem 6: the conditional effect is identifiable exactly when the joint one is, so a hedge thrown by the
        ID subcall is the witness for the conditional effect too."""
        algorithm = IDC()
        assert algorithm.identify(conditional_graph(CONDITIONAL_FIGURE_1B, {"X"}, {"Y"}, {"Z"})) is False

        id_algorithm = ID()
        assert id_algorithm.identify(graph(CONDITIONAL_FIGURE_1B, {"X"}, {"Y", "Z"})) is False

        # The two graphs carry different roles, so compare the structure the hedge is made of.
        for from_idc, from_id in zip(algorithm.hedge_, id_algorithm.hedge_):
            assert set(from_idc.nodes()) == set(from_id.nodes())
            assert sorted(from_idc.get_edges(data=True)) == sorted(from_id.get_edges(data=True))


class TestIDCIdentifiableFormulas:
    """Identifiable conditional effects, asserted against their exact closed-form expression."""

    def test_conditional_back_door(self):
        """Z confounds X and Y. Rule 2 absorbs Z, leaving the integrand of the back-door formula."""
        result = IDC().identify(conditional_graph(BACK_DOOR, {"X"}, {"Y"}, {"Z"}))
        assert result.to_latex() == r"P(Y \mid X, Z)"

    def test_conditional_front_door(self):
        """P_x(y|m) is the second factor of the front-door formula: rule 2 absorbs the mediator, and ID then has to
        go through Step 7 to get rid of the confounded X."""
        result = IDC().identify(conditional_graph(FRONT_DOOR, {"X"}, {"Y"}, {"M"}))
        assert result.to_latex() == r"\sum_{X} P(X) P(Y \mid M, X)"

    def test_chain(self):
        """Y is independent of X given M, and the returned P(y|m,x) is that same quantity."""
        result = IDC().identify(conditional_graph(CHAIN, {"X"}, {"Y"}, {"M"}))
        assert result.to_latex() == r"P(Y \mid M, X)"

    def test_context_that_is_a_descendant_of_the_outcome(self):
        """Nothing can be absorbed, so the answer is the ratio of the joint P_x(y, z) to its outcome marginal."""
        result = IDC().identify(conditional_graph([("X", "Y", "->"), ("Y", "Z", "->")], {"X"}, {"Y"}, {"Z"}))
        assert result.to_latex() == (r"\frac{P(Y \mid X) P(Z \mid X, Y)}{\sum_{Y} P(Y \mid X) P(Z \mid X, Y)}")

    def test_sequential_plan(self):
        """Section 2: the value of a conditional plan is assembled from terms P_{x,z}(y|z), where the intermediate
        observation Z is confounded with the outcome and so stays behind the bar."""
        result = IDC().identify(conditional_graph(SEQUENTIAL_PLAN, {"X1", "X2"}, {"Y"}, {"Z"}))
        assert result.to_latex() == (
            r"\frac{P(Z \mid X1) P(Y \mid X1, X2, Z)}{\sum_{Y} P(Z \mid X1) P(Y \mid X1, X2, Z)}"
        )

    def test_a_context_variable_outside_the_effect_is_dropped_by_id(self):
        """An irrelevant context variable is absorbed by rule 2 and then discarded by Step 2 of ID, which restricts
        the problem to An(Y)."""
        result = IDC().identify(conditional_graph(CHAIN + [("Y", "D", "->")], {"X"}, {"Y"}, {"M"}))
        assert result.to_latex() == r"P(Y \mid M, X)"


class TestIDCNonIdentifiable:
    """Conditional effects blocked by a hedge. Every one returns False and a witness pair."""

    @pytest.mark.parametrize(
        ("name", "edge_list", "exposures", "outcomes", "conditioning"),
        [
            ("figure 1 (b)", CONDITIONAL_FIGURE_1B, {"X"}, {"Y"}, {"Z"}),
            ("figure 1 (c)", CONDITIONAL_FIGURE_1C, {"X"}, {"W"}, {"Z"}),
            ("bow arc", BOW_ARC + [("X", "Z", "->")], {"X"}, {"Y"}, {"Z"}),
            ("napkin", NAPKIN, {"X"}, {"Y"}, {"W"}),
        ],
    )
    def test_returns_false_with_a_hedge(self, name, edge_list, exposures, outcomes, conditioning):
        algorithm = IDC()
        assert algorithm.identify(conditional_graph(edge_list, exposures, outcomes, conditioning)) is False

        forest, subforest = algorithm.hedge_
        # Definition 5: F' is a subset of F, F meets X, and F' does not.
        assert set(subforest.nodes()) < set(forest.nodes())
        assert set(forest.nodes()) & exposures
        assert not (set(subforest.nodes()) & exposures)

    def test_napkin_effect_is_identifiable_only_unconditionally(self):
        """The napkin effect P_x(y) is identifiable, but conditioning on the confounded W is not free: rule 2 cannot
        absorb W, and the joint P_x(y, w) is hedged by {W, X, Y} over {W, Y}."""
        assert ID().identify(graph(NAPKIN, {"X"}, {"Y"})) is not False

        algorithm = IDC()
        assert algorithm.identify(conditional_graph(NAPKIN, {"X"}, {"Y"}, {"W"})) is False
        forest, subforest = algorithm.hedge_
        assert set(forest.nodes()) == {"W", "X", "Y"}
        assert set(subforest.nodes()) == {"W", "Y"}

    def test_hedge_is_reset_between_runs(self):
        algorithm = IDC()
        assert algorithm.identify(conditional_graph(CONDITIONAL_FIGURE_1B, {"X"}, {"Y"}, {"Z"})) is False
        assert algorithm.hedge_ is not None
        assert algorithm.identify(conditional_graph(CONDITIONAL_FIGURE_1A, {"X"}, {"Y"}, {"Z"})) is not False
        assert algorithm.hedge_ is None


class TestIDCTheorem6:
    """Theorem 6: with a context that rule 2 cannot touch, P_x(y|z) is identifiable iff P_x(y, z) is."""

    @pytest.mark.parametrize(
        ("name", "edge_list", "exposures", "outcomes", "conditioning"),
        [
            ("descendant of the outcome", [("X", "Y", "->"), ("Y", "Z", "->")], {"X"}, {"Y"}, {"Z"}),
            ("sequential plan", SEQUENTIAL_PLAN, {"X1", "X2"}, {"Y"}, {"Z"}),
            ("figure 1 (b)", CONDITIONAL_FIGURE_1B, {"X"}, {"Y"}, {"Z"}),
            ("figure 1 (c)", CONDITIONAL_FIGURE_1C, {"X"}, {"W"}, {"Z"}),
            ("napkin", NAPKIN, {"X"}, {"Y"}, {"W"}),
        ],
    )
    def test_matches_the_joint_effect(self, name, edge_list, exposures, outcomes, conditioning):
        algorithm = IDC()
        admg = conditional_graph(edge_list, exposures, outcomes, conditioning)

        # The theorem is stated for a context that has a back-door path to Y, i.e. one Step 1 leaves untouched.
        assert not any(
            algorithm._is_rule_2_applicable(
                node,
                outcomes=frozenset(outcomes),
                exposures=frozenset(exposures),
                conditioning=frozenset(conditioning),
                causal_graph=admg,
            )
            for node in conditioning
        )

        conditional = algorithm.identify(admg)
        joint = ID().identify(graph(edge_list, exposures, outcomes | conditioning))
        assert (conditional is False) == (joint is False)


class TestIDCInputHandling:
    def test_dag_input(self):
        dag = DAG(ebunch=[("X", "M"), ("M", "Y")], exposures={"X"}, outcomes={"Y"})
        dag = dag.with_role("conditioning", ["M"])
        result = IDC().identify(dag)
        assert result.to_latex() == r"P(Y \mid M, X)"

    @pytest.mark.parametrize(
        ("exposures", "outcomes", "conditioning", "message"),
        [
            ({"X", "Y"}, {"Y"}, {"M"}, "'exposures' and 'outcomes' must be disjoint"),
            ({"X", "M"}, {"Y"}, {"M"}, "'exposures' and 'conditioning' must be disjoint"),
            ({"X"}, {"Y", "M"}, {"M"}, "'outcomes' and 'conditioning' must be disjoint"),
        ],
    )
    def test_roles_must_be_pairwise_disjoint(self, exposures, outcomes, conditioning, message):
        """Definition 2 defines P_x(y|z) only when X, Y and Z are disjoint."""
        with pytest.raises(ValueError, match=message):
            IDC().identify(conditional_graph(CHAIN, exposures, outcomes, conditioning))

    def test_conditioning_role_is_required(self):
        with pytest.raises(ValueError, match="'conditioning' role"):
            IDC().identify(graph(CHAIN, {"X"}, {"Y"}))

    def test_unsupported_graph_type(self):
        with pytest.raises(ValueError, match="must be an instance of"):
            IDC().identify("not a graph")

    def test_returns_an_expression_tree(self):
        result = IDC().identify(conditional_graph(CHAIN, {"X"}, {"Y"}, {"M"}))
        assert isinstance(result, ProbabilityExpressionTree)
        assert isinstance(result.root, ProbabilityNode)
