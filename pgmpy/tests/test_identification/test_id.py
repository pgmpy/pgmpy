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
# p. 1224, whose final formula the "figure 1 (a)" case of ``test_formula`` asserts.
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

CHAIN_FORMULA = r"\sum_{M} P(M \mid X) P(Y \mid M, X)"
FRONT_DOOR_FORMULA = r"\sum_{M} P(M \mid X) \left[ \sum_{X} P(X) P(Y \mid M, X) \right]"


def graph(edge_list, exposures, outcomes):
    """Build an ADMG with the exposure and outcome roles assigned."""
    return ADMG(edge_list=edge_list, exposures=set(exposures), outcomes=set(outcomes))


class TestIDIdentifiable:
    """Identifiable effects, asserted against their exact closed-form expression."""

    @pytest.mark.parametrize(
        ("edge_list", "exposures", "outcomes", "expected"),
        [
            # p. 1224: P_x(y1, y2) = sum_{w2} P(y2 | w2) P(w2) sum_{w1} P(y1 | x, w1) P(w1).
            pytest.param(
                FIGURE_1A,
                {"X"},
                {"Y1", "Y2"},
                r"\sum_{W2} P(W2) P(Y2 \mid W2) \left[ \sum_{W1} P(W1) P(Y1 \mid W1, X) \right]",
                id="figure 1 (a)",
            ),
            pytest.param(CHAIN, {"X"}, {"Y"}, CHAIN_FORMULA, id="chain"),
            pytest.param(BACK_DOOR, {"X"}, {"Y"}, r"\sum_{Z} P(Z) P(Y \mid X, Z)", id="back-door"),
            # The inner sum over X comes from Step 7; skipping Step 7 gives the wrong CHAIN_FORMULA.
            pytest.param(FRONT_DOOR, {"X"}, {"Y"}, FRONT_DOOR_FORMULA, id="front-door"),
            # The ratio only appears if Steps 6 and 7 factorise the carried estimand, not the observational joint.
            pytest.param(
                NAPKIN,
                {"X"},
                {"Y"},
                r"\frac{\sum_{W} P(W) P(X \mid W, Z) P(Y \mid W, X, Z)}"
                r"{\sum_{W, Y} P(W) P(X \mid W, Z) P(Y \mid W, X, Z)}",
                id="napkin",
            ),
            # {M, Y} is a C-component of G, so Step 6 returns the chain factorisation directly.
            pytest.param(CHAIN + [("M", "Y", "<>")], {"X"}, {"Y"}, CHAIN_FORMULA, id="mediator confounded with Y"),
            # Step 2 restricts the problem to An(Y), so the descendant D leaves the formula unchanged.
            pytest.param(FRONT_DOOR + [("Y", "D", "->")], {"X"}, {"Y"}, FRONT_DOOR_FORMULA, id="descendant of Y"),
            pytest.param([("X1", "Y", "->"), ("X2", "Y", "->")], {"X1", "X2"}, {"Y"}, r"P(Y \mid X1, X2)", id="two X"),
            pytest.param(
                [("X", "Y1", "->"), ("X", "Y2", "->")], {"X"}, {"Y1", "Y2"}, r"P(Y1 \mid X) P(Y2 \mid X)", id="two Y"
            ),
            # Step 1: with no exposures the effect is sum_{v \ y} P(v).
            pytest.param(CHAIN, set(), {"Y"}, "P(Y)", id="no exposures"),
        ],
    )
    def test_formula(self, edge_list, exposures, outcomes, expected):
        result = ID().identify(graph(edge_list, exposures, outcomes))
        assert isinstance(result, ProbabilityExpressionTree)
        assert result.to_latex() == expected

    @pytest.mark.parametrize(
        ("ebunch", "latents", "expected"),
        [
            pytest.param([("X", "M"), ("M", "Y")], set(), CHAIN_FORMULA, id="no latents"),
            # U is unobserved, so this is the bow arc, not a back-door graph.
            pytest.param([("U", "X"), ("U", "Y"), ("X", "Y")], {"U"}, False, id="latent confounder"),
            pytest.param([("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")], {"U"}, FRONT_DOOR_FORMULA, id="front-door"),
            # A latent on a directed path becomes a directed edge, not a bidirected one.
            pytest.param([("X", "U"), ("U", "Y")], {"U"}, r"P(Y \mid X)", id="latent on a path"),
            pytest.param([("X", "U"), ("U", "Y"), ("X", "Y")], {"U"}, r"P(Y \mid X)", id="latent mediator"),
        ],
    )
    def test_dag_latents_are_projected(self, ebunch, latents, expected):
        result = ID().identify(DAG(ebunch=ebunch, latents=latents, exposures={"X"}, outcomes={"Y"}))
        assert (result if result is False else result.to_latex()) == expected


class TestIDNonIdentifiable:
    """Effects blocked by a hedge: False, with the witness pair thrown by Step 5 in ``hedge_``."""

    @pytest.mark.parametrize(
        ("edge_list", "exposures", "outcomes", "forest", "subforest"),
        [
            # Theorem 2. F' is the root of the C-tree: "F is the C-tree itself, and F' is the singleton root Y".
            pytest.param(BOW_ARC, {"X"}, {"Y"}, {"X", "Y"}, {"Y"}, id="bow arc"),
            # Step 4 splits G \ X into {Z} and {Y}; only {Z} hedges, and that FAIL must abort the whole product.
            pytest.param(EXTENDED_BOW_ARC, {"X"}, {"Y"}, {"X", "Z"}, {"Z"}, id="extended bow arc"),
            pytest.param(INSTRUMENTAL_VARIABLE, {"X"}, {"Y"}, {"X", "Y"}, {"Y"}, id="instrumental variable"),
            # p. 1224. The hedge is local to the failing call (Theorem 6): Steps 3 and 2 have already narrowed the
            # problem to P_{w1,x}(y1) on An(Y1), and F' = F \ {X} once the directed W1 -> X edge is dropped.
            pytest.param(FIGURE_1B, {"X"}, {"Y1", "Y2"}, {"W1", "X", "Y1"}, {"Y1"}, id="figure 1 (b)"),
        ],
    )
    def test_returns_false_with_the_hedge(self, edge_list, exposures, outcomes, forest, subforest):
        algorithm = ID()
        assert algorithm.identify(graph(edge_list, exposures, outcomes)) is False
        assert set(algorithm.hedge_[0].nodes()) == forest
        assert set(algorithm.hedge_[1].nodes()) == subforest

        # A later identifiable run clears the hedge.
        assert algorithm.identify(graph(CHAIN, {"X"}, {"Y"})) is not False
        assert algorithm.hedge_ is None


class TestIDInputValidation:
    @pytest.mark.parametrize(
        ("causal_graph", "match"),
        [
            pytest.param(
                DAG(ebunch=[("U", "X"), ("X", "Y")], latents={"U"}, exposures={"U"}, outcomes={"Y"}),
                "cannot be both latent",
                id="latent exposure",
            ),
            pytest.param(
                DAG(ebunch=[("U", "X"), ("X", "Y")], latents={"U"}, exposures={"Y"}, outcomes={"U"}),
                "cannot be both latent",
                id="latent outcome",
            ),
            # Definition 2 defines P_x(Y) only when X and Y are disjoint.
            pytest.param(graph(CHAIN, {"X", "Y"}, {"Y"}), "disjoint", id="overlapping roles"),
            pytest.param("not a graph", "must be an instance of", id="unsupported graph type"),
        ],
    )
    def test_invalid_input_raises(self, causal_graph, match):
        with pytest.raises(ValueError, match=match):
            ID().identify(causal_graph)


class TestIDDistrictProduct:
    """``_district_product``, the chain-rule factorisation of Steps 6 and 7."""

    @pytest.mark.parametrize(
        ("estimand", "district", "ordered", "expected"),
        [
            # A district may be a singleton, and ProductNode rejects a single factor.
            pytest.param(
                ProbabilityNode({"X", "Y"}), {"Y"}, ["X", "Y"], ProbabilityNode({"Y"}, cond={"X"}), id="singleton"
            ),
            # With the original P(v) still carried, the conditionals are atomic terms.
            pytest.param(
                ProbabilityNode({"X", "M", "Y"}),
                {"X", "Y"},
                ["X", "M", "Y"],
                ProductNode([ProbabilityNode({"X"}), ProbabilityNode({"Y"}, cond={"X", "M"})]),
                id="plain joint",
            ),
            # Once Step 7 has replaced P by Q[S'], a conditional is a ratio of two marginals of that Q[S'].
            pytest.param(
                MarginalNode(ProbabilityNode({"X", "Y"}), sumset={"Z"}),
                {"Y"},
                ["X", "Y"],
                DivisionNode(
                    MarginalNode(ProbabilityNode({"X", "Y"}), sumset={"Z"}),
                    MarginalNode(ProbabilityNode({"X", "Y"}), sumset={"Y", "Z"}),
                ),
                id="derived estimand",
            ),
        ],
    )
    def test_district_product(self, estimand, district, ordered, expected):
        assert ID()._district_product(estimand=estimand, district=frozenset(district), ordered=ordered) == expected
