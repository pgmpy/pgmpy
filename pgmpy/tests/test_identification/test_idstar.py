"""Tests for the ID* algorithm and make-cg of Shpitser & Pearl (JMLR 2008), Figs. 10 and 12.

Identified formulas are checked numerically: ``_counterfactual_probability`` computes P(gamma) exactly in a random
discrete SCM by enumerating its noise variables, and ``_evaluate`` computes the returned expression from the
interventional distributions of the same SCM.
"""

from itertools import product
from math import prod

import numpy as np
import pytest

from pgmpy.base import ADMG, DAG
from pgmpy.identification import CounterfactualVariable, IDStar
from pgmpy.identification.probability_expression import (
    ConstantNode,
    MarginalNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
)

# Fig. 9 (a), p. 1957.
FIGURE_9A = [("X", "W", "->"), ("W", "Y", "->"), ("D", "Z", "->"), ("Z", "Y", "->"), ("X", "Y", "<>")]

# Fig. 13 (a), p. 1965: the zig-zag graph of Lemma 28.
ZIG_ZAG = [("X", "Y", "->"), ("X", "Z", "->"), ("Y", "W1", "<>"), ("W1", "W2", "<>"), ("W2", "Z", "<>")]

W_GRAPH = [("X", "Y", "->")]
BOW_ARC = [("X", "Y", "->"), ("X", "Y", "<>")]
BACK_DOOR = [("Z", "X", "->"), ("Z", "Y", "->"), ("X", "Y", "->")]
FRONT_DOOR = [("X", "M", "->"), ("M", "Y", "->"), ("X", "Y", "<>")]

# P(y_x, x', z_d, d), the running example of Section 4.
FIGURE_9_QUERY = {
    CounterfactualVariable("Y", {"X": "x"}): "y",
    CounterfactualVariable("X"): "x'",
    CounterfactualVariable("Z", {"D": "d"}): "z",
    CounterfactualVariable("D"): "d",
}

# Every value symbol a test uses, bound to a value of the binary domain of the SCMs below.
BINDINGS = {"x": 1, "x'": 0, "y": 1, "y'": 0, "z": 1, "z'": 0, "d": 0, "w": 1, "m": 1}
DOMAIN = (0, 1)


def admg(edge_list, nodes=()):
    graph = ADMG(edge_list=edge_list)
    graph.add_nodes_from(nodes)
    return graph


def cf(variable, interventions=None):
    return CounterfactualVariable(variable, interventions)


def edges(graph):
    """The edges of ``graph``, with each bidirected edge as an unordered pair."""
    return {(u, v, t) if t == "->" else (frozenset({u, v}), t) for u, v, t in graph.get_edges()}


def bidirected(u, v):
    return frozenset({u, v}), "<>"


def _random_scm(graph, rng):
    """A binary SCM inducing ``graph``: each variable has a noise of its own and each bidirected edge a shared one."""
    order = graph.get_topological_order()
    noises = {("U", v): rng.dirichlet(np.ones(4)) for v in order}
    noises.update({("U", u, v): rng.dirichlet(np.ones(2)) for u, v, _ in graph.get_edges(edge_types=["<>"])})
    noise_of = {v: [name for name in noises if v in name[1:]] for v in order}
    parents = {v: sorted(graph.get_parents(v)) for v in order}
    tables = {
        v: {
            key: int(rng.integers(2))
            for key in product(product(DOMAIN, repeat=len(parents[v])), *[range(len(noises[n])) for n in noise_of[v]])
        }
        for v in order
    }
    return order, noises, noise_of, parents, tables


def _worlds(scm):
    """Yield the probability of each joint noise configuration, with a solver for the world it induces."""
    order, noises, noise_of, parents, tables = scm
    names = list(noises)
    for states in product(*[range(len(noises[n])) for n in names]):
        u = dict(zip(names, states))

        def solve(intervention, u=u):
            values = {}
            for v in order:
                key = (tuple(values[p] for p in parents[v]), *[u[n] for n in noise_of[v]])
                values[v] = intervention[v] if v in intervention else tables[v][key]
            return values

        yield prod(noises[n][s] for n, s in u.items()), solve


def _counterfactual_probability(scm, event, bindings):
    return sum(
        p
        for p, solve in _worlds(scm)
        if all(
            solve({k: bindings[v] for k, v in node.interventions.items()})[node.variable] == bindings[value]
            for node, value in event.items()
        )
    )


def _interventional_probability(scm, outcome, intervention):
    return sum(p for p, solve in _worlds(scm) if all(solve(intervention)[v] == val for v, val in outcome.items()))


def _evaluate(node, scm, env):
    if isinstance(node, ConstantNode):
        return node.value
    if isinstance(node, ProbabilityNode):
        return _interventional_probability(
            scm, {v: env[node.values[v]] for v in node.variables}, {v: env[node.values[v]] for v in node.do}
        )
    if isinstance(node, ProductNode):
        return prod(_evaluate(child, scm, env) for child in node.children)
    symbols = sorted(node.sumset)
    return sum(
        _evaluate(node.children[0], scm, {**env, **dict(zip(symbols, values))})
        for values in product(DOMAIN, repeat=len(symbols))
    )


def _xor_scm():
    """The bow arc model X = U, Y = X xor U, with U a fair coin."""
    noises = {("U", "X"): np.array([1.0]), ("U", "Y"): np.array([1.0]), ("U", "X", "Y"): np.array([0.5, 0.5])}
    noise_of = {"X": [("U", "X"), ("U", "X", "Y")], "Y": [("U", "Y"), ("U", "X", "Y")]}
    tables = {"X": {((), 0, u): u for u in DOMAIN}, "Y": {((x,), 0, u): x ^ u for x, u in product(DOMAIN, DOMAIN)}}
    return ["X", "Y"], noises, noise_of, {"X": [], "Y": ["X"]}, tables


def assert_sound(graph, event, expression, rng, bindings=BINDINGS, n_models=3):
    """The expression equals P(gamma) in several random SCMs inducing ``graph``."""
    for _ in range(n_models):
        scm = _random_scm(graph, rng)
        assert _evaluate(expression.root, scm, bindings) == pytest.approx(
            _counterfactual_probability(scm, event, bindings), abs=1e-12
        )


class TestCounterfactualVariable:
    def test_equality_ignores_the_order_of_interventions(self):
        first = cf("Y", {"X": "x", "Z": "z"})
        second = cf("Y", {"Z": "z", "X": "x"})
        assert first == second
        assert hash(first) == hash(second)
        assert first != cf("Y", {"X": "x'", "Z": "z"})
        assert cf("Y") == cf("Y", {})

    def test_repr(self):
        assert repr(cf("Y", {"Z": "z", "X": "x"})) == "Y_{X=x, Z=z}"
        assert repr(cf("Y")) == "Y"


class TestMakeCounterfactualGraph:
    """make-cg (Fig. 10), checked against the counterfactual graphs drawn in the paper."""

    def test_figure_9c(self):
        """Fig. 9 (c) and Fig. 11: D, D_x and X, X_d merge first, then W, W_d, then the triplet Z, Z_x, Z_d (because D
        is observed at d, the value Z_d is fixed to), then Y, Y_d. Restricting to An(gamma') leaves Fig. 9 (c), and the
        query becomes P(y_x, x', z, d) (p. 1960)."""
        cf_graph, event = IDStar()._make_counterfactual_graph(admg(FIGURE_9A), FIGURE_9_QUERY)

        assert set(cf_graph.nodes()) == {cf("X", {"X": "x"}), cf("W", {"X": "x"}), cf("Y", {"X": "x"}), *map(cf, "DZX")}
        assert edges(cf_graph) == {
            (cf("X", {"X": "x"}), cf("W", {"X": "x"}), "->"),
            (cf("W", {"X": "x"}), cf("Y", {"X": "x"}), "->"),
            (cf("D"), cf("Z"), "->"),
            (cf("Z"), cf("Y", {"X": "x"}), "->"),
            bidirected(cf("X"), cf("Y", {"X": "x"})),
        }
        assert event == {cf("Y", {"X": "x"}): "y", cf("X"): "x'", cf("Z"): "z", cf("D"): "d"}

    def test_figure_9d(self):
        """Fig. 9 (d): nothing merges for P(y_{x,z}, x'), as X is observed at x' and fixed at x."""
        world = {"X": "x", "Z": "z"}
        cf_graph, event = IDStar()._make_counterfactual_graph(admg(FIGURE_9A), {cf("Y", world): "y", cf("X"): "x'"})

        assert set(cf_graph.nodes()) == {cf("X", world), cf("Z", world), cf("W", world), cf("Y", world), cf("X")}
        assert edges(cf_graph) == {
            (cf("X", world), cf("W", world), "->"),
            (cf("W", world), cf("Y", world), "->"),
            (cf("Z", world), cf("Y", world), "->"),
            bidirected(cf("X"), cf("Y", world)),
        }
        assert event == {cf("Y", world): "y", cf("X"): "x'"}

    def test_copies_of_a_variable_share_their_unobserved_parent(self):
        """Fig. 8 (b): the w-graph. Y_x and Y_x' differ in their X parent, so they stay apart but remain confounded."""
        cf_graph, _ = IDStar()._make_counterfactual_graph(
            admg(W_GRAPH), {cf("Y", {"X": "x"}): "y", cf("Y", {"X": "x'"}): "y'"}
        )
        assert edges(cf_graph) == {
            (cf("X", {"X": "x"}), cf("Y", {"X": "x"}), "->"),
            (cf("X", {"X": "x'"}), cf("Y", {"X": "x'"}), "->"),
            bidirected(cf("Y", {"X": "x"}), cf("Y", {"X": "x'"})),
        }

    def test_merged_nodes_with_different_values_are_inconsistent(self):
        """X observed at x makes Y_x and Y the same variable, which gamma cannot assign both y and y'."""
        cf_graph, event = IDStar()._make_counterfactual_graph(
            admg(W_GRAPH), {cf("Y", {"X": "x"}): "y", cf("X"): "x", cf("Y"): "y'"}
        )
        assert event is None


class TestIDStarSteps:
    """Each step of ID* (Fig. 12) on a query that reaches it."""

    def test_step_1_empty_conjunction(self):
        assert IDStar().identify(admg(W_GRAPH), event={}).root == ConstantNode(1)

    def test_step_2_axiom_of_effectiveness(self):
        result = IDStar().identify(admg(W_GRAPH), event={cf("X", {"X": "x"}): "x'", cf("Y"): "y"})
        assert result.root == ConstantNode(0)

    def test_step_3_drops_tautologies(self):
        result = IDStar().identify(admg(W_GRAPH), event={cf("X", {"X": "x"}): "x", cf("Y", {"X": "x"}): "y"})
        assert result.root == ProbabilityNode({"Y"}, do={"X"}, values={"Y": "y", "X": "x"})

    def test_step_5_inconsistent_counterfactual_graph(self):
        result = IDStar().identify(admg(W_GRAPH), event={cf("Y", {"X": "x"}): "y", cf("X"): "x", cf("Y"): "y'"})
        assert result.root == ConstantNode(0)

    def test_step_6_figure_9_matches_the_papers_worked_example(self):
        r"""p. 1963 and Tikka (2023): P(y_x, x', z_d, d) = \sum_w P_{z,w}(y, x') P_d(z) P_x(w) P(d)."""
        result = IDStar().identify(admg(FIGURE_9A), event=FIGURE_9_QUERY)
        assert result.root == MarginalNode(
            ProductNode(
                [
                    ProbabilityNode({"D"}, values={"D": "d"}),
                    ProbabilityNode({"W"}, do={"X"}, values={"W": "w", "X": "x"}),
                    ProbabilityNode({"X", "Y"}, do={"W", "Z"}, values={"X": "x'", "Y": "y", "W": "w", "Z": "z"}),
                    ProbabilityNode({"Z"}, do={"D"}, values={"Z": "z", "D": "d"}),
                ]
            ),
            sumset={"w"},
        )

    def test_step_6_drops_redundant_subscripts(self):
        r"""p. 1963: P(y_{x,z}, x') splits into P(y_{x,z,w}, x'_w) P(w_{x,z}), which reduce to P_{z,w}(y, x') and
        P_x(w). Fixing W cuts the only path from x to Y, so x is not kept in the subscript of Y."""
        result = IDStar().identify(admg(FIGURE_9A), event={cf("Y", {"X": "x", "Z": "z"}): "y", cf("X"): "x'"})
        assert result.to_latex() == (r"\sum_{w} P(W = w \mid do(X = x)) P(X = x', Y = y \mid do(W = w, Z = z))")

    def test_step_8_fails_on_a_conflict_between_subscript_and_observation(self):
        """Tikka (2023): adding X -> Y to Fig. 9 (a) makes the factor P(y_{x,z,w}, x'_{z,w}) conflict, x vs x'."""
        algorithm = IDStar()
        assert algorithm.identify(admg(FIGURE_9A + [("X", "Y", "->")]), event=FIGURE_9_QUERY) is False

        world = {"X": "x", "Z": "z", "W": "w"}
        assert set(algorithm.counterfactual_graph_.nodes()) == {
            cf("X", world),
            cf("Z", world),
            cf("W", world),
            cf("Y", world),
            cf("X"),
        }

    def test_step_8_fails_on_a_conflict_between_two_subscripts(self):
        """Lemma 27: P(y_x, y'_x') is not identifiable. The failing counterfactual graph is the w-graph."""
        algorithm = IDStar()
        assert algorithm.identify(admg(W_GRAPH), event={cf("Y", {"X": "x"}): "y", cf("Y", {"X": "x'"}): "y'"}) is False
        assert algorithm.counterfactual_graph_.get_subgraph(
            {cf("Y", {"X": "x"}), cf("Y", {"X": "x'"})}
        ).get_district() == {frozenset({cf("Y", {"X": "x"}), cf("Y", {"X": "x'"})})}

    def test_step_8_fails_on_the_zig_zag_graph(self):
        """Lemma 28: P(y_x, w1, w2, z_x') is not identifiable. The failing counterfactual graph is Fig. 13 (b)."""
        algorithm = IDStar()
        event = {cf("Y", {"X": "x"}): "y", cf("W1"): "w1", cf("W2"): "w2", cf("Z", {"X": "x'"}): "z"}
        assert algorithm.identify(admg(ZIG_ZAG), event=event) is False
        assert edges(algorithm.counterfactual_graph_) == {
            (cf("X", {"X": "x"}), cf("Y", {"X": "x"}), "->"),
            (cf("X", {"X": "x'"}), cf("Z", {"X": "x'"}), "->"),
            bidirected(cf("Y", {"X": "x"}), cf("W1")),
            bidirected(cf("W1"), cf("W2")),
            bidirected(cf("W2"), cf("Z", {"X": "x'"})),
        }

    def test_step_8_a_summed_node_conflicts_with_a_subscript(self):
        """In the bow arc, the C-component of P(y_x, y) also holds X, which gamma' leaves free. Its value is summed
        over and so is not x. Reading Step 8 as comparing observed values alone would return P_x(y). That is wrong:
        with X = U and Y = X xor U, P(Y_1 = 1, Y = 1) = 0 while P_1(Y = 1) = 1/2."""
        event = {cf("Y", {"X": "x"}): "y", cf("Y"): "y"}
        assert IDStar().identify(admg(BOW_ARC), event=event) is False

        scm = _xor_scm()
        assert _counterfactual_probability(scm, event, BINDINGS) == 0
        assert _interventional_probability(scm, {"Y": 1}, {"X": 1}) == 0.5

    def test_step_9_does_not_intervene_on_a_variable_of_gamma(self):
        """P(y_{x,z}, x, z') reaches Step 9 with S = {Y_{x,z}, X} and x in sub(S). Lemma 42 assumes var(S) and sub(S)
        are disjoint. Here X = x holds in S, so Y_{x,z} = Y_z and the factor is P_z(x, y). Intervening on X as well
        would be wrong: in the bow arc with X = U and Y = X xor U, P(Y_1 = 1, X = 1) = 0 while P_1(X = 1, Y = 1) =
        1/2."""
        graph = admg([("X", "Y", "->"), ("Z", "Y", "->"), ("X", "Y", "<>")])
        event = {cf("Y", {"X": "x", "Z": "z"}): "y", cf("X"): "x", cf("Z"): "z'"}
        result = IDStar().identify(graph, event=event)

        assert result.root == ProductNode(
            [
                ProbabilityNode({"X", "Y"}, do={"Z"}, values={"X": "x", "Y": "y", "Z": "z"}),
                ProbabilityNode({"Z"}, values={"Z": "z'"}),
            ]
        )
        assert_sound(graph, event, result, np.random.default_rng(0))

        scm = _xor_scm()
        assert _counterfactual_probability(scm, {cf("Y", {"X": "x"}): "y", cf("X"): "x"}, BINDINGS) == 0
        assert _interventional_probability(scm, {"X": 1, "Y": 1}, {"X": 1}) == 0.5


class TestIDStarSoundness:
    """Every identified expression equals P(gamma) in random SCMs inducing the graph."""

    @pytest.mark.parametrize(
        ("edge_list", "event", "expected"),
        [
            (FIGURE_9A, FIGURE_9_QUERY, None),
            (
                BACK_DOOR,
                {cf("Y", {"X": "x"}): "y", cf("X"): "x'"},
                r"\sum_{z} P(X = x' \mid do(Z = z)) P(Y = y \mid do(X = x, Z = z)) P(Z = z)",
            ),
            (
                FRONT_DOOR,
                {cf("Y", {"X": "x"}): "y", cf("X"): "x'"},
                r"\sum_{m} P(M = m \mid do(X = x)) P(X = x', Y = y \mid do(M = m))",
            ),
            (BOW_ARC, {cf("Y", {"X": "x"}): "y", cf("X"): "x"}, "P(X = x, Y = y)"),
            (W_GRAPH, {cf("Y", {"X": "x"}): "y", cf("X"): "x'"}, r"P(X = x') P(Y = y \mid do(X = x))"),
        ],
        ids=["figure 9", "effect of treatment on the treated, back-door", "ETT, front-door", "consistency", "w-graph"],
    )
    def test_known_queries(self, edge_list, event, expected):
        graph = admg(edge_list)
        result = IDStar().identify(graph, event=event)
        if expected is not None:
            assert result.to_latex() == expected
        assert_sound(graph, event, result, np.random.default_rng(1))

    def test_random_graphs_and_queries(self):
        """Random ADMGs over up to four variables, with random conjunctions of up to three events whose subscripts
        may name the event's own variable (Steps 2 and 3)."""
        rng = np.random.default_rng(2008)
        identified = 0
        for _ in range(150):
            names = ["A", "B", "C", "D"][: int(rng.integers(2, 5))]
            edge_list = [
                (u, v, edge_type)
                for i, u in enumerate(names)
                for v in names[i + 1 :]
                for edge_type, probability in (("->", 0.5), ("<>", 0.3))
                if rng.random() < probability
            ]
            graph = admg(edge_list, names)
            event = {
                cf(variable, {v: f"{v.lower()}{rng.integers(2)}" for v in rng.choice(names, rng.integers(0, 3))}): (
                    f"{variable.lower()}{rng.integers(2)}"
                )
                for variable in rng.choice(names, rng.integers(1, 4))
            }
            result = IDStar().identify(graph, event=event)
            if result is not False:
                identified += 1
                bindings = {f"{v.lower()}{i}": i for v in names for i in DOMAIN}
                assert_sound(graph, event, result, rng, bindings=bindings, n_models=1)
        assert identified > 100


class TestIDStarNonIdentifiable:
    def test_two_models_agree_on_every_experiment_but_not_on_the_w_graph_query(self):
        """Lemma 27 for X -> Y: Y = X xor U and Y = U agree on P(x) and on every P_x(y), yet give P(y_x, y'_x') = 1/2
        and 0 for x = 1, x' = 0, y = 1, y' = 0. ID* fails, as it must."""
        event = {cf("Y", {"X": "x"}): "y", cf("Y", {"X": "x'"}): "y'"}
        assert IDStar().identify(admg(W_GRAPH), event=event) is False

        noises = {("U", "X"): np.array([0.5, 0.5]), ("U", "Y"): np.array([0.5, 0.5])}
        noise_of = {"X": [("U", "X")], "Y": [("U", "Y")]}
        models = [
            {"X": {((), u): u for u in DOMAIN}, "Y": {((x,), u): mechanism(x, u) for x, u in product(DOMAIN, DOMAIN)}}
            for mechanism in (lambda x, u: x ^ u, lambda x, u: u)
        ]
        scms = [(["X", "Y"], noises, noise_of, {"X": [], "Y": ["X"]}, tables) for tables in models]

        for intervention, outcome in product([{}, {"X": 0}, {"X": 1}], product(DOMAIN, DOMAIN)):
            first, second = (_interventional_probability(s, dict(zip("XY", outcome)), intervention) for s in scms)
            assert first == second
        assert [_counterfactual_probability(s, event, BINDINGS) for s in scms] == [0.5, 0]

    def test_failure_propagates_out_of_step_6(self):
        """Fig. 9 (a) with X -> Y fails inside one of four Step 6 factors; the whole query must fail."""
        assert IDStar().identify(admg(FIGURE_9A + [("X", "Y", "->")]), event=FIGURE_9_QUERY) is False


class TestIDStarInputHandling:
    def test_dag_with_latents_matches_the_admg(self):
        dag = DAG(
            ebunch=[("X", "W"), ("W", "Y"), ("D", "Z"), ("Z", "Y"), ("U", "X"), ("U", "Y")],
            latents={"U"},
        )
        assert IDStar().identify(dag, event=FIGURE_9_QUERY) == IDStar().identify(admg(FIGURE_9A), event=FIGURE_9_QUERY)

    def test_latent_in_event_is_rejected(self):
        dag = DAG(ebunch=[("U", "X"), ("U", "Y"), ("X", "Y")], latents={"U"})
        with pytest.raises(ValueError, match="cannot be both latent"):
            IDStar().identify(dag, event={cf("Y", {"U": "u"}): "y"})

    def test_unknown_variable_is_rejected(self):
        with pytest.raises(ValueError, match="not nodes of the causal graph"):
            IDStar().identify(admg(W_GRAPH), event={cf("Y", {"Q": "q"}): "y"})

    def test_unsupported_graph_type(self):
        with pytest.raises(ValueError, match="must be an instance of"):
            IDStar().identify("not a graph", event={cf("Y"): "y"})

    def test_counterfactual_graph_is_reset_between_runs(self):
        algorithm = IDStar()
        assert algorithm.identify(admg(W_GRAPH), event={cf("Y", {"X": "x"}): "y", cf("Y", {"X": "x'"}): "y'"}) is False
        assert algorithm.counterfactual_graph_ is not None
        assert isinstance(algorithm.identify(admg(W_GRAPH), event={cf("Y"): "y"}), ProbabilityExpressionTree)
        assert algorithm.counterfactual_graph_ is None
