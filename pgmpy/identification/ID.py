from itertools import combinations

from pgmpy.base import ADMG, DAG
from pgmpy.identification import BaseFormulaIdentification
from pgmpy.identification.probability_expression import (
    DivisionNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
)


class ID(BaseFormulaIdentification):
    r"""
    Given a causal graph, identifies the causal effect using the ID algorithm.

    The class implements the recursive identification procedure of :footcite:t:`shpitser_2006` (Fig. 3) for ADMGs and
    DAGs with exposure and outcome roles.

    Parameters
    ----------
    causal_graph: ADMG | DAG
        An ADMG or DAG with exposures and outcomes roles assigned. The two roles must be disjoint.

    Returns
    -------
    expression: ProbabilityExpressionTree | False
        The symbolic formula for P(Y|do(X)), or False if the effect is not identifiable. When False, the witness
        hedge is stored in ``self.hedge_`` as the pair of graphs ``(G, G_S)`` thrown by Step 5; the C-forests forming
        the hedge are edge subgraphs of that pair (Theorem 6).

    Examples
    --------
    >>> from pgmpy.base import ADMG
    >>> from pgmpy.identification import ID
    >>> admg = ADMG(
    ...     edge_list=[("X", "M", "->"), ("M", "Y", "->"), ("X", "Y", "<>")],
    ...     exposures={"X"},
    ...     outcomes={"Y"},
    ... )
    >>> id_algo = ID()
    >>> id_algo.identify(admg).to_latex()
    '\\sum_{M} P(M \\mid X) \\left[ \\sum_{X} P(X) P(Y \\mid M, X) \\right]'

    >>> from pgmpy.base import DAG
    >>> dag = DAG(
    ...     ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
    ...     exposures={"X"},
    ...     outcomes={"Y"},
    ... )
    >>> ID().identify(dag).to_latex()
    '\\sum_{Z} P(Z) P(Y \\mid X, Z)'
    """

    supported_graph_types = (ADMG, DAG)

    def _identify(self, causal_graph):
        """Run the ID algorithm.

        Parameters
        ----------
        causal_graph: ADMG | DAG
            The causal graph with exposures and outcomes roles assigned.

        Returns
        -------
        expression: ProbabilityExpressionTree | False
            The identified formula, or False if the effect is not identifiable.
        """
        # Steps 1-7 are stated for semi-Markovian models, so a DAG is first replaced by its latent projection: the
        # latents must become bidirected edges before the districts of Steps 4-7 can see the confounding.
        if isinstance(causal_graph, DAG):
            # A latent has no observational or interventional quantity of its own, so it cannot carry a role the
            # estimand is defined over. Checked before projecting, while the latents are still nodes.
            for role in ("exposures", "outcomes"):
                if latent_roles := (set(causal_graph.latents) & set(causal_graph.get_role(role))):
                    raise ValueError(f"{sorted(latent_roles)} cannot be both latent and in '{role}'.")
            causal_graph = causal_graph.to_admg()

        exposures = frozenset(causal_graph.get_role("exposures"))
        outcomes = frozenset(causal_graph.get_role("outcomes"))
        variables = frozenset(causal_graph.nodes())

        # Definition 2 of the paper defines P_x(Y) only when X and Y are disjoint.
        if overlap := (exposures & outcomes):
            raise ValueError(f"'exposures' and 'outcomes' must be disjoint, but {sorted(overlap)} is in both.")

        # A single topological order over the original graph, valid for every induced subgraph the recursion builds.
        ordering = causal_graph.get_topological_order()

        # The estimand P. Starts as the observational joint P(v) and is rewritten by Steps 2 and 7.
        estimand = ProbabilityNode(variables)

        result = self._identify_recursive(outcomes, exposures, variables, causal_graph, estimand, ordering)
        if result is False:
            return False
        return ProbabilityExpressionTree(root=result)

    def _district_product(self, estimand, district, ordered):
        r"""Return the chain rule product :math:`\prod_{V_i \in district} P(v_i \mid v_\pi^{(i-1)})` of Steps 6 and 7.

        ``P`` is the estimand carried into the current call, not the observational joint, because Step 7 replaces it
        by ``Q[S']`` before recursing. Each conditional is therefore taken as a ratio of two marginals of ``P``,
        :math:`\sum_{v_{i+1}, \ldots, v_n} P \,/ \sum_{v_i, \ldots, v_n} P`, which is where the division in the napkin
        graph comes from. Both marginals collapse to plain joints while the estimand is still one, and the ratio is
        then written back as an atomic conditional.

        Parameters
        ----------
        estimand: _TreeNode
            The ``P`` argument of the current call.

        district: frozenset
            The C-component to factorise: ``S`` on Step 6, ``S'`` on Step 7.

        ordered: list
            The current variable set ``V`` in topological order.

        Returns
        -------
        product: _TreeNode
            The product of the per-variable conditionals, or that lone factor when the district is a singleton.
        """
        factors = []
        for i, v_i in enumerate(ordered):
            if v_i not in district:
                continue

            numerator = estimand.marginalize(ordered[i + 1 :])
            if i == 0:
                # No preceding variables, so the denominator would sum out everything and give 1.
                factors.append(numerator)
                continue

            denominator = estimand.marginalize(ordered[i:])
            if isinstance(numerator, ProbabilityNode) and isinstance(denominator, ProbabilityNode):
                # Both collapsed, so P(A) / P(A \ {V_i}) is the atomic P(v_i | a \ {v_i}).
                factors.append(ProbabilityNode(frozenset({v_i}), cond=denominator.variables))
            else:
                factors.append(DivisionNode(numerator, denominator))

        # A district may be a singleton, and ProductNode needs two factors.
        return factors[0] if len(factors) == 1 else ProductNode(factors)

    def _identify_recursive(self, outcomes, exposures, variables, causal_graph, estimand, ordering):
        """Recursive implementation of the ID algorithm, following Fig. 3 of the paper.

        Parameters
        ----------
        outcomes: frozenset
            ``y``, the outcome variables.

        exposures: frozenset
            ``x``, the variables being intervened on.

        variables: frozenset
            ``V``, the variables of the current subproblem; always the node set of ``causal_graph``.

        causal_graph: ADMG
            ``G``, the current graph; an induced subgraph of the original.

        estimand: _TreeNode
            ``P``, the carried expression. Begins as ``P(v)``, rewritten by Steps 2 and 7.

        ordering: list
            The topological order fixed over the original graph.

        Returns
        -------
        expression: _TreeNode | False
            The expression for ``P_x(y)``, or False if a hedge was found.
        """
        # Step 1: X is empty. The effect is a marginal of the carried estimand.
        if not exposures:
            return estimand.marginalize(variables - outcomes)

        # Step 2: V is not An(Y) in G. Restrict the whole problem to An(Y).
        ancestors = frozenset(causal_graph.get_ancestors(outcomes))
        if variables != ancestors:
            return self._identify_recursive(
                outcomes,
                exposures & ancestors,
                ancestors,
                causal_graph.get_ancestral_graph(ancestors),
                estimand.marginalize(variables - ancestors),
                ordering,
            )

        # Step 3: W = (V \ X) \ An(Y) in G with the incoming edges of X removed. Intervening on W is free (Lemma 6).
        w = (variables - exposures) - causal_graph.do(exposures).get_ancestors(outcomes)
        if w:
            return self._identify_recursive(outcomes, exposures | w, variables, causal_graph, estimand, ordering)

        # C(G \ X), sorted by the topological order so the emitted product is deterministic.
        position = {node: index for index, node in enumerate(ordering)}
        c_components = sorted(
            causal_graph.get_subgraph(variables - exposures).get_district(),
            key=lambda component: min(position[node] for node in component),
        )

        # Step 4: C(G \ X) = {S1, ..., Sk} with k > 1. Decompose into k independent subproblems.
        if len(c_components) > 1:
            factors = []
            for component in c_components:
                factor = self._identify_recursive(
                    component, variables - component, variables, causal_graph, estimand, ordering
                )
                if factor is False:
                    # FAIL propagates through recursive calls like an exception.
                    return False
                factors.append(factor)
            # k > 1 here, so there are always at least two factors.
            return ProductNode(factors).marginalize(variables - (outcomes | exposures))

        # C(G \ X) = {S}. S is bidirected connected in G \ X, and G \ X keeps every bidirected edge of G between its
        # nodes, so S is bidirected connected in G too and therefore lies inside exactly one C-component S' of G.
        S = c_components[0]
        S_prime = frozenset(causal_graph.get_district(next(iter(S))))

        # Step 5: C(G) = {G}, i.e. the whole of G is that one C-component. Store the hedge and fail.
        if S_prime == variables:
            self.hedge_ = (causal_graph, causal_graph.get_subgraph(S))
            return False

        ordered = [node for node in ordering if node in variables]

        # Step 6: S is itself a C-component of G. Return the sum over S \ Y of the chain rule product over S.
        if S_prime == S:
            return self._district_product(estimand, S, ordered).marginalize(S - outcomes)

        # Step 7: S is a strict subset of the C-component S'. Recurse into the subgraph over S' with the estimand
        # replaced by Q[S'].
        return self._identify_recursive(
            outcomes,
            exposures & S_prime,
            S_prime,
            causal_graph.get_subgraph(S_prime),
            self._district_product(estimand, S_prime, ordered),
            ordering,
        )


class IDC(BaseFormulaIdentification):
    """
    Given a causal graph, identifies the conditional causal effect using the IDC algorithm.

    The class implements the identification procedure of :footcite:t:`shpitser_2006a` (Fig. 3) for P(Y|do(X), Z),
    where Z is the set of variables with the ``conditioning`` role. Step 1 rewrites as much of Z as possible as
    interventions using rule 2 of do-calculus, and Step 2 identifies the remaining joint effect with ``ID`` and
    divides.

    Parameters
    ----------
    causal_graph: ADMG | DAG
        An ADMG or DAG with exposures, outcomes and conditioning roles assigned. The three roles must be pairwise
        disjoint. Use ``ID`` when there is nothing to condition on.

    Returns
    -------
    expression: ProbabilityExpressionTree | False
        The symbolic formula for P(Y|do(X), Z), or False if the effect is not identifiable. When False, the hedge
        thrown by the ``ID`` subcall is stored in ``self.hedge_``.

    Examples
    --------
    >>> from pgmpy.base import ADMG
    >>> from pgmpy.identification import IDC
    >>> admg = ADMG(
    ...     edge_list=[("X", "Z", "->"), ("Z", "Y", "->"), ("X", "Z", "<>")],
    ...     exposures={"X"},
    ...     outcomes={"Y"},
    ...     roles={"conditioning": {"Z"}},
    ... )
    >>> IDC().identify(admg).to_latex()
    'P(Y \\\\mid X, Z)'
    """

    supported_graph_types = (ADMG, DAG)
    required_roles = ("exposures", "outcomes", "conditioning")

    def _identify(self, causal_graph):
        """Run the IDC algorithm.

        Parameters
        ----------
        causal_graph: ADMG | DAG
            The causal graph with exposures, outcomes and conditioning roles assigned.

        Returns
        -------
        expression: ProbabilityExpressionTree | False
            The identified formula, or False if the effect is not identifiable.
        """
        # As for ``ID``, a DAG is first replaced by its latent projection. ``IDC`` checks ``conditioning`` alongside
        # the two roles ``ID`` validates, since P_x(y|z) is no more defined over an unobserved Z than over an
        # unobserved X or Y.
        if isinstance(causal_graph, DAG):
            for role in self.required_roles:
                if latent_roles := (set(causal_graph.latents) & set(causal_graph.get_role(role))):
                    raise ValueError(f"{sorted(latent_roles)} cannot be both latent and in '{role}'.")
            causal_graph = causal_graph.to_admg()

        roles = {
            "exposures": frozenset(causal_graph.get_role("exposures")),
            "outcomes": frozenset(causal_graph.get_role("outcomes")),
            "conditioning": frozenset(causal_graph.get_role("conditioning")),
        }

        # Definition 2 of the paper defines P_x(y|z) only when X, Y and Z are disjoint.
        for (first, first_variables), (second, second_variables) in combinations(roles.items(), 2):
            if overlap := (first_variables & second_variables):
                raise ValueError(f"'{first}' and '{second}' must be disjoint, but {sorted(overlap)} is in both.")

        exposures, outcomes, conditioning = roles["exposures"], roles["outcomes"], roles["conditioning"]
        result = self._identify_conditional(outcomes, exposures, conditioning, causal_graph)
        if result is False:
            return False
        return ProbabilityExpressionTree(root=result)

    def _is_rule_2_applicable(self, node, outcomes, exposures, conditioning, causal_graph):
        """Check whether rule 2 of do-calculus rewrites the conditioning on ``node`` as an intervention.

        The rule applies to a single Z in the conditioning set when (Y indep. Z | X, Z \\ {Z}) holds in G with the
        incoming edges of X and the outgoing edges of Z removed, that is, when Z has no back-door path to Y in the
        context of the effect.

        Parameters
        ----------
        node: hashable
            The conditioning variable Z being tested.

        outcomes: frozenset
            ``y``, the outcome variables.

        exposures: frozenset
            ``x``, the variables being intervened on.

        conditioning: frozenset
            ``z``, the remaining conditioning set, including ``node``.

        causal_graph: ADMG
            ``G``, the causal graph.

        Returns
        -------
        applicable: bool
            True if rule 2 applies to ``node``.
        """
        # G with the incoming edges of X removed ...
        graph = causal_graph.do(exposures)
        # ... and the outgoing edges of `node` removed. Only directed edges leave a node, so a bidirected edge at
        # `node` survives, which is what keeps a confounded conditioning variable behind the bar.
        for child in graph.get_children(node):
            graph.remove_edge(node, child, "->")

        connected = graph.get_mconnected_nodes(node, conditioning_set=exposures | (conditioning - {node}))
        return outcomes.isdisjoint(connected)

    def _identify_conditional(self, outcomes, exposures, conditioning, causal_graph):
        """Recursive implementation of the IDC algorithm, following Fig. 3 of the paper.

        Parameters
        ----------
        outcomes: frozenset
            ``y``, the outcome variables.

        exposures: frozenset
            ``x``, the variables being intervened on.

        conditioning: frozenset
            ``z``, the context the effect is conditioned on.

        causal_graph: ADMG
            ``G``, the causal graph. Unlike ID, IDC never recurses into a subgraph.

        Returns
        -------
        expression: _TreeNode | False
            The expression for ``P_x(y|z)``, or False if the ID subcall found a hedge.
        """
        # Step 1: rule 2 reads some Z in z as an intervention. Move it and start over. The topological order only
        # fixes which variable moves first when several qualify; by Corollary 1 the set left at Step 2 is the same
        # either way.
        for node in causal_graph.get_topological_order():
            if node in conditioning and self._is_rule_2_applicable(
                node, outcomes, exposures, conditioning, causal_graph
            ):
                return self._identify_conditional(outcomes, exposures | {node}, conditioning - {node}, causal_graph)

        # Step 2: nothing else can be moved. Identify the joint effect P_x(y, z) with ID and condition by dividing.
        # By Theorem 6 the conditional effect is identifiable exactly when that joint one is, so a hedge thrown by ID
        # is the witness for this effect too. Step 1 has moved part of z into x, so the roles are restated on a copy
        # of the graph rather than read off the original.
        joint_graph = causal_graph.with_role("exposures", exposures).with_role("outcomes", outcomes | conditioning)

        id_algorithm = ID()
        result = id_algorithm.identify(joint_graph)
        if result is False:
            self.hedge_ = id_algorithm.hedge_
            return False
        expression = result.root

        # P_x(y) already sums to one over y, so the division only bites when a conditioning set survives Step 1.
        if not conditioning:
            return expression
        return DivisionNode(expression, expression.marginalize(outcomes))
