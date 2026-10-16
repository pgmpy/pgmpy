from itertools import product

from pgmpy.base import ADMG, DAG
from pgmpy.identification import BaseFormulaIdentification
from pgmpy.identification.probability_expression import (
    ConstantNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
)


class CounterfactualVariable:
    r"""
    A counterfactual variable :math:`Y_{x}`: the variable ``Y`` in the submodel where ``X`` is fixed to ``x``.

    Parameters
    ----------
    variable : hashable
        The variable of the causal graph.

    interventions : dict, optional
        Maps each intervened variable to the value it is fixed to. Without interventions, the counterfactual variable
        is the observed variable itself.

    Examples
    --------
    >>> from pgmpy.identification import CounterfactualVariable
    >>> CounterfactualVariable("Y", {"X": "x"})
    Y_{X=x}
    >>> CounterfactualVariable("Y") == CounterfactualVariable("Y", {})
    True
    """

    def __init__(self, variable, interventions=None):
        self.variable = variable
        self.interventions = dict(interventions or {})

    def __repr__(self):
        if not self.interventions:
            return str(self.variable)
        subscript = ", ".join(f"{variable}={value}" for variable, value in sorted(self.interventions.items(), key=str))
        return f"{self.variable}_{{{subscript}}}"

    def __eq__(self, other):
        if not isinstance(other, CounterfactualVariable):
            return NotImplemented
        return (self.variable, self.interventions) == (other.variable, other.interventions)

    def __hash__(self):
        return hash((self.variable, frozenset(self.interventions.items())))


class IDStar(BaseFormulaIdentification):
    r"""
    Given a causal graph, identifies a counterfactual query using the ID* algorithm.

    This class implements the ID* algorithm :footcite:p:`shpitser_2008` for identifying counterfactual queries from
    interventional distributions. A counterfactual query is the probability of a set of events that can come from
    different hypothetical worlds. For example, the probability that Y would have been y had X been x, given that X
    was actually x'. The algorithm works by building a counterfactual graph that combines all the worlds in the query,
    and splitting the query into one part for each c-component of this graph. Each part is then either identified as
    an interventional distribution or shown to be non-identifiable.

    Following the worked example in the paper and :footcite:t:`tikka_2023`, the interventions on each variable are
    reduced to the ones that can still affect it in the counterfactual graph.

    Parameters
    ----------
    causal_graph : ADMG | DAG
        An ADMG, or a DAG whose unobserved variables have the ``latents`` role. No other roles are needed.

    event : dict
        The query, as a mapping from each ``CounterfactualVariable`` to its value. Different values of a variable,
        such as ``"x"`` and ``"x'"``, are treated as different.

    Returns
    -------
    expression : ProbabilityExpressionTree | False
        The symbolic formula for the probability of the query, or False if it is not identifiable from
        interventional distributions.

    Attributes
    ----------
    counterfactual_graph_ : ADMG | None
        When identification fails, the counterfactual graph of the failing call. Its only C-component holds the
        conflicting assignment that witnesses non-identifiability (Theorem 32). None otherwise.

    Examples
    --------
    The query :math:`P(y_{x}, x', z_{d}, d)` on Fig. 9 (a) of the paper (p. 1963):

    >>> from pgmpy.base import ADMG
    >>> from pgmpy.identification import CounterfactualVariable, IDStar
    >>> admg = ADMG(
    ...     edge_list=[("X", "W", "->"), ("W", "Y", "->"), ("D", "Z", "->"), ("Z", "Y", "->"), ("X", "Y", "<>")]
    ... )
    >>> event = {
    ...     CounterfactualVariable("Y", {"X": "x"}): "y",
    ...     CounterfactualVariable("X"): "x'",
    ...     CounterfactualVariable("Z", {"D": "d"}): "z",
    ...     CounterfactualVariable("D"): "d",
    ... }
    >>> IDStar().identify(admg, event=event).to_latex()
    "\\sum_{w} P(D = d) P(W = w \\mid do(X = x)) P(X = x', Y = y \\mid do(W = w, Z = z)) P(Z = z \\mid do(D = d))"

    The probability of necessity and sufficiency is not identifiable (Lemma 27):

    >>> event = {CounterfactualVariable("Y", {"X": "x"}): "y", CounterfactualVariable("Y", {"X": "x'"}): "y'"}
    >>> IDStar().identify(ADMG(edge_list=[("X", "Y", "->")]), event=event)
    False

    References
    ----------
    - :footcite:t:`shpitser_2008`
    - :footcite:t:`tikka_2023`
    """

    supported_graph_types = (ADMG, DAG)
    required_roles = ()

    def _identify(self, causal_graph, event):
        """Run the ID* algorithm.

        Parameters
        ----------
        causal_graph : ADMG | DAG
            The causal graph.

        event : dict
            The conjunction of counterfactual events.

        Returns
        -------
        expression : ProbabilityExpressionTree | False
            The identified formula, or False if the query is not identifiable.
        """
        mentioned = {node.variable for node in event} | {variable for node in event for variable in node.interventions}
        if missing := (mentioned - set(causal_graph.nodes())):
            raise ValueError(f"{sorted(missing, key=str)} in `event` are not nodes of the causal graph.")

        if isinstance(causal_graph, DAG):
            if latent_variables := (mentioned & set(causal_graph.latents)):
                raise ValueError(f"{sorted(latent_variables, key=str)} cannot be both latent and in `event`.")
            causal_graph = causal_graph.to_admg()

        self.counterfactual_graph_ = None
        result = self._identify_recursive(causal_graph, dict(event))
        if result is False:
            return False
        return ProbabilityExpressionTree(root=result)

    def _identify_recursive(self, causal_graph, event):
        r"""Recursive implementation of ID*, following Fig. 12 of the paper.

        Parameters
        ----------
        causal_graph : ADMG
            The causal graph. Unlike ID, ID* never recurses on a subgraph; only the query changes.

        event : dict
            The query, mapping each ``CounterfactualVariable`` to its value.

        Returns
        -------
        expression : _TreeNode | False
            The expression for the probability of the query, or False if it is not identifiable.
        """
        # Step 1: If the query is empty, its probability is 1.
        if not event:
            return ConstantNode(1)

        # Step 2: If a variable is intervened on and observed with a different value, the probability is 0.
        if any(node.interventions.get(node.variable, value) != value for node, value in event.items()):
            return ConstantNode(0)

        # Step 3: If a variable is intervened on and observed with the same value, remove it from the query.
        for node in event:
            if node.variable in node.interventions:
                return self._identify_recursive(causal_graph, {other: event[other] for other in event if other != node})

        # Step 4: Build the counterfactual graph and update the query to use its nodes.
        cf_graph, event = self._make_counterfactual_graph(causal_graph, event)

        # Step 5: If the query assigns different values to the same variable, the probability is 0.
        if event is None:
            return ConstantNode(0)

        fixed = {node for node in cf_graph.nodes() if node.variable in node.interventions}
        observable = set(cf_graph.nodes()) - fixed
        districts = cf_graph.get_subgraph(observable).get_district()

        # Step 6: If there is more than one c-component, identify each one separately and multiply them.
        if len(districts) > 1:
            # Step 6.1: Give a value to each variable that is not in the query, to sum over.
            values = dict(event)
            used = set(event.values()) | {value for node in event for value in node.interventions.values()}
            sumset = []
            for node in sorted(observable - set(event), key=str):
                symbol = str(node.variable).lower()
                while symbol in used:
                    symbol += "'"
                used.add(symbol)
                values[node] = symbol
                sumset.append(symbol)

            # Step 6.2: For each c-component, intervene on all the variables outside it. Keep only the interventions
            # that can still affect each variable, and identify the resulting query.
            factors = []
            for district in sorted(districts, key=lambda component: sorted(map(str, component))):
                outside = observable - district
                cut_graph = cf_graph.do(outside)
                term = {}
                for node in district:
                    interventions = {
                        ancestor.variable: ancestor.interventions[ancestor.variable]
                        if ancestor in fixed
                        else values[ancestor]
                        for ancestor in cut_graph.get_ancestors(node)
                        if ancestor in fixed or ancestor in outside
                    }
                    term[CounterfactualVariable(node.variable, interventions)] = values[node]

                factor = self._identify_recursive(causal_graph, term)
                if factor is False:
                    return False
                factors.append(factor)

            return ProductNode(factors).marginalize(sumset)

        # Step 7: There is only one c-component. Collect the values of all the intervened variables.
        subscripts = {}
        for node in fixed:
            subscripts.setdefault(node.variable, set()).add(node.interventions[node.variable])

        # Step 8: If an intervened variable has a different value anywhere else, the query is not identifiable. A
        # variable that is not in the query can take any value, so it also counts as a different value.
        for variable, subscript_values in subscripts.items():
            nodes = [node for node in observable if node.variable == variable]
            if any(node not in event for node in nodes) or len(subscript_values | {event[node] for node in nodes}) > 1:
                self.counterfactual_graph_ = cf_graph
                return False

        # Step 9: Return the distribution of the query variables under all the interventions. A variable that is in
        # the query and also intervened on stays in the query only.
        values = {node.variable: value for node, value in event.items()}
        interventions = {
            variable: next(iter(subscript_values))
            for variable, subscript_values in subscripts.items()
            if variable not in values
        }
        return ProbabilityNode(values.keys(), do=interventions.keys(), values=values | interventions)

    def _make_counterfactual_graph(self, causal_graph, event):
        r"""Build the counterfactual graph of a conjunction, following make-cg in Fig. 10 of the paper.

        Nodes are ``CounterfactualVariable`` instances named after the world they were created in. A node whose
        own variable is in its interventions is fixed by that world's action.

        Parameters
        ----------
        causal_graph : ADMG
            The causal graph.

        event : dict
            The query, mapping each ``CounterfactualVariable`` to its value. No variable is intervened on itself.

        Returns
        -------
        cf_graph : ADMG
            The counterfactual graph.

        event : dict | None
            The query updated to use the merged nodes, or None if it gives two merged nodes different values.
        """
        # Step 1: Build the parallel worlds graph, with one copy of the causal graph for each intervention.
        worlds = sorted(
            {frozenset(node.interventions.items()) for node in event},
            key=lambda world: (len(world), sorted(map(str, world))),
        )
        cf_graph = ADMG()
        for world, variable in product(worlds, causal_graph.nodes()):
            node = CounterfactualVariable(variable, dict(world))
            cf_graph.add_node(node)
            if variable not in node.interventions:
                for parent in causal_graph.get_parents(variable):
                    cf_graph.add_edge(CounterfactualVariable(parent, dict(world)), node, "->")

        # Step 1.1: Copies of a variable share its unobserved noise, and copies of two confounded variables share
        # their confounder. Intervened variables have no unobserved parents.
        confounded = [(variable, variable) for variable in causal_graph.nodes()]
        confounded += [(u, v) for u, v, _ in causal_graph.get_edges(edge_types=["<>"])]
        for (u, v), (world_u, world_v) in product(confounded, product(worlds, repeat=2)):
            node_u = CounterfactualVariable(u, dict(world_u))
            node_v = CounterfactualVariable(v, dict(world_v))
            if (
                node_u != node_v
                and u not in node_u.interventions
                and v not in node_v.interventions
                and not cf_graph.has_edge(node_u, node_v, "<>")
            ):
                cf_graph.add_edge(node_u, node_v, "<>")

        # Step 2: Copy the query so that it can be updated as nodes are merged.
        event = dict(event)

        # Step 3: In topological order, merge the copies of a variable that are the same random variable. The copy
        # with the fewest interventions is kept.
        for variable in causal_graph.get_topological_order():
            survivors = []
            for world in worlds:
                beta = CounterfactualVariable(variable, dict(world))
                if variable in beta.interventions:
                    continue

                for alpha in survivors:
                    # Step 3.1: Two copies are the same if each pair of their parents is the same node or has the
                    # same known value (Lemma 24). A parent with no known value is only equal to itself.
                    alpha_parents = {parent.variable: parent for parent in cf_graph.get_parents(alpha)}
                    beta_parents = {parent.variable: parent for parent in cf_graph.get_parents(beta)}
                    if all(
                        len(
                            {
                                node.interventions[node.variable]
                                if node.variable in node.interventions
                                else event.get(node, node)
                                for node in (alpha_parents[parent], beta_parents[parent])
                            }
                        )
                        == 1
                        for parent in alpha_parents
                    ):
                        break
                else:
                    survivors.append(beta)
                    continue

                # Step 3.2: If the query gives the two copies different values, the query is inconsistent.
                if alpha in event and beta in event and event[alpha] != event[beta]:
                    return cf_graph, None

                # Step 3.3: The merged node keeps the parents of the kept copy and gets the children and
                # confounders of both (Lemma 25).
                for child in cf_graph.get_children(beta):
                    cf_graph.add_edge(alpha, child, "->")
                for spouse in cf_graph.get_spouses(beta) - {alpha}:
                    if not cf_graph.has_edge(alpha, spouse, "<>"):
                        cf_graph.add_edge(alpha, spouse, "<>")
                cf_graph.remove_node(beta)

                # Step 3.4: Update the query to use the merged node.
                if beta in event:
                    event[alpha] = event.pop(beta)

        # Step 4: Keep only the ancestors of the query variables.
        return cf_graph.get_ancestral_graph(set(event)), event
