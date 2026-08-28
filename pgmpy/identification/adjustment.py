import itertools

import networkx as nx

from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.identification import BaseIdentification
from pgmpy.utils.sets import _powerset


class Adjustment(BaseIdentification):
    """
    Given a causal graph, finds the adjustment set.

    This class implements a few variants for computing adjustment sets for
    identifying the total causal effect of the variables in the `exposures`
    role on the variables in the `outcomes` role. Additionally, it provides methods to check if the
    current set of variables with role `adjustment` satisfy the backdoor
    criterion and to compute the backdoor adjustment formula.

    Parameters
    ----------
    variant: str
        The variant of backdoor identification to use. Default is 'minimal'.

        - 'all': Returns all adjustment sets that satisfy the backdoor criterion.
        - 'minimal': Returns the smallest adjustment set.
        - 'minimal_variance': Returns the adjustment set for which estimators achieve minimal variance.

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> dag = DAG(
    ...     ebunch=[
    ...         ("x1", "y1"),
    ...         ("x1", "z1"),
    ...         ("z1", "z2"),
    ...         ("z2", "x2"),
    ...         ("y2", "z2"),
    ...     ],
    ...     roles={"exposures": "x1", "outcomes": "y1"},
    ... )
    >>> dag_with_adj, success = Adjustment(variant="minimal").identify(dag)
    >>> roles = dag_with_adj.get_role_dict()
    >>> roles["exposures"]
    ['x1']
    >>> roles["outcomes"]
    ['y1']
    >>> Adjustment(variant="minimal").validate(dag_with_adj)
    True

    References
    ----------
    - :footcite:t:`perkovic_2018`
    - :footcite:t:`witte_2022`
    """

    def __init__(self, variant="minimal"):
        self.variant = variant
        if self.variant in ("minimal", "all"):
            self.supported_graph_types = (DAG, PDAG, ADMG, MAG)
        elif self.variant == "minimal_variance":
            self.supported_graph_types = (DAG, PDAG)

    @staticmethod
    def _proper_causal_paths(causal_graph):
        """
        Yield the node paths of all proper causal paths from `exposures` to `outcomes`.

        A proper causal path is a directed path from an exposure to an outcome
        that does not pass through another exposure. For graph types that also
        carry non-directed edges (ADMG, MAG), the paths are computed on the
        directed projection of the graph.
        """
        exposures = causal_graph.get_role("exposures")
        outcomes = causal_graph.get_role("outcomes")
        directed = causal_graph if isinstance(causal_graph, DAG) else causal_graph.get_directed_graph()
        for source in exposures:
            for path in nx.all_simple_paths(directed, source, outcomes):
                if set(path[1:]).intersection(exposures):
                    continue
                yield path

    def _get_proper_backdoor_graph(self, causal_graph, inplace=False):
        """
        Returns a proper backdoor graph of the `causal_graph`.

        For a `causal_graph` with variable roles `exposures` and `outcomes`
        defined, returns it's proper backdoor graph. A proper backdoor graph is
        a graph which removes the first edge of every proper causal path from
        `exposures` to `outcomes`.

        Parameters
        ----------
        causal_graph: pgmpy.base.DAG, pgmpy.base.PDAG, pgmpy.base.ADMG, or pgmpy.base.MAG
            The causal graph for which the proper backdoor graph is to be computed.

        inplace: boolean
            If inplace is True, modifies the object itself. Otherwise returns
            a modified copy of self.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> from pgmpy.identification import Adjustment
        >>> dag = DAG(
        ...     ebunch=[
        ...         ("x1", "y1"),
        ...         ("x1", "z1"),
        ...         ("z1", "z2"),
        ...         ("z2", "x2"),
        ...         ("y2", "z2"),
        ...     ],
        ...     roles={"exposures": "x1", "outcomes": "y1"},
        ... )
        >>> dag_proper = Adjustment()._get_proper_backdoor_graph(dag, inplace=False)
        >>> list(dag_proper.edges())
        [('x1', 'z1'), ('z1', 'z2'), ('z2', 'x2'), ('y2', 'z2')]

        References
        ----------
        - :footcite:t:`perkovic_2018`
        """
        model = causal_graph if inplace else causal_graph.copy()
        first_edges = {(path[0], path[1]) for path in self._proper_causal_paths(causal_graph)}
        if isinstance(causal_graph, DAG):
            model.remove_edges_from(first_edges)
        else:
            model.remove_edges_from([(u, v, "->") for u, v in first_edges])
        return model

    def _identify(self, causal_graph):
        """
        Identify adjustment sets using the backdoor criterion.

        Parameters
        ----------
        causal_graph: DAG | PDAG | ADMG | MAG | PAG
            The causal graph for which the adjustment sets are to be identified.

        Returns
        -------
        causal_graph: DAG | PDAG | ADMG | MAG | PAG
            The causal graph with the identified adjustment set added as role `adjustment`.

        success: bool
            True if the identification was successful, False otherwise.
        """
        # Step 1: If variant = "minimal", use the algorithm from [1]. Get the
        #         proper backdoor graph and compute the adjustment set.
        if self.variant == "minimal":
            if len(causal_graph.get_role("exposures")) != 1:
                raise NotImplementedError("Backdoor identification is only implemented for single exposure variable.")
            if len(causal_graph.get_role("outcomes")) != 1:
                raise NotImplementedError("Backdoor identification is only implemented for single outcome variable.")

            exposure = causal_graph.get_role("exposures")[0]
            outcome = causal_graph.get_role("outcomes")[0]

            backdoor_graph = self._get_proper_backdoor_graph(causal_graph, inplace=False)
            adjustment_set = backdoor_graph.minimal_dseparator(exposure, outcome)

            if adjustment_set is None:
                return causal_graph, False
            else:
                return (
                    causal_graph.with_role("adjustment", adjustment_set, inplace=False),
                    True,
                )

        # Step 2: If variant = "minimal_variance", use the algorithm from [2].
        #         O(X, Y, G) = pa(cn(X, Y, G), G) \ forb(X, Y, G)
        elif self.variant == "minimal_variance":
            raise NotImplementedError("Backdoor identification with minimal variance is not implemented yet.")

        # Step 3: If variant = "all", iterate over all possible sets of adjustment
        #         variables, and return all that are valid.
        elif self.variant == "all":
            exposure = causal_graph.get_role("exposures")[0]
            outcome = causal_graph.get_role("outcomes")[0]

            ancestors = causal_graph.get_ancestors([exposure, outcome])
            # Remove any variables on the path from exposure to outcome (these cannot be in the adjustment set)
            ancestors -= set(itertools.chain(*nx.all_simple_paths(causal_graph, exposure, outcome)))
            ancestors -= {exposure, outcome}
            ancestors -= set(causal_graph.latents)

            valid_adj_graphs = []
            for s in _powerset(ancestors):
                adj_causal_graph = causal_graph.with_role("adjustment", s, inplace=False)
                if self.validate(causal_graph=adj_causal_graph):
                    valid_adj_graphs.append(adj_causal_graph)

            return valid_adj_graphs, len(valid_adj_graphs) > 0

    def _validate(self, causal_graph):
        """
        Validate the causal graph for backdoor identification.

        Given a `causal_graph` with variable roles `exposures`, `outcomes`, and
        `adjustment` defined, this method checks if the given `adjustment` set
        satisfies the adjustment criterion [1]: it must not contain a forbidden
        node (an exposure, a node on a proper causal path from the exposures to
        the outcomes, or a descendant of such a node), and it must block every
        proper non-causal path, i.e. the exposures must be separated from the
        outcomes given the adjustment set in the proper backdoor graph.

        Parameters
        ----------
        causal_graph: DAG | PDAG | ADMG | MAG | PAG
            The causal graph to validate.

        Returns
        -------
        bool:
            True if the `adjustment` set is valid, False otherwise.

        References
        ----------
        - :footcite:t:`perkovic_2018`
        """
        exposures = causal_graph.get_role("exposures")
        outcomes = causal_graph.get_role("outcomes")
        adjustment_vars = set(causal_graph.get_role("adjustment"))

        is_dag = isinstance(causal_graph, DAG)
        directed = causal_graph if is_dag else causal_graph.get_directed_graph()

        # Condition 1: No adjustment variable may be forbidden - an exposure,
        # a node on a proper causal path, or a descendant of such a node.
        causal_path_nodes = set()
        for path in self._proper_causal_paths(causal_graph):
            causal_path_nodes.update(path[1:])

        forbidden = set(exposures).union(causal_path_nodes)
        for node in causal_path_nodes:
            forbidden.update(nx.descendants(directed, node))

        if adjustment_vars.intersection(forbidden):
            return False

        # Condition 2: The adjustment set must block every proper non-causal
        # path from the exposures to the outcomes.
        backdoor_graph = self._get_proper_backdoor_graph(causal_graph, inplace=False)

        # DAG has not migrated onto _CoreGraph yet and exposes d-separation as `is_dconnected`;
        # this branch collapses into the `is_mseparated` call once it does.
        if is_dag:
            return all(
                not backdoor_graph.is_dconnected(exposure_var, outcome_var, observed=list(adjustment_vars) or None)
                for exposure_var in exposures
                for outcome_var in outcomes
            )

        return all(
            backdoor_graph.is_mseparated(exposure_var, outcome_var, conditioning_set=adjustment_vars)
            for exposure_var in exposures
            for outcome_var in outcomes
        )
