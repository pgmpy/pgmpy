import itertools

import networkx as nx

from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.identification import BaseIdentification
from pgmpy.utils.sets import _powerset


class Adjustment(BaseIdentification):
    """
    Given a causal graph, finds the adjustment set.

    This class implements a few variants for computing adjustment sets for
    identifying the total causal effect of the `exposure` variables on
    `outcome` variables. Additionally, it provides methods to check if the
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
    ...     roles={"exposure": "x1", "outcome": "y1"},
    ... )
    >>> dag_with_adj = Adjustment(variant="minimal").identify(dag)
    >>> dag_with_adj.roles
    {'exposure': 'x1', 'outcome': 'y1', 'adjustment': ['z1', 'z2']}
    >>> Adjustment.validate(dag)

    References
    ----------
    [1] Perkovi, Emilija, et al. "Complete graphical characterization and
        construction of adjustment sets in Markov equivalence classes of ancestral
        graphs." Journal of Machine Learning Research.
    [2] Witte, Janine, et al. "On efficient adjustment in causal graphs."
        Journal of Machine Learning Research.
    """

    def __init__(self, variant="minimal"):
        self.variant = variant
        if self.variant in ("minimal", "all"):
            self.supported_graph_types = (DAG, PDAG, ADMG, MAG)
        elif self.variant == "minimal_variance":
            self.supported_graph_types = (DAG, PDAG)

    def _get_proper_backdoor_graph(self, causal_graph, inplace=False):
        """
        Returns a proper backdoor graph of the `causal_graph`.

        For a `causal_graph` with variable roles `exposure` and `outcome`
        defined, returns it's proper backdoor graph. A proper backdoor graph is
        a graph which removes the first edge of every proper causal path from
        `exposure` to `outcome`.

        Parameters
        ----------
        causal_graph: pgmpy.models.DAG
            The causal graph for which the proper backdoor graph is to be computed.

        inplace: boolean
            If inplace is True, modifies the object itself. Otherwise returns
            a modified copy of self.

        Examples
        --------
        >>> from pgmpy.models import DAG
        >>> from pgmpy.inference import Adjustment
        >>> dag = DAG(
        ...     ebunch=[
        ...         ("x1", "y1"),
        ...         ("x1", "z1"),
        ...         ("z1", "z2"),
        ...         ("z2", "x2"),
        ...         ("y2", "z2"),
        ...     ],
        ...     roles={"exposure": "x1", "outcome": "y1"},
        ... )
        >>> dag_proper = Adjustment()._get_proper_backdoor_graph(dag, inplace=False)
        >>> list(dag_proper.edges())
        [('x1', 'z1'), ('z1', 'z2'), ('z2', 'x2'), ('y2', 'z2')]

        References
        ----------
        [1] Perkovic, Emilija, et al. "Complete graphical characterization and
            construction of adjustment sets in Markov equivalence classes of
            ancestral graphs." The Journal of Machine Learning Research.
        """
        # TODO: Make this work for all graph types.
        model = causal_graph if inplace else causal_graph.copy()
        edges_to_remove = []
        for source in causal_graph.get_role("exposure"):
            paths = nx.all_simple_edge_paths(
                causal_graph, source, causal_graph.get_role("outcome")
            )
            for path in paths:
                edges_to_remove.append(path[0])
        model.remove_edges_from(edges_to_remove)
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
            if len(causal_graph.get_role("exposure")) != 1:
                raise NotImplementedError(
                    "Backdoor identification is only implemented for single exposure variable."
                )
            if len(causal_graph.get_role("outcome")) != 1:
                raise NotImplementedError(
                    "Backdoor identification is only implemented for single outcome variable."
                )

            exposure = causal_graph.get_role("exposure")[0]
            outcome = causal_graph.get_role("outcome")[0]

            backdoor_graph = self._get_proper_backdoor_graph(
                causal_graph, inplace=False
            )
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
            raise NotImplementedError(
                "Backdoor identification with minimal variance is not implemented yet."
            )

        # Step 3: If variant = "all", iterate over all possible sets of adjustment
        #         variables, and return all that are valid.
        elif self.variant == "all":
            exposure = causal_graph.get_role("exposure")[0]
            outcome = causal_graph.get_role("outcome")[0]

            ancestors = causal_graph._get_ancestors_of([exposure, outcome])
            # Remove any variables on the path from exposure to outcome (these cannot be in the adjustment set)
            ancestors -= set(
                itertools.chain(*nx.all_simple_paths(causal_graph, exposure, outcome))
            )
            ancestors -= {exposure, outcome}
            ancestors -= set(causal_graph.latents)

            valid_adj_graphs = []
            for s in _powerset(ancestors):
                adj_causal_graph = causal_graph.with_role(
                    "adjustment", s, inplace=False
                )
                if self.validate(causal_graph=adj_causal_graph):
                    valid_adj_graphs.append(adj_causal_graph)

            return valid_adj_graphs, len(valid_adj_graphs) > 0

    def _validate(self, causal_graph):
        """
        Validate the causal graph for backdoor identification.

        Given a `causal_graph` with variable roles `exposure`, `outcome`, and
        `adjustment` defined, this method checks if the given `adjustment` set
        is valid.

        Parameters
        ----------
        causal_graph: DAG | PDAG | ADMG | MAG | PAG
            The causal graph to validate.

        Returns
        -------
        bool:
            True if the `adjustment` set is valid, False otherwise.
        """
        exposure = causal_graph.get_role("exposure")
        outcome = causal_graph.get_role("outcome")
        adjustment_vars = causal_graph.get_role("adjustment")

        conditional_vars = exposure + adjustment_vars

        predecessors = set()
        for exposure_var in exposure:
            predecessors.update(causal_graph.predecessors(exposure_var))

        parents_d_sep = []
        for pred_var in predecessors:
            outcome_d_seps = []
            for outcome_var in outcome:
                outcome_d_seps.append(
                    causal_graph.is_dconnected(
                        pred_var, outcome_var, observed=conditional_vars
                    )
                )
            parents_d_sep.append(not any(outcome_d_seps))

        return all(parents_d_sep)
