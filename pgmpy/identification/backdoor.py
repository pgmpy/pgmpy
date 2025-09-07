import networkx as nx

from pgmpy.base import DAG
from pgmpy.identification import BaseIdentification
from pgmpy.utils.sets import _powerset


class BackdoorIdentification(BaseIdentification):
    """
    Backdoor identification for finding adjustment sets in causal graphs.

    This class implements the backdoor criterion for identifying causal effects
    in a directed acyclic graph (DAG). Additionally, it provides methods to
    check if the current set of variables with role `adjustment` satisfy the
    backdoor criterion and to compute the backdoor adjustment formula.

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
    >>> dag_with_adj = BackdoorIdentification(variant="minimal").identify(dag)
    >>> dag_with_adj.roles
    {'exposure': 'x1', 'outcome': 'y1', 'adjustment': ['z1', 'z2']}
    >>> BackdoorIdentification.validate(dag)
    """

    def __init__(self, variant="minimal"):
        self.variant = variant

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
        >>> from pgmpy.inference import BackdoorIdnentification
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
        >>> dag_proper = BackdoorIdentification()._get_proper_backdoor_graph(
        ...     dag, inplace=False
        ... )
        >>> list(dag_proper.edges())
        [('x1', 'z1'), ('z1', 'z2'), ('z2', 'x2'), ('y2', 'z2')]

        References
        ----------
        [1] Perkovic, Emilija, et al. "Complete graphical characterization and construction of adjustment sets in
            Markov equivalence classes of ancestral graphs." The Journal of Machine Learning Research 18.1
            (2017): 8132-8193.
        """
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
        causal_graph: DAG | PDAG | MAG | PAG
            The causal graph with the identified adjustment set added as role `adjustment`.
        """
        if not isinstance(causal_graph, DAG):
            raise NotImplementedError(
                "Backdoor identification is only implemented for DAGs."
            )

        backdoor_graph = self._get_proper_backdoor_graph(causal_graph, inplace=False)
        if self.variant == "minimal":
            return backdoor_graph.minimal_dseparator(
                causal_graph.get_roles("exposure"), causal_graph.get_roles("outcome")
            )

        elif self.variant == "minimal_variance":
            raise NotImplementedError(
                "Backdoor identification with minimal variance is not implemented yet."
            )

        elif self.variant == "all":
            ancestors = causal_graph.ancestors(
                causal_graph.get_roles("exposure") + causal_graph.get_roles("outcome")
            )

            valid_adjustment_sets = []
            for s in _powerset(ancestors):
                if self.validate(causal_graph=causal_graph, adjustment_set=s):
                    valid_adjustment_sets.append(s)
            return valid_adjustment_sets

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
        conditional_vars = causal_graph.get_roles("exposure") + causal_graph.get_roles(
            "adjustment"
        )

        parents_d_sep = []
        for p in self.dag.predecessors(causal_graph.get_roles("exposure")):
            parents_d_sep.append(
                not self.dag.is_dconnected(
                    p, causal_graph.get_roles("outcome"), observed=conditional_vars
                )
            )
        return all(parents_d_sep)
