import networkx as nx

from pgmpy.identification import BaseIdentification


class BackdoorIdentification(BaseIdentification):
    """
    Backdoor identification for finding adjustment sets in causal graphs.

    This class implements the backdoor criterion for identifying causal effects
    in a directed acyclic graph (DAG). It provides methods to check if a set of
    variables satisfies the backdoor criterion and to compute the backdoor
    adjustment formula.
    """

    def __init__(self, variant="minimal"):
        """
        Initialize the BackdoorIdentification instance.

        Parameters
        ----------
        variant: str
            The variant of backdoor identification to use. Default is 'minimal'.
            - 'minimal': Returns the smallest adjustment set.
            - 'all': Returns all adjustment sets that satisfy the backdoor criterion.
        """
        self.variant = variant

    def get_proper_backdoor_graph(self, causal_graph, inplace=False):
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
        >>> dag_proper = BackdoorIdentification().get_proper_backdoor_graph(
        ...     dag, inplace=False
        ... )
        >>> dag_proper.edges()

        References
        ----------
        [1] Perkovic, Emilija, et al. "Complete graphical characterization and construction of adjustment sets in
            Markov equivalence classes of ancestral graphs." The Journal of Machine Learning Research 18.1
            (2017): 8132-8193.
        """
        model = causal_graph if inplace else causal_graph.copy()
        edges_to_remove = []
        for source in causal_graph.get_roles("exposure"):
            paths = nx.all_simple_edge_paths(
                causal_graph, source, causal_graph.get_roles("outcome")
            )
            for path in paths:
                edges_to_remove.append(path[0])
        model.remove_edges_from(edges_to_remove)
        return model

    def _identify(self, causal_graph):
        """
        Identify adjustment sets using the backdoor criterion.

        Parameters:
        causal_graph (pgmpy.models.DAG): The causal graph to analyze.

        Returns:
        list: A list of adjustment sets that satisfy the backdoor criterion.
        """
        backdoor_graph = self.get_proper_backdoor_graph(causal_graph, inplace=False)
        return backdoor_graph.minimal_dseparator(
            causal_graph.get_roles("exposure"), causal_graph.get_roles("outcome")
        )
