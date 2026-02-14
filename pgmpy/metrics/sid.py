from pgmpy.metrics import _BaseSupervisedMetric
from pgmpy.base import DAG
import networkx as nx


class SID(_BaseSupervisedMetric):
    """
    Structural Intervention Distance (SID).

    Implements SID as defined in:

    Peters, J., & Bühlmann, P. (2013).
    "Structural Intervention Distance for Evaluating Causal Graphs".
    arXiv:1306.1043

    SID counts the number of ordered pairs (i, j), i != j,
    for which the intervention distribution P(j | do(i))
    differs between the true and estimated DAG.

    This implementation assumes both graphs are DAGs
    without latent confounding.
    """

    _tags = {
        "supported_graph_types": (DAG,)
    }

    def _intervention_graph(self, graph, node):
        """
        Returns graph after intervention do(node),
        i.e., removing all incoming edges into node.
        """
        g_do = graph.copy()

        for parent in list(g_do.predecessors(node)):
            g_do.remove_edge(parent, node)

        return g_do

    def _evaluate(self, true_causal_graph, est_causal_graph):

        if set(true_causal_graph.nodes()) != set(est_causal_graph.nodes()):
            raise ValueError("The graphs must have the same nodes.")

        if not nx.is_directed_acyclic_graph(true_causal_graph):
            raise ValueError("true_causal_graph must be a DAG.")

        if not nx.is_directed_acyclic_graph(est_causal_graph):
            raise ValueError("est_causal_graph must be a DAG.")

        nodes = list(true_causal_graph.nodes())
        sid = 0

        # Precompute intervention descendants
        true_effects = {}
        est_effects = {}

        for i in nodes:
            g_true_do = self._intervention_graph(true_causal_graph, i)
            g_est_do = self._intervention_graph(est_causal_graph, i)

            true_effects[i] = nx.descendants(g_true_do, i)
            est_effects[i] = nx.descendants(g_est_do, i)

        for i in nodes:
            for j in nodes:
                if i == j:
                    continue

                true_has_effect = j in true_effects[i]
                est_has_effect = j in est_effects[i]

                if true_has_effect != est_has_effect:
                    sid += 1

        return sid
