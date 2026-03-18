import networkx as nx


def generate_complete_dag(dag, ngraphs=10):
    """
    Given a DAG, return ngraphs complete graphs which are a super DAG of the original one.
    Note: their could be more than one possibility, so we return ngraphs of them.
    """
    dags = []
    for argsort in nx.all_topological_sorts(nx.DiGraph(dag)):
        a = dag.copy()
        for i, parent in enumerate(argsort):
            for child in argsort[i + 1 :]:
                a[parent, child] = 1
        dags.append(a)
        if len(dags) == ngraphs:
            return dags
    return dags
