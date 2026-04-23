import numpy as np
import pandas as pd
import pytest

from pgmpy.base import PDAG
from pgmpy.estimators import GES
from pgmpy.example_models import load_model


@pytest.fixture
def random_data_estimator():
    rand_data = pd.DataFrame(
        np.random.randint(0, 5, size=(int(1e4), 2)),
        columns=list("AB"),
        dtype="category",
    )
    rand_data["C"] = rand_data["B"]
    return GES(rand_data, use_cache=False)


@pytest.fixture
def titanic_estimators():
    titanic_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")

    titanic_data1 = titanic_data[["Survived", "Sex", "Pclass", "Age", "Embarked"]]
    est1 = GES(titanic_data1, use_cache=False)

    titanic_data2 = titanic_data[["Survived", "Sex", "Pclass"]].astype("category")
    est2 = GES(titanic_data2, use_cache=False)

    return est1, est2


@pytest.fixture
def gaussian_data():
    data = pd.read_csv(
        "pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv",
        index_col=0,
    )
    return data.iloc[:50, :5]


# Optional manual parity check against `causal-learn==0.1.4.5`.
# This is intentionally kept as documentation instead of an executable test,
# because `causal-learn` is an optional dependency.
#
# Reproduction:
#   python - <<'PY'
#   import numpy as np
#   import pandas as pd
#   from causallearn.search.ScoreBased.GES import ges
#   from pgmpy.estimators import BDeu, GES
#
#   def causallearn_cpdag_edge_sets(graph, columns):
#       directed = set()
#       undirected = set()
#       adjacency = graph.graph
#       for i in range(len(columns)):
#           for j in range(i + 1, len(columns)):
#               a, b = adjacency[i, j], adjacency[j, i]
#               if a == -1 and b == -1:
#                   undirected.add(tuple(sorted((columns[i], columns[j]))))
#               elif a == -1 and b == 1:
#                   directed.add((columns[i], columns[j]))
#               elif a == 1 and b == -1:
#                   directed.add((columns[j], columns[i]))
#       return directed, undirected
#
#   def pgmpy_cpdag_edge_sets(pdag):
#       directed = set(pdag.directed_edges)
#       undirected = {tuple(sorted(edge)) for edge in pdag.undirected_edges}
#       return directed, undirected
#
#   def run_case(name, data, columns):
#       df = pd.DataFrame(
#           {col: pd.Series(data[:, i], dtype="category") for i, col in enumerate(columns)}
#       )
#       causal_graph = ges(data, score_func="local_score_BDeu", node_names=columns)["G"]
#       pgmpy_graph = GES(df, use_cache=False).estimate(
#           scoring_method=BDeu(df, equivalent_sample_size=1)
#       )
#       print(name, causallearn_cpdag_edge_sets(causal_graph, columns), pgmpy_cpdag_edge_sets(pgmpy_graph))
#
#   columns = ["A", "B", "C"]
#
#   rng = np.random.default_rng(0)
#   n = 5000
#   a = rng.binomial(1, 0.5, size=n)
#   b = np.where(rng.random(n) < 0.95, a, 1 - a)
#   c = np.where(rng.random(n) < 0.95, b, 1 - b)
#   run_case("chain", np.column_stack([a, b, c]).astype(int), columns)
#
#   rng = np.random.default_rng(1)
#   a = rng.binomial(1, 0.5, size=n)
#   b = rng.binomial(1, 0.5, size=n)
#   base = np.logical_or(a, b).astype(int)
#   noise = rng.binomial(1, 0.03, size=n)
#   c = np.logical_xor(base, noise).astype(int)
#   run_case("collider", np.column_stack([a, b, c]).astype(int), columns)
#   PY
#
# Observed results:
#   chain:
#       causal-learn -> (set(), {("A", "B"), ("B", "C")})
#       pgmpy       -> (set(), {("A", "B"), ("B", "C")})
#   collider:
#       causal-learn -> ({("A", "C"), ("B", "C")}, set())
#       pgmpy       -> ({("A", "C"), ("B", "C")}, set())


def test_insert_orients_t_away_from_v():
    est = GES(pd.DataFrame({"A": [0, 1], "B": [0, 1], "C": [0, 1]}), use_cache=False)

    pdag = PDAG(undirected_ebunch=[("B", "C")])
    pdag.add_nodes_from(["A", "B", "C"])

    new_model = est.insert("A", "B", {"C"}, pdag)

    assert new_model.directed_edges == {("A", "B"), ("C", "B")}
    assert new_model.undirected_edges == set()


def test_legal_edge_deletions_include_both_orders_for_undirected_edges():
    est = GES(pd.DataFrame({"A": [0, 1], "B": [0, 1]}), use_cache=False)

    pdag = PDAG(undirected_ebunch=[("A", "B")])
    pdag.add_nodes_from(["A", "B"])

    assert set(est._legal_edge_deletions(pdag)) == {("A", "B"), ("B", "A")}


def test_cancer_model():
    cancer_model = load_model("bnlearn/cancer")
    data = cancer_model.simulate(3000, seed=0)

    est = GES(data)
    dag = est.estimate()

    assert set(cancer_model.edges) <= set(dag.edges)


def test_estimate_gaussian(gaussian_data):
    est = GES(gaussian_data)

    for score in ["aic-g", "bic-g"]:
        est.estimate(scoring_method=score)
