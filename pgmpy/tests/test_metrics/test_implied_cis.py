import networkx as nx
import numpy as np

from pgmpy.base import DAG
from pgmpy.estimators.CITests import chi_square
from pgmpy.metrics import ImpliedCIs
from pgmpy.utils import get_example_model

RNG = np.random.default_rng(42)
SEED = 42
N_SAMPLES = int(1e3)


def simulate(model, n_samples=N_SAMPLES, seed=SEED):
    return model.simulate(n_samples, seed=seed)


def random_dag_from_nodes(nodes, rng=RNG, edge_prob=0.1):
    n = len(nodes)
    adj = np.tril(rng.choice([0, 1], p=[1 - edge_prob, edge_prob], size=(n, n)), k=-1)
    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    nx.relabel_nodes(g, {i: nodes[i] for i in range(n)}, copy=False)
    dag = DAG(g.edges())
    dag.add_nodes_from(nodes)
    return dag


def implied_ci_tests(X, causal_graph, ci_test=chi_square):
    return ImpliedCIs(ci_test=ci_test)(X=X, causal_graph=causal_graph)


def assert_pvalues(df, expected, col="p-value", ndigits=4):
    got = df.loc[:, col].to_numpy().round(ndigits).tolist()
    assert got == expected


# --- cancer ---
model_cancer = get_example_model("cancer")
df_cancer = simulate(model_cancer)

cancer_tests = implied_ci_tests(df_cancer, model_cancer)
assert cancer_tests.shape[0] == 6
assert_pvalues(
    cancer_tests,
    [0.9816, 1.0, 0.3491, 0.8061, 0.8960, 0.9917],
)

# --- alarm ---
model_alarm = get_example_model("alarm")
df_alarm = simulate(model_alarm)

alarm_tests_true = implied_ci_tests(df_alarm, model_alarm)
assert alarm_tests_true.shape[0] == 620

model_alarm_random = random_dag_from_nodes(list(model_alarm.nodes()))
alarm_tests_random = implied_ci_tests(df_alarm, model_alarm_random)
assert alarm_tests_random.shape[0] == 601
