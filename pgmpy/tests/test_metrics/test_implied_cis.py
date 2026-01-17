import networkx as nx
import numpy as np

from pgmpy.base import DAG
from pgmpy.estimators.CITests import chi_square
from pgmpy.metrics import ImpliedCIs
from pgmpy.utils import get_example_model

rng = np.random.default_rng(42)

model_cancer = get_example_model("cancer")
df_cancer = model_cancer.simulate(int(1e3), seed=42)
n_cancer = len(model_cancer.nodes())

model_alarm = get_example_model("alarm")
n_alarm = len(model_alarm.nodes())
model_alarm_random = nx.from_numpy_array(
    np.tril(rng.choice([0, 1], p=[0.9, 0.1], size=(n_alarm, n_alarm)), k=-1),
    create_using=nx.DiGraph,
)
nx.relabel_nodes(
    model_alarm_random,
    {i: list(model_alarm.nodes())[i] for i in range(n_alarm)},
    copy=False,
)
model_alarm_random = DAG(model_alarm_random.edges())

model_cancer_random = nx.from_numpy_array(
    np.tril(rng.choice([0, 1], p=[0.9, 0.1], size=(n_cancer, n_cancer)), k=-1),
    create_using=nx.DiGraph,
)
nx.relabel_nodes(
    model_cancer_random,
    {i: list(model_cancer.nodes())[i] for i in range(n_cancer)},
    copy=False,
)
model_cancer_random = DAG(model_cancer_random.edges())
df_alarm = model_alarm.simulate(int(1e3), seed=42)

implied_cis = ImpliedCIs(ci_test=chi_square)
cancer_tests = implied_cis(X=df_cancer, causal_graph=model_cancer)

assert cancer_tests.shape[0] == 6
assert list(cancer_tests.loc[:, "p-value"].values.round(4)) == [
    0.9816,
    1.0,
    0.3491,
    0.8061,
    0.896,
    0.9917,
]

implied_cis = ImpliedCIs(ci_test=chi_square)
alarm_tests_true = implied_cis(X=df_alarm, causal_graph=model_alarm)

assert alarm_tests_true.shape[0] == 620

implied_cis = ImpliedCIs(ci_test=chi_square)
alarm_tests_random = implied_cis(X=df_alarm, causal_graph=model_alarm_random)
assert alarm_tests_random.shape[0] == 528
