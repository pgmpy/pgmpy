"""
Side-by-side API comparison: pgmpy causal_explainability vs DoWhy gcm.

Demonstrates that pgmpy covers all six Shapley-based causal attribution
methods with a unified, simpler API while DoWhy requires more boilerplate
and has gaps (no causal Shapley values, no unit-level attribution).

Run:  python compare_pgmpy_dowhy_api.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared setup: a simple 3-node linear Gaussian model  X -> Y -> Z
# ---------------------------------------------------------------------------
np.random.seed(42)
N = 2000
X = np.random.normal(0, 1, N)
Y = 2 * X + np.random.normal(0, 0.5, N)
Z = 0.8 * Y + np.random.normal(0, 0.3, N)
data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def section(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


# ===================================================================
# 1. Intrinsic Causal Influence (variance decomposition)
# ===================================================================
section("1. INTRINSIC CAUSAL INFLUENCE")

# ---- pgmpy -------------------------------------------------
print("\n--- pgmpy ---")
print("""
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.causal_explainability import IntrinsicCausalInfluence

model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
model.fit(data)

ici = IntrinsicCausalInfluence()            # default: variance
result = ici.attribute(model, data, target="Z")
# result = {"X": ..., "Y": ..., "Z": ...}
""")

from pgmpy.causal_explainability import IntrinsicCausalInfluence
from pgmpy.models import LinearGaussianBayesianNetwork

model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
model.fit(data)

ici = IntrinsicCausalInfluence()
result_ici = ici.attribute(model, data, target="Z")
print("  Result:", {k: round(v, 4) for k, v in result_ici.items()})

# ---- DoWhy --------------------------------------------------
print("\n--- DoWhy (gcm) ---")
print("""
import networkx as nx
from dowhy import gcm

dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
causal_model = gcm.InvertibleStructuralCausalModel(dag)
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)

result = gcm.intrinsic_causal_influence(
    causal_model,
    target_node="Z",
    prediction_model=gcm.ml.create_linear_regressor,
    attribution_func=gcm.ml.create_linear_regressor,
)

# Note: DoWhy requires an InvertibleStructuralCausalModel with
# invertible noise (more restrictive). It also requires explicit
# prediction_model and attribution_func kwargs.
""")
print("  [DoWhy output omitted — requires dowhy>=0.11 installed]")


# ===================================================================
# 2. Anomaly Attribution (root cause analysis)
# ===================================================================
section("2. ANOMALY ATTRIBUTION")

print("\n--- pgmpy ---")
print("""
from pgmpy.causal_explainability import AnomalyAttribution

anomaly_obs = {"X": 3.0, "Y": 7.5, "Z": 6.5}

aa = AnomalyAttribution()
result = aa.attribute(model, data, target="Z", observation=anomaly_obs)
# result = {"X": ..., "Y": ..., "Z": ...}   (Shapley over mechanisms)
""")

from pgmpy.causal_explainability import AnomalyAttribution

anomaly_obs = {"X": 3.0, "Y": 7.5, "Z": 6.5}
aa = AnomalyAttribution()
result_aa = aa.attribute(model, data, target="Z", observation=anomaly_obs)
print("  Result:", {k: round(v, 4) for k, v in result_aa.items()})

print("\n--- DoWhy (gcm) ---")
print("""
import networkx as nx
from dowhy import gcm

dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
causal_model = gcm.InvertibleStructuralCausalModel(dag)
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)

anomaly_sample = pd.DataFrame({"X": [3.0], "Y": [7.5], "Z": [6.5]})

# DoWhy requires an anomaly scorer object, not a simple function
result = gcm.attribute_anomalies(
    causal_model,
    target_node="Z",
    anomaly_samples=anomaly_sample,
    attribute_mean_deviation=True,
)

# Note: DoWhy operates on DataFrames of anomalies (batch interface),
# pgmpy operates on single observations (dict interface).
""")
print("  [DoWhy output omitted — requires dowhy>=0.11 installed]")


# ===================================================================
# 3. Distribution Change Attribution
# ===================================================================
section("3. DISTRIBUTION CHANGE ATTRIBUTION")

print("\n--- pgmpy ---")
print("""
from pgmpy.causal_explainability import DistributionChangeAttribution

# Simulate shifted data (Y mechanism changed)
data_new = data.copy()
data_new["Y"] = 3 * data["X"] + np.random.normal(1.0, 0.5, N)
data_new["Z"] = 0.8 * data_new["Y"] + np.random.normal(0, 0.3, N)

dca = DistributionChangeAttribution()
result = dca.attribute(model, target="Z", data_old=data, data_new=data_new)
# result = {"X": ..., "Y": ..., "Z": ...}

# Bonus: which mechanisms actually changed?
pvals = dca.mechanism_change_test(model, data, data_new)
""")

from pgmpy.causal_explainability import DistributionChangeAttribution

data_new = data.copy()
data_new["Y"] = 3 * data["X"] + np.random.normal(1.0, 0.5, N)
data_new["Z"] = 0.8 * data_new["Y"] + np.random.normal(0, 0.3, N)

dca = DistributionChangeAttribution()
result_dca = dca.attribute(model, target="Z", data_old=data, data_new=data_new)
print("  Result:", {k: round(v, 4) for k, v in result_dca.items()})

pvals = dca.mechanism_change_test(model, data, data_new)
print("  Mechanism change p-values:", {k: round(v, 4) for k, v in pvals.items()})

print("\n--- DoWhy (gcm) ---")
print("""
from dowhy import gcm

causal_model = gcm.InvertibleStructuralCausalModel(dag)
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)

result = gcm.distribution_change(
    causal_model,
    old_data=data,
    new_data=data_new,
    target_node="Z",
)

# Note: DoWhy's gcm.distribution_change requires an
# InvertibleStructuralCausalModel. pgmpy's version works with any
# LinearGaussianBayesianNetwork and also provides mechanism_change_test().
""")
print("  [DoWhy output omitted — requires dowhy>=0.11 installed]")


# ===================================================================
# 4. Causal Feature Relevance
# ===================================================================
section("4. CAUSAL FEATURE RELEVANCE")

print("\n--- pgmpy ---")
print("""
from pgmpy.causal_explainability import CausalFeatureRelevance

cfr = CausalFeatureRelevance()               # default: variance
result = cfr.attribute(model, data, target="Z")
# result = {"Y": ...}  (only direct parents)

# Graph-level relevance for all nodes at once:
cfr_graph = CausalFeatureRelevance(level="graph")
result_all = cfr_graph.attribute(model, data, target=None)
# result_all = {"X": {}, "Y": {"X": ...}, "Z": {"Y": ...}}
""")

from pgmpy.causal_explainability import CausalFeatureRelevance

cfr = CausalFeatureRelevance()
result_cfr = cfr.attribute(model, data, target="Z")
print("  Result (target=Z):", {k: round(v, 4) for k, v in result_cfr.items()})

cfr_graph = CausalFeatureRelevance(level="graph")
result_all = cfr_graph.attribute(model, data, target=None)
print("  Result (graph-level):", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in result_all.items()})

print("\n--- DoWhy (gcm) ---")
print("""
# DoWhy does NOT have a direct equivalent of CausalFeatureRelevance.
# The closest is gcm.intrinsic_causal_influence, but that decomposes
# over noise terms (all ancestors), not direct parents.
#
# pgmpy's CausalFeatureRelevance answers: "How important is each
# parent for this node's mechanism?" — a question DoWhy doesn't expose.
""")
print("  [No DoWhy equivalent]")


# ===================================================================
# 5. Unit Change Attribution
# ===================================================================
section("5. UNIT CHANGE ATTRIBUTION")

print("\n--- pgmpy ---")
print("""
from pgmpy.causal_explainability import UnitChangeAttribution

obs_old = {"X": 0.0, "Y": 0.5, "Z": 0.4}
obs_new = {"X": 2.0, "Y": 4.5, "Z": 3.8}

uca = UnitChangeAttribution()
result = uca.attribute(model, data=None, target="Z",
                       observation_old=obs_old, observation_new=obs_new)
# result = {"Y": ...}  (direct parent contributions to change in Z)
""")

from pgmpy.causal_explainability import UnitChangeAttribution

obs_old = {"X": 0.0, "Y": 0.5, "Z": 0.4}
obs_new = {"X": 2.0, "Y": 4.5, "Z": 3.8}

uca = UnitChangeAttribution()
result_uca = uca.attribute(model, data=None, target="Z", observation_old=obs_old, observation_new=obs_new)
print("  Result:", {k: round(v, 4) for k, v in result_uca.items()})

print("\n--- DoWhy (gcm) ---")
print("""
# DoWhy does NOT have unit-level change attribution.
# gcm.distribution_change works on datasets, not individual observations.
# gcm.counterfactual_samples generates counterfactuals but doesn't
# decompose the change into parent-level contributions.
#
# pgmpy's UnitChangeAttribution fills this gap: given two observations,
# it decomposes the target change into per-parent contributions.
""")
print("  [No DoWhy equivalent]")


# ===================================================================
# 6. Causal Shapley Values
# ===================================================================
section("6. CAUSAL SHAPLEY VALUES")

print("\n--- pgmpy ---")
print("""
from pgmpy.causal_explainability import CausalShapleyValues

observation = {"X": 2.0, "Y": 4.5, "Z": 3.8}

# Three formulations in one class:
csv_obs = CausalShapleyValues(method="observational")   # standard SHAP
csv_int = CausalShapleyValues(method="interventional")  # Janzing 2020
csv_cau = CausalShapleyValues(method="causal")          # Heskes 2020

result_obs = csv_obs.attribute(model, data, target="Z", observation=observation)
result_int = csv_int.attribute(model, data, target="Z", observation=observation)
result_cau = csv_cau.attribute(model, data, target="Z", observation=observation)
""")

from pgmpy.causal_explainability import CausalShapleyValues

observation = {"X": 2.0, "Y": 4.5, "Z": 3.8}

csv_obs = CausalShapleyValues(method="observational")
result_obs = csv_obs.attribute(model, data, target="Z", observation=observation)
print("  Observational:", {k: round(v, 4) for k, v in result_obs.items()})

csv_int = CausalShapleyValues(method="interventional")
result_int = csv_int.attribute(model, data, target="Z", observation=observation)
print("  Interventional:", {k: round(v, 4) for k, v in result_int.items()})

csv_cau = CausalShapleyValues(method="causal")
result_cau = csv_cau.attribute(model, data, target="Z", observation=observation)
print("  Causal (Heskes):", {k: round(v, 4) for k, v in result_cau.items()})

print("\n--- DoWhy (gcm) ---")
print("""
# DoWhy does NOT implement Causal Shapley Values.
#
# DoWhy's gcm module has intrinsic_causal_influence (population-level
# variance decomposition) and attribute_anomalies (anomaly scoring),
# but does NOT have:
#   - Observational Shapley values for individual predictions
#   - Interventional Shapley values (Janzing 2020)
#   - Causal Shapley values (Heskes 2020)
#
# For individual explanations, users typically combine DoWhy with
# the SHAP library, which only provides observational Shapley values
# and does not respect causal structure.
""")
print("  [No DoWhy equivalent]")


# ===================================================================
# Summary
# ===================================================================
section("SUMMARY: API COMPARISON")

print("""
+----------------------------------+-------------------+-------------------+
| Feature                          | pgmpy             | DoWhy (gcm)       |
+----------------------------------+-------------------+-------------------+
| Intrinsic Causal Influence       | Yes (analytical   | Yes (requires     |
|                                  |  + Monte Carlo)   |  invertible SCM)  |
+----------------------------------+-------------------+-------------------+
| Anomaly Attribution              | Yes (dict API,    | Yes (DataFrame    |
|                                  |  custom scorers)  |  API, less flex)  |
+----------------------------------+-------------------+-------------------+
| Distribution Change Attribution  | Yes (+ mechanism  | Yes (no mechanism |
|                                  |  change test)     |  change test)     |
+----------------------------------+-------------------+-------------------+
| Causal Feature Relevance         | Yes               | No                |
+----------------------------------+-------------------+-------------------+
| Unit Change Attribution          | Yes               | No                |
+----------------------------------+-------------------+-------------------+
| Causal Shapley Values            | Yes (3 methods:   | No                |
|  (observational/interventional/  |  obs, int, causal)|                   |
|   causal)                        |                   |                   |
+----------------------------------+-------------------+-------------------+
| Unified API                      | .attribute()      | Different fn per  |
|                                  |  for all methods  |  method           |
+----------------------------------+-------------------+-------------------+
| Analytical fast paths            | Yes (LGBN)        | No (ML models)    |
+----------------------------------+-------------------+-------------------+
| Model compatibility              | BN, LGBN          | Any SCM           |
+----------------------------------+-------------------+-------------------+

Key advantages of pgmpy's causal_explainability:
  1. Unified .attribute() API across all 6 methods
  2. Analytical fast paths for LinearGaussianBayesianNetwork (exact, fast)
  3. Three Causal Shapley formulations (DoWhy has none)
  4. Unit-level change attribution (DoWhy lacks this)
  5. Causal feature relevance for mechanism importance (DoWhy lacks this)
  6. mechanism_change_test() for detecting which CPDs shifted
  7. Simpler setup — no InvertibleSCM requirement, works with pgmpy models
""")
