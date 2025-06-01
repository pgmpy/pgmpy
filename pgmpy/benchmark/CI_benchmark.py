import inspect
import warnings

import numpy as np
import pandas as pd

from pgmpy.benchmark.data_generator_mechanism import DGP_REGISTRY
from pgmpy.estimators import CITests

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

DGM_TO_CITESTS = {
    "linear_gaussian": ["pearsonr", "gcm", "pillai"],
    "nonlinear_gaussian": ["gcm", "pillai"],
    "non_gaussian_continuous": ["gcm", "pillai"],
    "discrete_categorical": [
        "chi_square",
        "log_likelihood",
        "modified_log_likelihood",
        "pillai",
    ],
    "mixed_data": ["pillai"],
    # "user_defined": ["pearsonr", "gcm", "chi_square", "log_likelihood", "modified_log_likelihood", "pillai"], All CITests availablle for custom dgm.
}


# Custom generator example for user_defined
def pgmpy_custom_generator(n_samples, effect_size=1.0, noise_std=1.0, seed=None):
    # Example: simple linear_gaussian with 1 Z
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_samples,))
    X = effect_size * Z + rng.normal(scale=noise_std, size=n_samples)
    Y = effect_size * Z + rng.normal(scale=noise_std, size=n_samples)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


dgms = {name: DGP_REGISTRY[name] for name in DGM_TO_CITESTS.keys()}

sample_sizes = [100, 500, 1000]
noise_levels = [0.5, 1.0]
n_repeats = 50
significance_level = 0.05
effect_sizes = {"null": 0, "alt": 1.0}

ci_tests = {
    "pearsonr": CITests.pearsonr,
    "gcm": CITests.gcm,
    "chi_square": CITests.chi_square,
    "log_likelihood": CITests.log_likelihood,
    "modified_log_likelihood": CITests.modified_log_likelihood,
    "pillai": CITests.pillai_trace,
}


def has_zero_counts(df, x, y, z_cols):
    group_cols = [x, y] + list(z_cols)
    try:
        counts = df.groupby(group_cols, observed=False).size()
        return (counts == 0).any()
    except Exception:
        return False


def call_dgm(dgm, dgm_name, my_custom_generator, **kwargs):
    if dgm_name == "user_defined":
        return dgm(generator_func=my_custom_generator, **kwargs)
    else:
        sig = inspect.signature(dgm)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return dgm(**valid_kwargs)


results = []

for dgm_name, dgm in dgms.items():
    print(f"Running DGM: {dgm_name}")
    compatible_tests = DGM_TO_CITESTS[dgm_name]
    for n in sample_sizes:
        for noise in noise_levels:
            for effect_type, eff in effect_sizes.items():
                for rep in range(n_repeats):
                    dgm_kwargs = dict(
                        n_samples=n, effect_size=eff, noise_std=noise, seed=rep
                    )
                    df = call_dgm(dgm, dgm_name, pgmpy_custom_generator, **dgm_kwargs)
                    # Find Z columns (all columns starting with "Z")
                    z_cols = [col for col in df.columns if col.lower().startswith("z")]
                    if not z_cols:
                        z_cols = [col for col in df.columns if col not in ["X", "Y"]]
                    # Ensure categorical columns are marked as such for discrete/mixed DGMs
                    if dgm_name in ["discrete_categorical", "mixed_data"]:
                        for col in df.columns:
                            if df[col].dtype != float and df[col].nunique() < 10:
                                df[col] = df[col].astype("category")
                    for test_name in compatible_tests:
                        ci_func = ci_tests[test_name]
                        skip_test = False
                        if test_name in [
                            "chi_square",
                            "log_likelihood",
                            "modified_log_likelihood",
                        ]:
                            if has_zero_counts(df, "X", "Y", z_cols):
                                accepted = None
                                skip_test = True
                        if not skip_test:
                            try:
                                accepted = ci_func(
                                    "X",
                                    "Y",
                                    z_cols,
                                    df,
                                    boolean=True,
                                    significance_level=significance_level,
                                )
                            except Exception as e:
                                accepted = None
                        results.append(
                            {
                                "dgm": dgm_name,
                                "sample_size": n,
                                "noise_level": noise,
                                "effect_type": effect_type,
                                "effect_size": eff,
                                "repeat": rep,
                                "ci_test": test_name,
                                "rejected_null": (
                                    not accepted if accepted is not None else None
                                ),
                            }
                        )

df_results = pd.DataFrame(results)

summary = []
for (dgm, n, noise, ci_test), group in df_results.groupby(
    ["dgm", "sample_size", "noise_level", "ci_test"]
):
    null_group = group[group.effect_type == "null"]
    alt_group = group[group.effect_type == "alt"]
    type1 = null_group["rejected_null"].mean()
    type2 = 1 - alt_group["rejected_null"].mean()
    power = 1 - type2
    summary.append(
        {
            "dgm": dgm,
            "sample_size": n,
            "noise_level": noise,
            "ci_test": ci_test,
            "type1_error": type1,
            "type2_error": type2,
            "power": power,
            "N_null": len(null_group),
            "N_alt": len(alt_group),
        }
    )

df_summary = pd.DataFrame(summary)
df_results.to_csv("ci_benchmark_raw_results.csv", index=False)
df_summary.to_csv("ci_benchmark_summary.csv", index=False)
print(df_summary)
print(
    "\nDetailed results and summary saved to ci_benchmark_raw_results.csv and ci_benchmark_summary.csv"
)
