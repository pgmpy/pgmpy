import inspect
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
}


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


def run_benchmark(
    dgms,
    dgm_to_citests,
    ci_tests,
    sample_sizes,
    noise_levels,
    n_repeats,
    significance_level,
    n_cond_vars=1,
):
    results = []
    for dgm_name, dgm in dgms.items():
        print(f"Running DGM: {dgm_name}")
        compatible_tests = dgm_to_citests[dgm_name]
        for n in sample_sizes:
            for noise in noise_levels:
                for effect_type, eff in [("null", 0.0), ("alt", 1.0)]:
                    for rep in range(n_repeats):
                        dgm_kwargs = dict(
                            n_samples=n,
                            effect_size=eff,
                            noise_std=noise,
                            n_cond_vars=n_cond_vars,
                            seed=rep,
                            dependent=True if effect_type == "alt" else False,
                        )
                        df = dgm(**dgm_kwargs)
                        z_cols = [
                            col
                            for col, typ in df.attrs["variable_types"].items()
                            if col.startswith("Z")
                        ]
                        for test_name in compatible_tests:
                            ci_func = ci_tests[test_name]
                            accepted = None
                            try:
                                accepted = ci_func(
                                    "X",
                                    "Y",
                                    z_cols,
                                    df,
                                    boolean=True,
                                    significance_level=significance_level,
                                )
                            except Exception:
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
    return df_results, df_summary


def plot_benchmarks(df_summary, plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)
    for dgm in df_summary["dgm"].unique():
        df_dgm = df_summary[df_summary["dgm"] == dgm]
        for metric in ["type1_error", "power"]:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df_dgm,
                x="sample_size",
                y=metric,
                hue="ci_test",
                marker="o",
            )
            plt.title(f"{metric.replace('_', ' ').capitalize()} for {dgm}")
            plt.ylabel(metric.replace("_", " ").capitalize())
            plt.xlabel("Sample Size")
            plt.legend(title="CI Test")
            plt.tight_layout()
            plot_filename = f"{plot_dir}/{dgm}_{metric}.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved plot: {plot_filename}")


if __name__ == "__main__":
    # Configurable parameters
    sample_sizes = [100, 500, 1000]
    noise_levels = [0.5, 1.0]
    n_repeats = 50
    significance_level = 0.05
    n_cond_vars = 1

    df_results, df_summary = run_benchmark(
        dgms,
        DGM_TO_CITESTS,
        ci_tests,
        sample_sizes,
        noise_levels,
        n_repeats,
        significance_level,
        n_cond_vars,
    )

    df_results.to_csv("ci_benchmark_raw_result.csv", index=False)
    df_summary.to_csv("ci_benchmark_summaries.csv", index=False)
    print(df_summary)
    print(
        "\nDetailed results and summary saved to ci_benchmark_raw_result.csv and ci_benchmark_summaries.csv"
    )

    plot_benchmarks(df_summary)
