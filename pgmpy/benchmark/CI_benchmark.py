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

sample_sizes = [20, 40, 100, 500, 1000]
n_cond_vars_list = [1, 3]
noise_levels = [0.5, 1]
n_repeats = 10
significance_levels = [0.001, 0.01, 0.1, 1.0]
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
    significance_levels,
    n_cond_vars_list,
):
    results = []
    for dgm_name, dgm in dgms.items():
        print(f"\nRunning DGM: {dgm_name}")
        compatible_tests = dgm_to_citests[dgm_name]
        for n_cond_vars in n_cond_vars_list:
            for n in sample_sizes:
                for noise in noise_levels:
                    for significance_level in significance_levels:
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
                                            "significance_level": significance_level,
                                            "n_cond_vars": n_cond_vars,
                                            "effect_type": effect_type,
                                            "effect_size": eff,
                                            "repeat": rep,
                                            "ci_test": test_name,
                                            "rejected_null": (
                                                not accepted
                                                if accepted is not None
                                                else None
                                            ),
                                        }
                                    )
    df_results = pd.DataFrame(results)
    summary = []
    group_cols = [
        "dgm",
        "sample_size",
        "noise_level",
        "significance_level",
        "n_cond_vars",
        "ci_test",
    ]
    for keys, group in df_results.groupby(group_cols):
        null_group = group[group.effect_type == "null"]
        alt_group = group[group.effect_type == "alt"]
        type1 = null_group["rejected_null"].mean()
        type2 = 1 - alt_group["rejected_null"].mean()
        power = 1 - type2
        summary.append(
            dict(
                zip(group_cols, keys),
                type1_error=type1,
                type2_error=type2,
                power=power,
                N_null=len(null_group),
                N_alt=len(alt_group),
            )
        )
    df_summary = pd.DataFrame(summary)
    return df_results, df_summary


def plot_benchmarks(df_summary, plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)
    methods = sorted(df_summary["ci_test"].unique())
    palette = sns.color_palette("Set1", n_colors=len(methods))

    for dgm in df_summary["dgm"].unique():
        df_dgm = df_summary[df_summary["dgm"] == dgm]
        sample_sizes = sorted(df_dgm["sample_size"].unique())
        n_cond_vars_list = sorted(df_dgm["n_cond_vars"].unique())

        fig2, axes2 = plt.subplots(
            len(sample_sizes),
            len(n_cond_vars_list),
            figsize=(4 * len(n_cond_vars_list), 2.5 * len(sample_sizes)),
            sharex=True,
            sharey=True,
        )
        if len(sample_sizes) == 1 and len(n_cond_vars_list) == 1:
            axes2 = np.array([[axes2]])
        elif len(sample_sizes) == 1 or len(n_cond_vars_list) == 1:
            axes2 = axes2.reshape(len(sample_sizes), len(n_cond_vars_list))

        for i, n in enumerate(sample_sizes):
            for j, k in enumerate(n_cond_vars_list):
                ax = axes2[i, j]
                subset = df_dgm[
                    (df_dgm["sample_size"] == n) & (df_dgm["n_cond_vars"] == k)
                ]
                for method, color in zip(methods, palette):
                    s = subset[subset["ci_test"] == method]
                    if not s.empty:
                        x = np.log10(s["significance_level"])
                        y = np.log10(s["type2_error"])
                        # Sort for smooth lines
                        sort_idx = np.argsort(x)
                        ax.plot(
                            x.iloc[sort_idx],
                            y.iloc[sort_idx],
                            marker="o",
                            linestyle="-",
                            label=method,
                            color=color,
                        )
                if i == 0:
                    ax.set_title(f"Cond.vars: {k}")
                if j == 0:
                    ax.set_ylabel(f"n={n}\nlog10 Type II Error")
                if i == len(sample_sizes) - 1:
                    ax.set_xlabel("log10 Significance Level")
                ax.grid(True, alpha=0.4)
        handles, labels = axes2[0, 0].get_legend_handles_labels()
        fig2.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(methods),
            bbox_to_anchor=(0.5, 1.15),
        )
        fig2.suptitle(f"Type II Error vs Significance Level for {dgm}", y=1.12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname1 = f"{plot_dir}/{dgm}_fig2_typeII_vs_signif.png"
        plt.savefig(fname1, bbox_inches="tight")
        plt.close(fig2)

        fig3, axes3 = plt.subplots(
            len(n_cond_vars_list),
            1,
            figsize=(5, 2.5 * len(n_cond_vars_list)),
            sharex=True,
            sharey=True,
        )
        if len(n_cond_vars_list) == 1:
            axes3 = [axes3]
        for j, k in enumerate(n_cond_vars_list):
            ax = axes3[j]
            subset = df_dgm[df_dgm["n_cond_vars"] == k]
            for method, color in zip(methods, palette):
                s = subset[subset["ci_test"] == method]
                if not s.empty:
                    sort_idx = np.argsort(s["sample_size"])
                    ax.plot(
                        s["sample_size"].iloc[sort_idx],
                        s["power"].iloc[sort_idx],
                        marker="o",
                        linestyle="-",
                        label=method,
                        color=color,
                    )
            ax.set_title(f"Cond.vars: {k}")
            if j == 0:
                ax.set_ylabel("Power")
            if j == len(n_cond_vars_list) - 1:
                ax.set_xlabel("Sample Size")
            ax.grid(True, alpha=0.4)
        handles, labels = axes3[0].get_legend_handles_labels()
        fig3.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(methods),
            bbox_to_anchor=(0.5, 1.15),
        )
        fig3.suptitle(f"Power vs Sample Size for {dgm}", y=1.12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname2 = f"{plot_dir}/{dgm}_fig3_power_vs_samplesize.png"
        plt.savefig(fname2, bbox_inches="tight")
        plt.close(fig3)

        print(f"Saved Plot 1: {fname1}")
        print(f"Saved Plot 2: {fname2}")


if __name__ == "__main__":
    df_results, df_summary = run_benchmark(
        dgms,
        DGM_TO_CITESTS,
        ci_tests,
        sample_sizes,
        noise_levels,
        n_repeats,
        significance_levels,
        n_cond_vars_list,
    )

    df_results.to_csv("ci_benchmark_raw_result.csv", index=False)
    df_summary.to_csv("ci_benchmark_summaries.csv", index=False)
    print(df_summary)
    print(
        "\nDetailed results and summary saved to ci_benchmark_raw_result.csv and ci_benchmark_summaries.csv"
    )

    plot_benchmarks(df_summary)
