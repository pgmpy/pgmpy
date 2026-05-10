import os
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyagrum as gum
import rpy2.robjects as ro
from plotly.subplots import make_subplots
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import sem

from pgmpy import config
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.example_models import load_model
from pgmpy.models import DiscreteBayesianNetwork

warnings.filterwarnings("ignore")

# add R path
if os.name == "nt":
    r_path = os.environ.get("R_BIN_PATH") or r"C:\Program Files\R\R-4.6.0\bin\x64"
    if os.path.exists(r_path):
        os.environ["PATH"] += os.pathsep + r_path


def get_bnlearn_string(model):
    res = ""
    for node in model.nodes():
        parents = model.get_parents(node)
        if not parents:
            res += f"[{node}]"
        else:
            res += f"[{node}|{':'.join(parents)}]"
    return res


def main():
    # load real model
    print("Loading 'alarm' model...")
    model = load_model("bnlearn/alarm")
    net = DiscreteBayesianNetwork(model.edges())

    # setup pyAgrum network dynamically
    print("Setting up pyAgrum network...")
    bn = gum.BayesNet()
    nodes_map = {}
    for node in model.nodes():
        card = model.get_cpds(node).variable_card
        nodes_map[node] = bn.add(gum.LabelizedVariable(node, node, card))
    for u, v in model.edges():
        bn.addArc(nodes_map[u], nodes_map[v])

    # wrapping R code to fit bnlearn model
    print("Setting up bnlearn (R) network...")
    bnlearn_str = get_bnlearn_string(model)
    ro.r(f"""
    library(bnlearn)
    net_r <- model2network("{bnlearn_str}")
    fit_r <- function(df) {{
        df[] <- lapply(df, factor)
        fit <- bn.fit(net_r, df, method="mle")
        return(TRUE)
    }}
    """)
    fit_r = ro.globalenv["fit_r"]

    # benchmark config stuff
    test_sz = [50000, 100000, 200000, 400000]
    n_iter = 3
    res = []

    print("running benchmarks. checking sizes:", test_sz)

    for sz in test_sz:
        print(f"handling {sz} samples.")
        df = model.simulate(n_samples=sz, show_progress=False)

        t_np, t_torch, t_agrum, t_r = [], [], [], []

        for i in range(n_iter):
            # pgmpy numpy backend
            config.set_backend("numpy")
            net_np = DiscreteBayesianNetwork(model.edges())
            t0 = time.perf_counter()
            est_np = MaximumLikelihoodEstimator(net_np, df)
            net_np.add_cpds(*est_np.get_parameters())
            t_np.append(time.perf_counter() - t0)

            # pgmpy pytorch backend
            config.set_backend("torch")
            net_torch = DiscreteBayesianNetwork(model.edges())
            t0 = time.perf_counter()
            est_torch = MaximumLikelihoodEstimator(net_torch, df)
            net_torch.add_cpds(*est_torch.get_parameters())
            t_torch.append(time.perf_counter() - t0)

            # agrum benchmark (pyAgrum's learning is single-threaded and can be very slow on large datasets)
            t0 = time.perf_counter()
            learner = gum.BNLearner(df)
            if hasattr(learner, "useSmoothingPrior"):
                learner.useSmoothingPrior(1)
            elif hasattr(learner, "useAprioriSmoothing"):
                learner.useAprioriSmoothing(1)
            learner.learnParameters(bn.dag())
            t_agrum.append(time.perf_counter() - t0)

            # R backend
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(df)
            t0 = time.perf_counter()
            fit_r(r_df)
            t_r.append(time.perf_counter() - t0)

        res.append({"Size": sz, "Library": "pgmpy (NumPy)", "MeanTime": np.mean(t_np), "StdErr": sem(t_np)})
        res.append({"Size": sz, "Library": "pgmpy (PyTorch)", "MeanTime": np.mean(t_torch), "StdErr": sem(t_torch)})
        res.append(
            {
                "Size": sz,
                "Library": "pyagrum (C++)",
                "MeanTime": np.nanmean(t_agrum) if not np.isnan(t_agrum).all() else np.nan,
                "StdErr": np.nan,
            }
        )
        res.append({"Size": sz, "Library": "bnlearn (R)", "MeanTime": np.mean(t_r), "StdErr": sem(t_r)})

    # plotting
    df_results = pd.DataFrame(res)
    df_results["txt"] = df_results["MeanTime"].apply(lambda val: f"{val:.3f}s" if pd.notna(val) else "Failed")

    color_map = {
        "pgmpy (NumPy)": "#2563EB",
        "pgmpy (PyTorch)": "#F59E0B",
        "pyagrum (C++)": "#DC2626",
        "bnlearn (R)": "#10B981",
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{s:,} Samples" for s in test_sz],
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    grid = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, sz in enumerate(test_sz):
        r, c = grid[idx]
        chunk = df_results[df_results["Size"] == sz].sort_values("Library")

        for libname in ["pgmpy (NumPy)", "pgmpy (PyTorch)", "pyagrum (C++)", "bnlearn (R)"]:
            row = chunk[chunk["Library"] == libname]

            if not row.empty:
                fig.add_trace(
                    go.Bar(
                        x=[libname],
                        y=row["MeanTime"].values,
                        error_y=dict(type="data", array=row["StdErr"].values, visible=True)
                        if pd.notna(row["StdErr"].values[0])
                        else None,
                        marker=dict(color=color_map[libname]),
                        text=row["txt"].values,
                        textposition="outside",
                        showlegend=(r == 1 and c == 1),
                        name=libname,
                        legendgroup=libname,
                        hovertemplate="<b>%{x}</b><br>Time: %{y:.3f}s<extra></extra>"
                        if pd.notna(row["MeanTime"].values[0])
                        else "<b>%{x}</b><br>Status: Failed<extra></extra>",
                    ),
                    row=r,
                    col=c,
                )

    # set y-axis titles and ranges
    for i in range(1, 3):
        for j in range(1, 3):
            sub_idx = (i - 1) * 2 + j
            if sub_idx <= len(test_sz):
                sz = test_sz[sub_idx - 1]
                mx = df_results[(df_results["Size"] == sz) & (pd.notna(df_results["MeanTime"]))]["MeanTime"].max()
                if pd.notna(mx):
                    fig.update_yaxes(title_text="Mean Time (s)", row=i, col=j, range=[0, mx * 1.3])

    fig.update_layout(
        title_text="Detailed Performance (Scaled per Size)",
        title_x=0.5,
        template="plotly_white",
        height=800,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#E5E7EB",
            borderwidth=1,
        ),
        margin=dict(t=100, b=80, l=80, r=50),
        font=dict(size=11),
    )

    docs_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(docs_dir, "runtime_comparison.html")
    fig.write_html(out_file)

    config.set_backend("numpy")  # reset to default just in case
    print("done! saved to", out_file)


if __name__ == "__main__":
    main()
