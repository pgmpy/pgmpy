import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import sem

# add R path to windows env so rpy2 doesnt complain
os.environ["PATH"] += os.pathsep + r"C:\Program Files\R\R-4.6.0\bin\x64"

import pyagrum as gum
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from pgmpy import config
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

warnings.filterwarnings("ignore")

# just setting up a basic network for fun
conn = [("A", "B"), ("A", "C")]
net = BayesianNetwork(conn)

bn = gum.BayesNet()
A_n = bn.add(gum.LabelizedVariable("A", "A", 2))
B_n = bn.add(gum.LabelizedVariable("B", "B", 2))
C_n = bn.add(gum.LabelizedVariable("C", "C", 2))
bn.addArc(A_n, B_n)
bn.addArc(A_n, C_n)

# wrapping some r stuff here
ro.r("""
library(bnlearn)
fit_r <- function(df) {
    df[] <- lapply(df, factor)
    net <- model2network("[A][B|A][C|A]")
    fit <- bn.fit(net, df, method="mle")
    return(TRUE)
}
""")
fit_r = ro.globalenv["fit_r"]


def make_fake_data(sz):
    np.random.seed(42)
    return pd.DataFrame(
        {"A": np.random.randint(0, 2, sz), "B": np.random.randint(0, 2, sz), "C": np.random.randint(0, 2, sz)}
    )


# benchmark config stuff
test_sz = [10000, 100000, 500000, 1000000]
n_iter = 3
res = []

print("running benchmarks... checking sizes:", test_sz)

for sz in test_sz:
    print(f"handling {sz} samples...")
    df = make_fake_data(sz)

    t_np, t_torch, t_agrum, t_r = [], [], [], []

    for i in range(n_iter):
        # pgmpy numpy backend
        config.set_backend("numpy")
        t0 = time.perf_counter()
        net.fit(df, estimator=MaximumLikelihoodEstimator)
        t_np.append(time.perf_counter() - t0)

        # pgmpy pytorch backend
        config.set_backend("torch")
        t0 = time.perf_counter()
        net.fit(df, estimator=MaximumLikelihoodEstimator)
        t_torch.append(time.perf_counter() - t0)

        # agrum stuff
        t0 = time.perf_counter()
        learner = gum.BNLearner(df)
        learner.learnParameters(bn.dag())
        t_agrum.append(time.perf_counter() - t0)

        # r backend
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)
        t0 = time.perf_counter()
        fit_r(r_df)
        t_r.append(time.perf_counter() - t0)

    res.append({"Size": sz, "Library": "pgmpy (NumPy)", "MeanTime": np.mean(t_np), "StdErr": sem(t_np)})
    res.append({"Size": sz, "Library": "pgmpy (PyTorch)", "MeanTime": np.mean(t_torch), "StdErr": sem(t_torch)})
    res.append({"Size": sz, "Library": "pyagrum (C++)", "MeanTime": np.mean(t_agrum), "StdErr": sem(t_agrum)})
    res.append({"Size": sz, "Library": "bnlearn (R)", "MeanTime": np.mean(t_r), "StdErr": sem(t_r)})


# ===========================
# now lets plot this stuff
# ===========================
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df_results = pd.DataFrame(res)
df_results["lbl"] = df_results["Size"].apply(lambda x: f"{x:,} Samples")
df_results["txt"] = df_results["MeanTime"].apply(lambda val: f"{val:.3f}s")

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
                    error_y=dict(type="data", array=row["StdErr"].values, visible=True),
                    marker=dict(color=color_map[libname]),
                    text=row["txt"].values,
                    textposition="outside",
                    showlegend=(r == 1 and c == 1),
                    name=libname,
                    legendgroup=libname,
                    hovertemplate="<b>%{x}</b><br>Time: %{y:.3f}s<extra></extra>",
                ),
                row=r,
                col=c,
            )

# making room for text on top
for i in range(1, 3):
    for j in range(1, 3):
        sub_idx = (i - 1) * 2 + j
        if sub_idx <= len(test_sz):
            sz = test_sz[sub_idx - 1]
            mx = df_results[df_results["Size"] == sz]["MeanTime"].max()
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

out_file = "runtime_comparison.html"
fig.write_html(out_file)

config.set_backend("numpy")  # reset back to numpy just in case
print("done! saved to", out_file)
