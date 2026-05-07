import os
import time
import warnings
import pandas as pd
import numpy as np
from scipy.stats import sem

# add R path to windows env
os.environ['PATH'] += os.pathsep + r'C:\Program Files\R\R-4.6.0\bin\x64'

import pyagrum as gum
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy import config
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

warnings.filterwarnings("ignore")

# define network for testing
edges = [('A', 'B'), ('A', 'C')]
model_pgmpy = BayesianNetwork(edges)

bn = gum.BayesNet()
A_n = bn.add(gum.LabelizedVariable('A', 'A', 2))
B_n = bn.add(gum.LabelizedVariable('B', 'B', 2))
C_n = bn.add(gum.LabelizedVariable('C', 'C', 2))
bn.addArc(A_n, B_n)
bn.addArc(A_n, C_n)

# R function wrapper
ro.r('''
library(bnlearn)
learn_params_r <- function(df) {
    df[] <- lapply(df, factor)
    net <- model2network("[A][B|A][C|A]")
    fit <- bn.fit(net, df, method="mle")
    return(TRUE)
}
''')
learn_params_r = ro.globalenv['learn_params_r']

def gen_dummy_data(num_samps):
    np.random.seed(42)
    return pd.DataFrame({
        'A': np.random.randint(0, 2, num_samps),
        'B': np.random.randint(0, 2, num_samps),
        'C': np.random.randint(0, 2, num_samps)
    })

# bench config
test_sizes = [10000, 100000, 500000, 1000000]
iters = 3
final_res = []

print("starting benchmarks... sizes:", test_sizes)

for sz in test_sizes:
    print(f"processing {sz} samples...")
    dummy_df = gen_dummy_data(sz)
    
    times_np, times_torch, times_agrum, times_r = [], [], [], []

    for it in range(iters):
        # pgmpy numpy
        config.set_backend("numpy")
        start_t = time.perf_counter()
        model_pgmpy.fit(dummy_df, estimator=MaximumLikelihoodEstimator)
        times_np.append(time.perf_counter() - start_t)
        
        # pgmpy pytorch
        config.set_backend("torch")
        start_t = time.perf_counter()
        model_pgmpy.fit(dummy_df, estimator=MaximumLikelihoodEstimator)
        times_torch.append(time.perf_counter() - start_t)
        
        # agrum
        start_t = time.perf_counter()
        learner = gum.BNLearner(dummy_df)
        learner.learnParameters(bn.dag())
        times_agrum.append(time.perf_counter() - start_t)
        
        # r
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_dataframe = ro.conversion.py2rpy(dummy_df)
        start_t = time.perf_counter()
        learn_params_r(r_dataframe)
        times_r.append(time.perf_counter() - start_t)

    final_res.append({'Size': sz, 'Library': 'pgmpy (NumPy)', 'MeanTime': np.mean(times_np), 'StdErr': sem(times_np)})
    final_res.append({'Size': sz, 'Library': 'pgmpy (PyTorch)', 'MeanTime': np.mean(times_torch), 'StdErr': sem(times_torch)})
    final_res.append({'Size': sz, 'Library': 'pyagrum (C++)', 'MeanTime': np.mean(times_agrum), 'StdErr': sem(times_agrum)})
    final_res.append({'Size': sz, 'Library': 'bnlearn (R)', 'MeanTime': np.mean(times_r), 'StdErr': sem(times_r)})


# Plotting the results
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

res_df = pd.DataFrame(final_res)
res_df['lbl'] = res_df['Size'].apply(lambda x: f"{x:,} Samples")
res_df['txt'] = res_df['MeanTime'].apply(lambda val: f"{val:.3f}s")

colors = {
    'pgmpy (NumPy)': '#2563EB',
    'pgmpy (PyTorch)': '#F59E0B',
    'pyagrum (C++)': '#DC2626',
    'bnlearn (R)': '#10B981'
}

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f"{s:,} Samples" for s in test_sizes],
    specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]],
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

grid_pos = [(1, 1), (1, 2), (2, 1), (2, 2)]

for idx, sz in enumerate(test_sizes):
    r, c = grid_pos[idx]
    subset = res_df[res_df['Size'] == sz].sort_values('Library')
    
    for lib_name in ['pgmpy (NumPy)', 'pgmpy (PyTorch)', 'pyagrum (C++)', 'bnlearn (R)']:
        curr_data = subset[subset['Library'] == lib_name]
        
        if not curr_data.empty:
            fig.add_trace(
                go.Bar(
                    x=[lib_name],
                    y=curr_data['MeanTime'].values,
                    error_y=dict(
                        type='data',
                        array=curr_data['StdErr'].values,
                        visible=True
                    ),
                    marker=dict(color=colors[lib_name]),
                    text=curr_data['txt'].values,
                    textposition='outside',
                    showlegend=(r == 1 and c == 1),
                    name=lib_name,
                    legendgroup=lib_name,
                    hovertemplate='<b>%{x}</b><br>Time: %{y:.3f}s<extra></extra>'
                ),
                row=r, col=c
            )

# adjust ranges to fit text
for i in range(1, 3):
    for j in range(1, 3):
        sub_idx = (i - 1) * 2 + j
        if sub_idx <= len(test_sizes):
            sz = test_sizes[sub_idx - 1]
            max_v = res_df[res_df['Size'] == sz]['MeanTime'].max()
            fig.update_yaxes(title_text="Mean Time (s)", row=i, col=j, range=[0, max_v * 1.3])

fig.update_layout(
    title_text='Detailed Performance (Scaled per Size)',
    title_x=0.5,
    template='plotly_white',
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
        borderwidth=1
    ),
    margin=dict(t=100, b=80, l=80, r=50),
    font=dict(size=11)
)

out_file = "runtime_comparison.html"
fig.write_html(out_file)

config.set_backend("numpy")
print("done! saved to", out_file)