import pandas as pd
import plotly.graph_objects as go

# Feature matrix for comparison
features = [
    {'Category': 'Parameter Learning', 'Feature': 'Maximum Likelihood Estimation (MLE)',       'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': True},
    {'Category': 'Parameter Learning', 'Feature': 'Bayesian Estimation',                        'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Parameter Learning', 'Feature': 'Expectation Maximization (EM)',             'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Parameter Learning', 'Feature': 'Dirichlet Prior',                            'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Structure Learning', 'Feature': 'Score-based (BIC, AIC, K2)',               'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Structure Learning', 'Feature': 'Constraint-based (PC, IC)',                  'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Structure Learning', 'Feature': 'Hybrid (MMHC, H2PC)',                       'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Structure Learning', 'Feature': 'Exhaustive Search',                          'pgmpy': True,  'bnlearn (R)': False, 'pyagrum': False, 'pomegranate': False},
    {'Category': 'Structure Learning', 'Feature': 'Tabu Search',                                'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Data Types', 'Feature': 'Discrete / Categorical Data',                       'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': True},
    {'Category': 'Data Types', 'Feature': 'Continuous Data (Gaussian)',                        'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': True},
    {'Category': 'Data Types', 'Feature': 'Mixed Data (Continuous + Discrete)',                'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Data Types', 'Feature': 'Time Series / Dynamic BN',                          'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Backend & API', 'Feature': 'Pandas DataFrame Support',                       'pgmpy': True,  'bnlearn (R)': False, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Backend & API', 'Feature': 'NumPy Backend',                                  'pgmpy': True,  'bnlearn (R)': False, 'pyagrum': False, 'pomegranate': False},
    {'Category': 'Backend & API', 'Feature': 'PyTorch / GPU Support',                          'pgmpy': True,  'bnlearn (R)': False, 'pyagrum': False, 'pomegranate': True},
    {'Category': 'Backend & API', 'Feature': 'C/C++ Backend',                                  'pgmpy': False, 'bnlearn (R)': True,  'pyagrum': True,  'pomegranate': False},
    {'Category': 'Backend & API', 'Feature': 'Incremental / Online Learning',                  'pgmpy': True,  'bnlearn (R)': False, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Inference', 'Feature': 'Variable Elimination',                              'pgmpy': True,  'bnlearn (R)': False, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Inference', 'Feature': 'Belief Propagation',                                 'pgmpy': True,  'bnlearn (R)': False, 'pyagrum': True,  'pomegranate': False},
    {'Category': 'Inference', 'Feature': 'Sampling (MCMC, Gibbs, Likelihood Weighting)',       'pgmpy': True,  'bnlearn (R)': True, 'pyagrum': True,  'pomegranate': False},
]

df = pd.DataFrame(features)
packages = ['pgmpy', 'bnlearn (R)', 'pyagrum', 'pomegranate']

# prepare column data
col_data = [
    df['Category'].tolist(),
    df['Feature'].tolist(),
    df['pgmpy'].map({True: '✓', False: '✗'}).tolist(),
    df['bnlearn (R)'].map({True: '✓', False: '✗'}).tolist(),
    df['pyagrum'].map({True: '✓', False: '✗'}).tolist(),
    df['pomegranate'].map({True: '✓', False: '✗'}).tolist(),
]

# cell coloring
cell_colors = []
for i in range(len(df)):
    row_colors = []
    bg = '#f8fafc' if i % 2 == 0 else '#f1f5f9'
    row_colors.append(bg)
    row_colors.append(bg)
    for j in range(4):
        val = df.iloc[i][packages[j]]
        row_colors.append('rgba(34, 197, 94, 0.15)' if val else 'rgba(239, 68, 68, 0.10)')
    cell_colors.append(row_colors)

# create table
fig = go.Figure(data=[go.Table(
    header=dict(
        values=['Category', 'Feature', 'pgmpy', 'bnlearn (R)', 'pyagrum', 'pomegranate'],
        fill=dict(color='#1e293b'),
        font=dict(color='white', size=12, family='Arial'),
        align='center',
        height=30,
    ),
    cells=dict(
        values=col_data,
        fill=dict(color=list(zip(*cell_colors))),
        font=dict(size=11, family='Arial', color='#334155'),
        align='center',
        height=28,
    ),
    columnwidth=[130, 260, 70, 90, 70, 90],
)])

fig.update_layout(
    title=dict(
        text='Feature Comparison: pgmpy vs Other Packages',
        x=0.5,
        font=dict(size=16, color='#1e293b', family='Arial'),
    ),
    height=700,
    width=900,
    margin=dict(t=50, b=20, l=20, r=20),
)

fig.write_html('feature_comparison.html')
print('done. feature comparison saved to feature_comparison.html')
