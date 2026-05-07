import pandas as pd
import plotly.graph_objects as go

# features we comparing, kept short so linter dont yell
def main():
    cats = [
        "Parameter Learning",
        "Parameter Learning",
        "Parameter Learning",
        "Parameter Learning",
        "Structure Learning",
        "Structure Learning",
        "Structure Learning",
        "Structure Learning",
        "Structure Learning",
        "Data Types",
        "Data Types",
        "Data Types",
        "Data Types",
        "Backend & API",
        "Backend & API",
        "Backend & API",
        "Backend & API",
        "Backend & API",
        "Inference",
        "Inference",
        "Inference",
    ]
    
    fts = [
        "Maximum Likelihood Estimation (MLE)",
        "Bayesian Estimation",
        "Expectation Maximization (EM)",
        "Dirichlet Prior",
        "Score-based (BIC, AIC, K2)",
        "Constraint-based (PC, IC)",
        "Hybrid (MMHC, H2PC)",
        "Exhaustive Search",
        "Tabu Search",
        "Discrete / Categorical Data",
        "Continuous Data (Gaussian)",
        "Mixed Data (Continuous + Discrete)",
        "Time Series / Dynamic BN",
        "Pandas DataFrame Support",
        "NumPy Backend",
        "PyTorch / GPU Support",
        "C/C++ Backend",
        "Incremental / Online Learning",
        "Variable Elimination",
        "Belief Propagation",
        "Sampling (MCMC, Gibbs, Likelihood Weighting)",
    ]
    
    pgmpy_ok = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
    
    bnlearn_ok = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    
    pyagrum_ok = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    
    pom_ok = [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    
    data = pd.DataFrame(
        {
            "Category": cats,
            "Feature": fts,
            "pgmpy": pgmpy_ok,
            "bnlearn (R)": bnlearn_ok,
            "pyagrum": pyagrum_ok,
            "pomegranate": pom_ok,
        }
    )
    
    libs = ["pgmpy", "bnlearn (R)", "pyagrum", "pomegranate"]
    hdr_cols = ["#1e293b", "#334155", "#2563eb", "#f97316", "#ef4444", "#22c55e"]
    
    # symbol mapping stuff
    sym = {True: "\u2713", False: "\u2717"}
    
    tbl_cols = [
        data["Category"].tolist(),
        data["Feature"].tolist(),
        data["pgmpy"].map(sym).tolist(),
        data["bnlearn (R)"].map(sym).tolist(),
        data["pyagrum"].map(sym).tolist(),
        data["pomegranate"].map(sym).tolist(),
    ]
    
    # figuring out cell colors for each row
    cell_cols = []
    for r in range(len(data)):
        row_data = []
        bg = "#f8fafc" if r % 2 == 0 else "#f1f5f9"
        row_data.append(bg)
        row_data.append(bg)
    
        for c in range(4):
            val = data.iloc[r][libs[c]]
            if val:
                row_data.append("rgba(34, 197, 94, 0.15)")
            else:
                row_data.append("rgba(239, 68, 68, 0.10)")
    
        cell_cols.append(row_data)
    
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Category", "Feature", "pgmpy", "bnlearn (R)", "pyagrum", "pomegranate"],
                    fill=dict(color=hdr_cols),
                    font=dict(color="white", size=13, family="Segoe UI, system-ui, sans-serif"),
                    align="center",
                    height=35,
                ),
                cells=dict(
                    values=tbl_cols,
                    # zipping colors to fix plotly column color thing
                    fill=dict(color=list(zip(*cell_cols))),
                    font=dict(size=12, family="Segoe UI, system-ui, sans-serif", color="#334155"),
                    align="center",
                    height=30,
                ),
                columnwidth=[140, 280, 80, 100, 80, 100],
            )
        ]
    )
    
    fig.update_layout(
        title=dict(
            text="Feature Comparison: pgmpy vs Other Packages",
            x=0.5,
            font=dict(size=18, color="#1e293b", family="Segoe UI, system-ui, sans-serif"),
        ),
        height=750,
        width=1200,
        margin=dict(t=60, b=20, l=150, r=150),
    )
    
    fig.write_html("feature_comparison.html")
    print("done exporting feature table!")
if __name__ == "__main__":
    main()
