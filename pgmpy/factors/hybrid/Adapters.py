import numpy as np

from pgmpy.estimators import MaximumLikelihoodEstimator as MLE
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD


class TabularAdapter:
    """
    Adapter for fitting and representing tabular CPDs.
    """

    def __init__(self, variable, estimator=None, parents=None):
        self.variable = variable
        self.estimator = estimator
        self.parents = parents if parents is not None else []

    def fit(self, data):
        if self.estimator not in ("MLE", MLE):
            raise ValueError("For tabular tag, only MLE estimator is currently supported.")

        variable_states = sorted(data[self.variable].dropna().unique())
        if not self.parents:
            counts = data[self.variable].value_counts().reindex(variable_states, fill_value=0)
            probs = counts.values / counts.values.sum()
            values = [[prob] for prob in probs]
            self.fitted_cpd_ = TabularCPD(variable=self.variable, variable_card=len(variable_states), values=values)
            return self

        parent_states = [sorted(data[parent].dropna().unique()) for parent in self.parents]
        grouped = (
            data.groupby(self.parents + [self.variable], dropna=False)
            .size()
            .unstack(self.variable, fill_value=0)
            .reindex(columns=variable_states, fill_value=0)
        )
        grouped = grouped.T
        grouped = grouped / grouped.sum(axis=0).replace(0, 1)
        self.fitted_cpd_ = TabularCPD(
            variable=self.variable,
            variable_card=len(variable_states),
            values=grouped.values,
            evidence=self.parents,
            evidence_card=[len(states) for states in parent_states],
        )
        return self

    def __repr__(self):
        cpd = self.fitted_cpd_
        var_str = f"<FunctionalCPD(tabular) representing P({cpd.variable}:{cpd.variable_card}"

        evidence = cpd.variables[1:]
        evidence_card = cpd.cardinality[1:]
        if evidence:
            evidence_str = " | " + ", ".join([f"{var}:{card}" for var, card in zip(evidence, evidence_card)])
        else:
            evidence_str = ""
        return var_str + evidence_str + f") at {hex(id(self))}>"


class LinearGaussianAdapter:
    """
    Adapter for fitting and representing linear Gaussian CPDs.
    """

    def __init__(self, variable, estimator=None, parents=None):
        self.variable = variable
        self.estimator = estimator
        self.parents = parents if parents is not None else []

    def fit(self, data):
        if self.estimator not in ("MLE", "OLS", None):
            raise ValueError(f"For linear tag, MLE/OLS is supported. Got {self.estimator}")

        target_data = data[self.variable].values
        if not self.parents:
            mean = np.mean(target_data)
            std = np.std(target_data)
            beta = [mean]
        else:
            evidence_data = data[self.parents].values
            design_matrix = np.c_[np.ones(evidence_data.shape[0]), evidence_data]

            beta, residuals, _, _ = np.linalg.lstsq(design_matrix, target_data, rcond=None)
            if len(residuals) > 0:
                variance = residuals[0] / len(target_data)
            else:
                predictions = design_matrix @ beta
                variance = np.mean((target_data - predictions) ** 2)
            std = np.sqrt(variance)

        self.fitted_cpd_ = LinearGaussianCPD(variable=self.variable, beta=beta, std=std, evidence=self.parents)
        return self

    def __repr__(self):
        cpd = self.fitted_cpd_
        beta_str = f"{cpd.beta[0]:.3f}"
        for i, parent in enumerate(cpd.evidence):
            beta_str += f" + {cpd.beta[i + 1]:.3f}*{parent}"

        return (
            f"<FunctionalCPD(linear) representing P({cpd.variable} | {', '.join(cpd.evidence)}) "
            f"~ N({beta_str}, std={cpd.std:.3f}) at {hex(id(self))}>"
        )
