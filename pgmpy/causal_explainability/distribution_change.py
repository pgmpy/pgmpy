import networkx as nx
import numpy as np
from scipy import stats

from pgmpy.causal_explainability._base import _BaseAttribution
from pgmpy.causal_explainability._shapley import ShapleyEngine


def _kl_divergence_gaussian(mu0, std0, mu1, std1):
    """KL(N(mu1,std1^2) || N(mu0,std0^2))."""
    return np.log(std0 / std1) + (std1**2 + (mu1 - mu0) ** 2) / (2 * std0**2) - 0.5


def _kl_divergence_samples(p_samples, q_samples):
    """Estimate KL(Q || P) using KDE."""
    kde_p = stats.gaussian_kde(p_samples)
    kde_q = stats.gaussian_kde(q_samples)
    log_q = np.log(kde_q(q_samples) + 1e-300)
    log_p = np.log(kde_p(q_samples) + 1e-300)
    return np.mean(log_q - log_p)


class DistributionChangeAttribution(_BaseAttribution):
    """Attributes distributional shift in a target to mechanism changes.

    Parameters
    ----------
    divergence : callable(P_samples, Q_samples) -> float, optional
        Default: KL divergence (closed-form for LinearGaussian, KDE otherwise).
    """

    def __init__(self, divergence=None):
        self.divergence = divergence

    def _validate(self, model, data, target, **kwargs):
        pass  # Custom validation in attribute()

    def attribute(self, model, data=None, target=None, **kwargs):
        data_old = kwargs.get("data_old")
        data_new = kwargs.get("data_new")
        if data_old is None or data_new is None:
            raise ValueError("data_old and data_new are required.")
        if target is None:
            raise ValueError("target is required.")
        if target not in model.nodes():
            raise ValueError(f"Target '{target}' not in model.")
        return self._attribute(model, data, target, **kwargs)

    def _attribute(self, model, data, target, **kwargs):
        data_old = kwargs["data_old"]
        data_new = kwargs["data_new"]
        n_samples = kwargs.get("n_samples", 500)

        from pgmpy.models import LinearGaussianBayesianNetwork

        nodes = list(nx.topological_sort(model))

        if isinstance(model, LinearGaussianBayesianNetwork):
            cpds_old = self._fit_cpds(model, data_old)
            cpds_new = self._fit_cpds(model, data_new)

            def value_fn(coalition):
                active = {nodes[i] for i in coalition}
                hybrid_cpds = {}
                for node in nodes:
                    hybrid_cpds[node] = cpds_new[node] if node in active else cpds_old[node]
                mu_old, std_old = self._propagate_lgbn(nodes, cpds_old)
                mu_hyb, std_hyb = self._propagate_lgbn(nodes, hybrid_cpds)

                if self.divergence is not None:
                    rng = np.random.default_rng(42)
                    p = rng.normal(mu_old[target], std_old[target], n_samples)
                    q = rng.normal(mu_hyb[target], std_hyb[target], n_samples)
                    return self.divergence(p, q)

                if std_old[target] < 1e-10 and std_hyb[target] < 1e-10:
                    return 0.0 if abs(mu_old[target] - mu_hyb[target]) < 1e-10 else float("inf")
                if std_old[target] < 1e-10 or std_hyb[target] < 1e-10:
                    return float("inf")
                return _kl_divergence_gaussian(mu_old[target], std_old[target], mu_hyb[target], std_hyb[target])

            engine = ShapleyEngine(n_players=len(nodes), value_function=value_fn, method="auto")
            indexed = engine.compute()
            return {nodes[i]: v for i, v in indexed.items()}
        else:
            return self._attribute_general(model, data_old, data_new, target, nodes, n_samples)

    def _fit_cpds(self, model, data):
        from pgmpy.factors.continuous import LinearGaussianCPD

        cpds = {}
        for node in model.nodes():
            parents = list(model.get_parents(node))
            if not parents:
                cpds[node] = LinearGaussianCPD(node, beta=[data[node].mean()], std=max(data[node].std(), 1e-10))
            else:
                X = np.column_stack([np.ones(len(data))] + [data[p].values for p in parents])
                y = data[node].values
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                std = max(np.std(residuals), 1e-10)
                cpds[node] = LinearGaussianCPD(node, beta=beta.tolist(), std=std, evidence=parents)
        return cpds

    def _propagate_lgbn(self, nodes, cpds):
        mu, var = {}, {}
        for node in nodes:
            cpd = cpds[node]
            parents = cpd.evidence
            node_mu = cpd.beta[0]
            node_var = cpd.std**2
            for j, p in enumerate(parents):
                node_mu += cpd.beta[j + 1] * mu[p]
                node_var += cpd.beta[j + 1] ** 2 * var[p]
            mu[node] = node_mu
            var[node] = node_var
        return mu, {n: np.sqrt(max(v, 0)) for n, v in var.items()}

    def _attribute_general(self, model, data_old, data_new, target, nodes, n_samples):
        rng = np.random.default_rng(42)

        def value_fn(coalition):
            samples_old = model.simulate(n_samples=n_samples, seed=int(rng.integers(1e9)), show_progress=False)
            target_old = samples_old[target].values
            samples_new = model.simulate(n_samples=n_samples, seed=int(rng.integers(1e9)), show_progress=False)
            target_hyb = samples_new[target].values
            if self.divergence is not None:
                return self.divergence(target_old, target_hyb)
            return max(_kl_divergence_samples(target_old, target_hyb), 0)

        engine = ShapleyEngine(n_players=len(nodes), value_function=value_fn, method="auto")
        indexed = engine.compute()
        return {nodes[i]: v for i, v in indexed.items()}

    def mechanism_change_test(self, model, data_old, data_new, significance_level=0.05):
        """Test which mechanisms changed between two datasets.

        Returns dict[str, float] mapping node -> p_value.
        """
        import pandas as pd
        from scipy.stats import pearsonr

        combined = pd.concat([data_old.assign(_indicator=0), data_new.assign(_indicator=1)], ignore_index=True)
        pvals = {}
        for node in model.nodes():
            parents = list(model.get_parents(node))
            residuals_node = combined[node].values
            residuals_ind = combined["_indicator"].values.astype(float)

            if parents:
                X = np.column_stack([np.ones(len(combined))] + [combined[p].values for p in parents])
                beta_n = np.linalg.lstsq(X, residuals_node, rcond=None)[0]
                residuals_node = residuals_node - X @ beta_n
                beta_i = np.linalg.lstsq(X, residuals_ind, rcond=None)[0]
                residuals_ind = residuals_ind - X @ beta_i

            _, pval = pearsonr(residuals_node, residuals_ind)
            pvals[node] = pval
        return pvals
