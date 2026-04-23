import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from pgmpy.base import DAG
from pgmpy.ci_tests import get_ci_test
from pgmpy.global_vars import config
from pgmpy.metrics import _BaseUnsupervisedMetric


class FisherC(_BaseUnsupervisedMetric):
    """
    Returns a p-value for testing whether the given data is faithful to the
    model structure's constraints.

    Each missing edge in a model structure implies a CI statement. This test
    uses constructs implied CIs such that they are independent of each other,
    run statistical tests for each of them on the data, and finally combines
    them using the Fisher's method.

    Parameters
    ----------
    ci_test: str or callable
        The CI test to use for statistical testing. Can be a string name of any test
        in :mod:`pgmpy.ci_tests` (e.g. ``"chi_square"``, ``"pearsonr"``) or a callable.

    compute_rmsea: bool (default: False)
        While calculating Fisher C statistic if RMSEA value required should be
        included in method call as True. Returns a tuple of (p-value, rmsea) if
        True otherwise only the p-value.

    show_progress: bool (default: True)
        Whether to show the progress of testing.

    Returns
    -------
    float (default): The p-value for the fit of the model structure to the data. A low
        p-value (e.g. <0.05) represents that the model structure doesn't fit the
        data well. This is returned if the compute_rmsea parameter is False.

    tuple: A (float, float) tuple packing p-value and rmsea value is returned if RMSEA
            computation is necessary, i.e., compute_rmsea is True in the method call

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/cancer")
    >>> df = model.simulate(int(1e3))
    >>> fisher_c = FisherC(ci_test="chi_square", compute_rmsea=False)
    >>> fisher_c(X=df, causal_graph=model)
    0.7504
    """

    _tags = {
        "name": "fisher_c",
        "requires_true_graph": False,
        "requires_data": True,
        "lower_is_better": False,
        "supported_graph_types": (DAG,),
        "is_default": False,
    }

    def __init__(self, ci_test=None, compute_rmsea=False, show_progress=True):
        self.ci_test = ci_test
        self.compute_rmsea = compute_rmsea
        self.show_progress = show_progress

    def _evaluate(self, X, causal_graph):
        if len(causal_graph.latents) > 0:
            raise ValueError("This test can not be performed on models with latent variables.")

        cis = []
        ci_test = get_ci_test(test=self.ci_test, data=X)

        if self.show_progress and config.SHOW_PROGRESS:
            comb_iter = tqdm(
                combinations(causal_graph.nodes(), 2),
                total=math.comb(len(causal_graph.nodes()), 2),
            )
        else:
            comb_iter = combinations(causal_graph.nodes(), 2)

        for u, v in comb_iter:
            if not ((u in causal_graph[v]) or (v in causal_graph[u])):
                Z = set(causal_graph.predecessors(u)).union(causal_graph.predecessors(v))
                ci_test.is_independent(X=u, Y=v, Z=list(Z))
                cis.append([u, v, Z, ci_test.p_value_])
        cis = pd.DataFrame(cis, columns=["u", "v", "cond_vars", "p_value"])
        cis.loc[:, "p_value"] = cis.loc[:, "p_value"].clip(lower=1e-6)

        C = -2 * np.log(cis.loc[:, "p_value"]).sum()
        p_value = 1 - stats.chi2.cdf(C, df=2 * cis.shape[0])
        rmsea = np.nan

        if self.compute_rmsea:
            if len(X) != 1 and len(cis) != 0:
                rmsea = np.sqrt(max((C - 2 * len(cis)) / (2 * len(cis) * (len(X) - 1)), 0))
            return (p_value, rmsea)

        else:
            return p_value
