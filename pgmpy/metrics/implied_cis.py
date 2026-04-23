import math
from itertools import combinations

import pandas as pd
from tqdm import tqdm

from pgmpy.base import DAG
from pgmpy.ci_tests import get_ci_test
from pgmpy.global_vars import config
from pgmpy.metrics import _BaseUnsupervisedMetric


class ImpliedCIs(_BaseUnsupervisedMetric):
    """
    Tests the implied Conditional Independences (CI) of the DAG in the given data.

    Each missing edge in a model structure implies a CI statement. If the distribution of the data is faithful to the
    constraints of the model structure, these CI statements should hold in the data as well. This function runs
    statistical tests for each implied CI on the given data.

    Parameters
    ----------
    ci_test: str or callable
        The CI test to use for statistical testing. Can be a string name of any test
        in :mod:`pgmpy.ci_tests` (e.g. ``"chi_square"``, ``"pearsonr"``) or a callable.

    show_progress: bool (default: True)
        Whether to show the progress of testing.

    Returns
    -------
    pd.DataFrame: Returns a dataframe with each implied CI of the model and a p-value
        corresponding to it from the statistical test. A low p-value (e.g. <0.05)
        represents that the CI does not hold in the data.

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> from pgmpy.metrics import ImpliedCIs
    >>> model = load_model("bnlearn/cancer")
    >>> df = model.simulate(int(1e3))
    >>> implied_cis = ImpliedCIs(ci_test="chi_square", show_progress=False)
    >>> implied_cis.evaluate(X=df, causal_graph=model)
           u         v cond_vars   p-value
    0  Pollution    Smoker        []  0.189851
    1  Pollution      Xray  [Cancer]  0.404149
    2  Pollution  Dyspnoea  [Cancer]  0.613370
    3     Smoker      Xray  [Cancer]  0.352665
    4     Smoker  Dyspnoea  [Cancer]  1.000000
    5       Xray  Dyspnoea  [Cancer]  0.888619
    """

    _tags = {
        "name": "implied_cis",
        "requires_true_graph": False,
        "requires_data": True,
        "lower_is_better": None,
        "supported_graph_types": (DAG,),
        "is_default": False,
    }

    def __init__(self, ci_test=None, show_progress=True):
        self.ci_test = ci_test
        self.show_progress = show_progress

    def _evaluate(self, X, causal_graph):
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
                Z = list(causal_graph.minimal_dseparator(u, v))
                ci_test(X=u, Y=v, Z=Z)
                cis.append([u, v, Z, ci_test.p_value_])
        cis = pd.DataFrame(cis, columns=["u", "v", "cond_vars", "p-value"])
        return cis
