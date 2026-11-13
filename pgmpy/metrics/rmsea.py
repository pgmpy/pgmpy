import numpy as np

from pgmpy.base import DAG
from pgmpy.metrics import BaseUnsupervisedMetric


class RMSEA(BaseUnsupervisedMetric):
    _tags = {
        "name": "rmsea",
        "requires_data": True,
        "lower_is_better": True,
        "supported_graph_types": [DAG],
        "output_type": "scalar",
    }

    @staticmethod
    def compute_rmsea(test_statistic, deg_of_freedom, data):
        r"""
        Compute the Root Mean Square Error of Approximation given data,
        a test_statistic and degrees of freedom.

        .. math::

            \text{RMSEA} = \sqrt{\max\left(\frac{\chi^2 - \text{df}}{\text{df} \cdot (N - 1)}, 0\right)}

        where :math:`\chi^2` is the fit statistic, :math:`\text{df}` its degrees of freedom, and
        :math:`N` the sample size (``len(data)``). An acceptable value is generally <0.10
        and a good value is <0.08.

        Parameters
        -----
        test_statistic: float
        The chi-square distributed test statistic from the model fit.

        deg_of_freedom: int
        The degrees of freedom associated with the model

        data: pd.DataFrame
        The raw input observations or sample data matrix used to estimate
        the model parameters and fit statistics.

        Returns
        -----
        RMSEA

        Raises
        ------
        ValueError
            If ``deg_of_freedom`` is 0 or ``data`` has fewer than 2 rows.
        """
        if deg_of_freedom == 0:
            raise ValueError("RMSEA is undefined for a model with 0 degrees of freedom.")
        if len(data) < 2:
            raise ValueError(f"RMSEA requires at least 2 observations, but got {len(data)}.")

        return np.sqrt(
            max(
                (test_statistic - deg_of_freedom) / (deg_of_freedom * (len(data) - 1)),
                0,
            )
        )
