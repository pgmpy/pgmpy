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
        """
        Computes the Root Mean Square Error of Approximation given data,
        a test_statistic(X^2 / chi_squared) and degrees of freedom. This statistic
        penalizes models based on sample size. An acceptable value is generally <0.10
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
        """
        if len(data) != 1 and deg_of_freedom != 0:
            return np.sqrt(
                max(
                    (test_statistic - deg_of_freedom) / (deg_of_freedom * (len(data) - 1)),
                    0,
                )
            )
