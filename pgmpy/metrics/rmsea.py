import numpy as np

from pgmpy.base import DAG


class RMSEA:
    """
    Computes the Root Mean Square Error of Approximation given inputs, a Fisher C statistic and degrees of freedom.
    """
    _tags = {
        "name": "RMSEA",
        "requires_true_graph": False,
        "requires_data": False,
        "lower_is_better": True,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
        "output_type": "scalar"
    }

    def compute_rmsea(self,test_statistic, deg_of_freedom, data):
        if len(data) <= 1 or deg_of_freedom <= 0:
            return np.nan
        return np.sqrt(max((test_statistic - deg_of_freedom) / (deg_of_freedom * (len(data) - 1)), 0))



