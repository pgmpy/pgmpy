# This extension template provides instructions to add new causal discovery algorithms to pgmpy.
#
# Please follow the following steps:
# 1. Copy this file to `pgmpy/causal_discovery` and rename the file as `[MyCausalDiscoveryAlgo].py`
# 2. Go through the file and address all the TODOs.
# 3. Add an import statement in the `pgmpy/causal_discovery/__init__` file.
# 4. If you would like to contribute the dataset to pgmpy, please add tests for the algorithm in
#   `pgmpy/tests/test_causal_discovery/test_[MyCausalDiscoveryAlgo].py` file.


# TODO: Add any other necessary imports here.
import pandas as pd

from pgmpy.base import DAG  # noqa: F401
from pgmpy.causal_discovery._base import _BaseCausalDiscovery

# TODO: If the algorithm falls into a standard category (like constraint-based, score-based, etc.), mixin classes can be
# imported for additional functionality. For example:
#
# from pgmpy.casual_discovery import _ConstraintMixin


class MyCausalDiscoveryAlgo(_BaseCausalDiscovery):
    # TODO: If applicable, additionally inherit mixin classes here.
    """
    [One line description of the algorithm.]

    [Detailed description of the algorithm.]

    Parameters
    ----------
    hyperparam1: type, optional
        Description of hyperparam1.

    hyperparam2: type, optional
        Description of hyperparam2.

    Attributes
    ----------
    causal_graph_: []

    adjacency_matrix_: []

    n_features_in_: int
        The number of features in the dataset used to learn the causal graph.

    feature_names_in_: np.ndarray
        The feature names in the dataset used to learn the causal graph.

    Examples
    --------
    >>> from pgmpy.causal_discovery import MyCausalDiscoveryAlgo
    >>> ...

    References
    ----------
    .. [1] Citation1
    .. [2] Citation2
    """

    # TODO: Add all hyperparameters required for the algorithm in the init method.
    def __init__(self, hyperparam1=None, hyperparam2=None):

        # TODO: Assign all hyperparameters as attributes to the instance.
        self.hyperparam1 = hyperparam1
        self.hyperparam2 = hyperparam2

    def _fit(self, X: pd.DataFrame):

        # TODO: Add logic to learn the causal graph from the data X. Methods from mixin classes can be used here if
        #       applicable.

        # TODO: After learning the causal graph, assign the learned graph to self.causal_graph_ attribute. Can be an
        #       instance of pgmpy.base.DAG, PDAG, MAG, PAG, or ADMG, depending on the algorithm or the hyperparameters.
        self.causal_graph_ = None

        # TODO: Additionally, assign the adjacency matrix of the learned graph to self.adjacency_matrix_ attribute.
        self.adjacency_matrix_ = None

        # TODO: Assign the number of features and feature names from the data X to the respective attributes.
        self.n_features_in_ = None
        self.feature_names_in_ = None

        return self
