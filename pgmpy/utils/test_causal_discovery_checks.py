import numpy as np
import pandas as pd


def check_causal_discovery_interface(estimator):
    """
    Check if a causal discovery estimator follows the pgmpy interface.
    """

    # Check required methods
    assert hasattr(estimator, "fit"), "Estimator must implement fit()"
    assert hasattr(estimator, "score"), "Estimator must implement score()"

    # Generate dummy dataset
    data = pd.DataFrame(
        np.random.randint(0, 2, size=(100, 5)), columns=[f"X{i}" for i in range(5)]
    )

    # Run fit
    estimator.fit(data)

    # Check required attributes
    assert hasattr(estimator, "causal_graph_"), "fit() must set causal_graph_"
    assert hasattr(estimator, "adjacency_matrix_"), "fit() must set adjacency_matrix_"
    assert hasattr(estimator, "n_features_in_"), "fit() must set n_features_in_"
    assert hasattr(estimator, "feature_names_in_"), "fit() must set feature_names_in_"

    # Check score method works
    estimator.score(data)
