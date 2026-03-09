import numpy as np
import pandas as pd


def check_causal_discovery(estimator):
    """
    Checks whether a given causal discovery algorithm estimator complies with
    the pgmpy causal discovery interface.

    The algorithm must have:
    - a `fit` method that takes a dataset (pandas DataFrame) and returns the modified object.
    - a `score` method for evaluation.
    - `causal_graph_` attribute which is assigned after fitting.
    - `n_features_in_` and `feature_names_in_` attributes which should be populated after fitting.

    Parameters
    ----------
    estimator: An instance of the causal discovery algorithm class

    Raises
    ------
    AttributeError: If the estimator does not have a required method or attribute.
    RuntimeError: If the estimator fails when run on a dummy dataset.

    Returns
    -------
    None
    """

    name = estimator.__class__.__name__

    # 1. Check for required methods
    if not hasattr(estimator, "fit"):
        raise AttributeError(f"{name} must have a 'fit' method.")

    if not hasattr(estimator, "score"):
        raise AttributeError(f"{name} must have a 'score' method.")

    # 2. Check if it handles fit on proper data
    # We use a dummy dataset to test fit.
    X = pd.DataFrame(np.random.randn(15, 3), columns=["A", "B", "C"])
    try:
        estimator.fit(X)
    except Exception as e:
        raise RuntimeError(
            f"The `fit` method of {name} failed when run on a dummy dataset with error: {e}"
        )

    # 3. Check for specific properties that must be populated after run
    if not hasattr(estimator, "causal_graph_"):
        raise AttributeError(
            f"The 'fit' method of {name} must set the internal attribute 'causal_graph_'."
        )

    if not hasattr(estimator, "n_features_in_"):
        raise AttributeError(
            f"The 'fit' method of {name} must set the internal attribute 'n_features_in_'."
        )

    if not hasattr(estimator, "feature_names_in_"):
        raise AttributeError(
            f"The 'fit' method of {name} must set the internal attribute 'feature_names_in_'."
        )

    # 4. We can optionally verify that adjacency_matrix_ is populated
    if not hasattr(estimator, "adjacency_matrix_"):
        raise AttributeError(
            f"The 'fit' method of {name} must set the internal attribute 'adjacency_matrix_'."
        )

    return True
