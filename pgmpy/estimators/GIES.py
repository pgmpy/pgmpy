from itertools import combinations

import numpy as np
import pandas as pd

from pgmpy.base import IPDAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.ScoreCache import ScoreCache
from pgmpy.estimators.StructureScore import StructureScore, get_scoring_method


class GIES(StructureEstimator):
    """
    Greedy Interventional Equivalence Search (GIES) algorithm for learning causal structure
    from a combination of observational and interventional data.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame object containing the data. Must have a column named 'intervention_targets'
        that specifies which variables were intervened on for each sample. Use None for
        observational samples.
    scoring_method: str (default: 'bic-g')
        The scoring method to use. For continuous data, use 'bic-g' (Gaussian BIC).
        For discrete data, use 'bic-d' (Discrete BIC).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import GIES
    >>> # Create sample data with interventions
    >>> data = pd.DataFrame(np.random.randn(1000, 3), columns=['A', 'B', 'C'])
    >>> data['intervention_targets'] = [None] * 800 + ['A'] * 100 + ['B'] * 100
    >>> # Learn the structure
    >>> gies = GIES(data)
    >>> ipdag = gies.estimate()
    """

    def __init__(self, data, scoring_method="bic-g"):
        """
        Initialize the GIES estimator.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame object containing the data. Must have a column named 'intervention_targets'
            that specifies which variables were intervened on for each sample. Use None for
            observational samples.
        scoring_method: str (default: 'bic-g')
            The scoring method to use. For continuous data, use 'bic-g' (Gaussian BIC).
            For discrete data, use 'bic-d' (Discrete BIC).
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")

        if "intervention_targets" not in data.columns:
            raise ValueError("data must contain a column named 'intervention_targets'")

        self.data = data
        self.scoring_method = scoring_method

        # Create base scorer and score cache
        base_scorer, self.score_cache = get_scoring_method(
            scoring_method, data, use_cache=True
        )
        self.structure_score = self.score_cache

    def estimate(self, return_type="ipdag", max_iter=100, show_progress=True):
        """
        Estimate the I-PDAG using the GIES algorithm.

        Parameters
        ----------
        return_type: str (default: 'ipdag')
            The type of object to return. Can be 'ipdag' or 'dag'.
        max_iter: int (default: 100)
            Maximum number of iterations to run the algorithm.
        show_progress: bool (default: True)
            Whether to show progress during the algorithm execution.

        Returns
        -------
        IPDAG or DAG: The estimated causal structure.
        """
        # Initialize empty I-PDAG
        ipdag = IPDAG(
            intervention_targets=self.data["intervention_targets"].unique().tolist()
        )
        ipdag.add_nodes_from(self.data.columns.drop("intervention_targets"))

        # Forward phase
        for _ in range(max_iter):
            best_score = float("-inf")
            best_operation = None

            # Try adding edges
            for u, v in combinations(ipdag.nodes(), 2):
                if not ipdag.is_adjacent(u, v):
                    # Check if adding u->v is valid
                    if not ipdag.is_interventional_ancestor(v, u):
                        score = self._score_operation(ipdag, "add", u, v)
                        if score > best_score:
                            best_score = score
                            best_operation = ("add", u, v)

            # Try reversing edges
            for u, v in ipdag.directed_edges:
                if not ipdag.is_interventional_ancestor(u, v):
                    score = self._score_operation(ipdag, "reverse", u, v)
                    if score > best_score:
                        best_score = score
                        best_operation = ("reverse", u, v)

            if best_operation is None:
                break

            # Apply the best operation
            op, u, v = best_operation
            if op == "add":
                ipdag.add_edge(u, v)
            else:  # reverse
                ipdag.remove_edge(u, v)
                ipdag.add_edge(v, u)

        # Backward phase
        for _ in range(max_iter):
            best_score = float("-inf")
            best_operation = None

            # Try removing edges
            for u, v in ipdag.edges():
                score = self._score_operation(ipdag, "remove", u, v)
                if score > best_score:
                    best_score = score
                    best_operation = ("remove", u, v)

            if best_operation is None:
                break

            # Apply the best operation
            op, u, v = best_operation
            ipdag.remove_edge(u, v)

        if return_type == "ipdag":
            return ipdag
        elif return_type == "dag":
            return ipdag.to_dag()
        else:
            raise ValueError("return_type must be either 'ipdag' or 'dag'")

    def _score_operation(self, ipdag, operation, u, v):
        """
        Score a potential operation on the I-PDAG.

        Parameters
        ----------
        ipdag: IPDAG
            The current I-PDAG.
        operation: str
            The operation to score ('add', 'remove', or 'reverse').
        u, v: str
            The nodes involved in the operation.

        Returns
        -------
        float: The score of the operation.
        """
        # Create a copy of the I-PDAG
        ipdag_copy = ipdag.copy()

        # Apply the operation
        if operation == "add":
            ipdag_copy.add_edge(u, v)
        elif operation == "remove":
            if ipdag_copy.has_edge(u, v):
                ipdag_copy.remove_edge(u, v)
            else:
                return float("-inf")  # Invalid operation
        else:  # reverse
            if ipdag_copy.has_edge(u, v):
                ipdag_copy.remove_edge(u, v)
                ipdag_copy.add_edge(v, u)
            else:
                return float("-inf")  # Invalid operation

        # Calculate the score
        return self.structure_score.score(ipdag_copy)
