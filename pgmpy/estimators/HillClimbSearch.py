#!/usr/bin/env python
from itertools import permutations
from collections import deque

import networkx as nx
from tqdm import trange

from pgmpy.estimators import StructureEstimator, K2Score, ScoreCache
from pgmpy.base import DAG
from pgmpy.global_vars import SHOW_PROGRESS


class HillClimbSearch(StructureEstimator):
    def __init__(self, data, scoring_method=None, use_cache=True, **kwargs):
        """
        Class for heuristic hill climb searches for DAGs, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        scoring_method: Instance of a `StructureScore`-subclass (`K2Score` is used as default)
            An instance of `K2Score`, `BDeuScore`, or `BicScore`.
            This score is optimized during structure estimation by the `estimate`-method.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.

        use_caching: boolean
            If True, uses caching of score for faster computation.
            Note: Caching only works for scoring methods which are decomposible. Can
            give wrong results in case of custom scoring methods.

        References
        ----------
        Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.4.3 (page 811ff)
        """
        if scoring_method is not None:
            if use_cache:
                self.scoring_method = ScoreCache.ScoreCache(scoring_method, data)
            else:
                self.scoring_method = scoring_method
        else:
            self.scoring_method = ScoreCache.ScoreCache(K2Score(data, **kwargs), data)

        super(HillClimbSearch, self).__init__(data, **kwargs)

    def _legal_operations(self, model, tabu_list, max_indegree, black_list, white_list):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        local_score = self.scoring_method.local_score
        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )

        for (X, Y) in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                    (operation not in tabu_list)
                    and ((X, Y) not in black_list)
                    and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = local_score(Y, new_parents) - local_score(
                            Y, old_parents
                        )
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for (X, Y) in model.edges():
            operation = ("-", (X, Y))
            if operation not in tabu_list:
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)
                score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for (X, Y) in model.edges():
            # Check if flipping creates any cycles
            if any(map(lambda path: len(path) > 1, nx.all_simple_paths(model, X, Y))):
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                    and ((Y, X) not in black_list)
                    and ((Y, X) in white_list)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                            local_score(X, new_X_parents)
                            + local_score(Y, new_Y_parents)
                            - local_score(X, old_X_parents)
                            - local_score(Y, old_Y_parents)
                        )
                        yield (operation, score_delta)

    def estimate(
        self,
        start=None,
        tabu_length=0,
        max_indegree=None,
        black_list=None,
        white_list=None,
        epsilon=1e-4,
        max_iter=1e6,
        show_progress=True,
    ):
        """
        Performs local hill climb search to estimates the `DAG` structure
        that has optimal score, according to the scoring method supplied in the constructor.
        Starts at model `start` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no parametrization.

        Parameters
        ----------
        start: DAG instance
            The starting point for the local search. By default a completely disconnected network is used.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.
        black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None
        white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.

        max_iter: int (default: 1e6)
            The maximum number of iterations allowed. Returns the learned model when the
            number of iterations is greater than `max_iter`.

        Returns
        -------
        model: `DAG` instance
            A `DAG` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> est = HillClimbSearch(data, scoring_method=BicScore(data))
        >>> best_model = est.estimate()
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        [('B', 'J'), ('A', 'J')]
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        [('J', 'A'), ('B', 'J')]
        """

        # Step 1: Initial checks for arguments
        if start is None:
            start = DAG()
            start.add_nodes_from(self.variables)
        elif not isinstance(start, DAG) or not set(start.nodes()) == set(
            self.variables
        ):
            raise ValueError(
                "'start' should be a DAG with the same variables as the data set, or 'None'."
            )

        black_list = set() if black_list is None else set(black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if white_list is None
            else set(white_list)
        )
        if max_indegree is None:
            max_indegree = float("inf")

        # Step 2: Initialize variables
        tabu_list = deque(maxlen=tabu_length)
        current_model = start

        if show_progress and SHOW_PROGRESS:
            iteration = trange(int(max_iter))
        else:
            iteration = range(int(max_iter))

        # Step 3: For each iteration, find the best scoring operation and
        #         do that to the current model.
        for _ in iteration:
            best_operation, best_score_delta = max(
                self._legal_operations(
                    current_model, tabu_list, max_indegree, black_list, white_list
                ),
                key=lambda t: t[1],
            )

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == "+":
                current_model.add_edge(*best_operation[1])
                tabu_list.append(("-", best_operation[1]))
            elif best_operation[0] == "-":
                current_model.remove_edge(*best_operation[1])
                tabu_list.append(("+", best_operation[1]))
            elif best_operation[0] == "flip":
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list.append(best_operation)

        # Step 4: Return if no more improvements or maximum iterations reached.
        return current_model
