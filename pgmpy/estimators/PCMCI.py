#!/usr/bin/env python

from itertools import chain, combinations, permutations
import copy

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.CITests import get_ci_test
from pgmpy.global_vars import logger


class PCMCI(StructureEstimator, TimeSeriesDAG):
    """Class for constraint-based estimation of time series causal graphs using the PCMCI algorithm.

    PCMCI is a two-step procedure that first applies a variant of the PC algorithm to
    identify the potential causal parents and then uses the Momentary Conditional Independence test to determine causal
    links while accounting for the time-lagged confounders and autocorrelation effects.

    Parameters
    ----------
    data : pandas.DataFrame
        The time series data where each column represents one variable and each row represents
        one time point.
    independencies : Independencies, optional
        Independencies object specifying the conditional independencies in the data.

    References
    ----------
    [1] Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D. (2019).
        Detecting and quantifying causal associations in large nonlinear time series datasets.
        Science Advances, 5(11), eaau4996.
    [2] Runge, J. (2018). Causal network reconstruction from time series: From theoretical
        assumptions to practical estimation. Chaos: An Interdisciplinary Journal of Nonlinear
        Science, 28(7), 075310.
    """

    def __init__(self, data=None, independencies=None, **kwargs):
        """Initialize the PCMCI estimator.

        Parameters
        ----------
        data : pandas.DataFrame
            The time series data where each column represents one variable and each row represents
            one time point.
        independencies : Independencies, optional
            Independencies object specifying the conditional independencies in the data.
        """
        self.data = data
        self.independencies = independencies

        TimeSeriesDAG.__init__(self)
        StructureEstimator.__init__(
            self, data=data, independencies=independencies, **kwargs
        )

    def estimate(
        self,
        ci_test="pearsonr",
        significance_level=0.05,
        show_progress=True,
        n_jobs=-1,
        max_cond_vars=5,
        max_time_lag=3,
        return_type="ts_dag",
        **kwargs,
    ):
        """
        Estimate a time series causal graph from the given dataset using the PCMCI algorithm.

        Parameters
        ----------
        ci_test : str or callable
            The conditional independence test to use. If string, should be one of:
            "pearsonr", "chi_square", "g_sq", etc.

        max_time_lag : int
            Maximum time lag to consider for causal relationships.

        significance_level : float, default=0.05
            Significance level for the conditional independence test.

        max_cond_vars : int, default=5
            Maximum number of conditioning variables to consider for the conditional independence test.

        show_progress : bool, default=True
            Whether to show a progress bar during the estimation process.

        n_jobs : int
            Number of jobs to run in parallel. Default is -1, which means using all processors.
            If 1, no parallel computing is used.
            If -1, all processors are used.

        return_type : str (one of "ts_dag", "summary_graph", "skeleton")
            The type of structure to return.

        Returns
        -------
        Estimated Model : TimeSeriesDAG or tuple
            The estimated time series causal graph or skeleton, depending on return_type.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import PCMCI
        >>> np.random.seed(42)
        >>> # Generate simple AR process: X causes Y with lag 1
        >>> T = 1000
        >>> data = pd.DataFrame(np.random.randn(T, 2), columns=['X', 'Y'])
        >>> for t in range(1, T):
        ...     data.loc[t, 'Y'] += 0.5 * data.loc[t-1, 'X'] + 0.5 * data.loc[t-1, 'Y'] + 0.1 * np.random.randn()
        >>> pcmci = PCMCI(data)
        >>> ts_dag = pcmci.estimate(max_time_lag=2)
        >>> print(ts_dag.edges())
        [(('X', 1), ('Y', 0)), (('Y', 1), ('Y', 0))]
        """
        if self.data is None:
            raise ValueError("Data must be provided to estimate the model")

        # Get the appropriate CI test
        ci_test_func = get_ci_test(
            ci_test, full=True, data=self.data, independencies=self.independencies
        )

        # Run the PC algorithm to find the skeleton with time-lagged variables
        skeleton, separating_sets = self._build_time_series_skeleton(
            ci_test=ci_test_func,
            max_time_lag=max_time_lag,
            significance_level=significance_level,
            show_progress=show_progress,
            n_jobs=n_jobs,
            max_cond_vars=max_cond_vars,
            **kwargs,
        )

        if return_type == "skeleton":
            return skeleton, separating_sets

        # Orient the edges based on the time order
        ts_dag = self._orient_time_series_edges(
            skeleton,
            separating_sets,
            max_time_lag=max_time_lag,
        )

        # Run the MCI tests to further refine the causal links
        ts_dag = self._run_mci_tests(
            ts_dag,
            ci_test=ci_test_func,
            significance_level=significance_level,
            show_progress=show_progress,
            n_jobs=n_jobs,
            max_cond_vars=max_cond_vars,
            **kwargs,
        )

        if return_type == "ts_dag":
            return ts_dag
        elif return_type == "summary_graph":
            # Implementation for summary graph would go here
            raise NotImplementedError("Summary graph is not yet implemented")
        else:
            raise ValueError(
                "Invalid return_type. Must be one of 'ts_dag', 'summary_graph', or 'skeleton'."
            )

    def _build_time_series_skeleton(
        self,
        ci_test,
        max_time_lag=3,
        significance_level=0.05,
        show_progress=True,
        n_jobs=-1,
        max_cond_vars=5,
        **kwargs,
    ):
        """
        Build a skeleton graph for time series data using PC algorithm adapted for time series.
        This is the first phase of PCMCI which identifies potential parent variables.

        Parameters
        ----------
        ci_test : callable
            The conditional independence test to use.

        max_time_lag : int
            Maximum time lag to consider for causal relationships.

        significance_level : float
            Significance level for the conditional independence test.
            Default is 0.05.

        max_cond_vars : int
            Maximum number of conditioning variables to consider for the conditional independence test.
            Default is 5.

        Returns
        -------
        skeleton : networkx.Graph
            The skeleton graph with potential causal links.

        separating_sets : dict
            The separating sets used for the conditional independence tests.
        """
        # Initialize the structures
        cond_set_size = 0
        separating_sets = {}

        # Get the list of variables from the data columns
        vars = list(self.data.columns)

        # Create all possible time-lagged variables
        time_lagged_variables = []
        for var in vars:
            time_lagged_variables.append((var, 0))  # Contemporary
            for lag in range(1, max_time_lag + 1):
                time_lagged_variables.append((var, lag))  # Lagged

        # Create the initial skeleton graph with all time-lagged variables as nodes
        graph = nx.Graph()
        graph.add_nodes_from(time_lagged_variables)

        # Add edges between variables according to the temporal constraints
        # - Variables at current time can be connected to all past variables and other current variables
        # - Variables at past times can only be connected to variables at the same time
        for (var1, lag1), (var2, lag2) in combinations(time_lagged_variables, 2):
            # Skip the self loops
            if var1 == var2 and lag1 == lag2:
                continue

            # Apply the temporal constraints
            if lag1 > 0 and lag2 > 0:
                # Both are past variables, only connect if same lag
                if lag1 == lag2:
                    graph.add_edge((var1, lag1), (var2, lag2))
            else:
                # At least one is contemporary, add edge
                graph.add_edge((var1, lag1), (var2, lag2))

        # Create lagged data for CI tests
        lagged_data = self._create_lagged_data(self.data, max_time_lag)

        # Show the initial progress bar
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=min(max_cond_vars, graph.number_of_nodes() - 2))
            pbar.set_description("Working on conditional variables")

        # Run the PC stable algorithm
        while cond_set_size <= max_cond_vars:
            edges_before = graph.number_of_edges()

            # Find nodes with enough neighbors for this conditional set size
            nodes_to_check = [
                node
                for node in time_lagged_variables
                if len(list(graph.neighbors(node))) > cond_set_size
            ]

            if not nodes_to_check:
                break  # No more nodes with enough neighbors

            # Run PC algorithm for current cond_set_size
            self._run_pc_iteration(
                graph,
                nodes_to_check,
                lagged_data,
                ci_test,
                separating_sets,
                cond_set_size,
                significance_level,
                n_jobs=n_jobs,
                **kwargs,
            )

            edges_after = graph.number_of_edges()

            # If no edges were removed, increase cond_set_size
            if edges_before == edges_after:
                cond_set_size += 1
                if show_progress and config.SHOW_PROGRESS:
                    pbar.update(1)
                    pbar.set_description(
                        f"Working on conditional variables: {cond_set_size}/{max_cond_vars}"
                    )

            # If max_cond_vars reached, break
            if cond_set_size > max_cond_vars:
                logger.info(
                    f"Maximum number of conditional variables {max_cond_vars} reached. Stopping the search."
                )
                break

        if show_progress and config.SHOW_PROGRESS:
            pbar.close()

        return graph, separating_sets

    def _run_pc_iteration(
        self,
        graph,
        nodes,
        lagged_data,
        ci_test,
        separating_sets,
        cond_set_size,
        significance_level,
        n_jobs=1,
        **kwargs,
    ):
        """
        Run one iteration of the PC algorithm for time series data for a specific conditional set size.

        Parameters
        ----------
        graph : networkx.Graph
            The current skeleton graph.
        nodes : list
            List of nodes to check.
        lagged_data : pandas.DataFrame
            DataFrame with lagged variables.
        ci_test : callable
            Conditional independence test function.
        separating_sets : dict
            Dictionary to store separating sets.
        cond_set_size : int
            Size of conditioning sets to check.
        significance_level : float
            Alpha level for CI tests.
        n_jobs : int
            Number of parallel jobs.
        """
        # Get all edges to test
        edges_to_test = []
        for node in nodes:
            neighbors = list(graph.neighbors(node))
            for neigh in neighbors:
                # Add each edge only once
                if (node, neigh) not in edges_to_test and (
                    neigh,
                    node,
                ) not in edges_to_test:
                    edges_to_test.append((node, neigh))

        # For each edge, test conditional independence with all possible conditioning sets
        if n_jobs != 1 and len(edges_to_test) > 0:
            # Parallel implementation
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._test_edge_independence)(
                    u,
                    v,
                    graph,
                    lagged_data,
                    ci_test,
                    cond_set_size,
                    significance_level,
                    **kwargs,
                )
                for u, v in edges_to_test
            )

            # Process results
            for (u, v), (is_independent, sep_set) in zip(edges_to_test, results):
                if is_independent:
                    separating_sets[frozenset((u, v))] = sep_set
                    if graph.has_edge(u, v):  # Check if edge still exists
                        graph.remove_edge(u, v)
        else:
            # Sequential implementation
            for u, v in edges_to_test:
                is_independent, sep_set = self._test_edge_independence(
                    u,
                    v,
                    graph,
                    lagged_data,
                    ci_test,
                    cond_set_size,
                    significance_level,
                    **kwargs,
                )
                if is_independent:
                    separating_sets[frozenset((u, v))] = sep_set
                    if graph.has_edge(u, v):  # Check if edge still exists
                        graph.remove_edge(u, v)

    def _test_edge_independence(
        self,
        u,
        v,
        graph,
        lagged_data,
        ci_test,
        cond_set_size,
        significance_level,
        **kwargs,
    ):
        """
        Test if edge (u, v) should be removed based on conditional independence tests.

        Parameters
        ----------
        u, v : tuple
            The nodes to test, each is (variable, lag).
        graph : networkx.Graph
            The current skeleton graph.
        lagged_data : pandas.DataFrame
            DataFrame with lagged variables.
        ci_test : callable
            Conditional independence test function.
        cond_set_size : int
            Size of conditioning sets to check.
        significance_level : float
            Alpha level for CI tests.

        Returns
        -------
        is_independent : bool
            True if u and v are conditionally independent.
        sep_set : list
            The separating set that renders u and v conditionally independent.
        """
        # Find potential separating sets
        potential_sepsets = self._get_potential_sepsets(u, v, graph, cond_set_size)

        for sep_set in potential_sepsets:
            # Test conditional independence using the time-series aware CI test wrapper
            is_independent = self._ci_test_wrapper(
                u, v, sep_set, lagged_data, ci_test, significance_level, **kwargs
            )

            if is_independent:
                return True, sep_set

        return False, None

    def _get_potential_sepsets(self, u, v, graph, cond_set_size):
        """
        Get potential separating sets for nodes u and v, respecting temporal constraints.

        Parameters
        ----------
        u, v : tuple
            The nodes to test, each is (variable, lag).
        graph : networkx.Graph
            The current skeleton graph.
        cond_set_size : int
            Size of conditioning sets to check.

        Returns
        -------
        sepsets : list
            List of potential separating sets.
        """
        _, u_lag = u
        _, v_lag = v

        # Get neighbors of u and v
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))

        # Remove v from neighbors of u and vice versa
        neighbors_u.discard(v)
        neighbors_v.discard(u)

        # Apply temporal constraints - only consider nodes that respect causality
        min_lag = min(u_lag, v_lag)

        valid_neighbors_u = set()
        for neigh in neighbors_u:
            _, neigh_lag = neigh
            if neigh_lag >= min_lag:  # Only consider nodes at the same time or later
                valid_neighbors_u.add(neigh)

        valid_neighbors_v = set()
        for neigh in neighbors_v:
            _, neigh_lag = neigh
            if neigh_lag >= min_lag:  # Only consider nodes at the same time or later
                valid_neighbors_v.add(neigh)

        # Combine neighbors for potential separating sets
        all_valid_neighbors = valid_neighbors_u | valid_neighbors_v

        # Generate combinations of valid neighbors of the correct size
        return list(combinations(all_valid_neighbors, cond_set_size))

    def _ci_test_wrapper(
        self, u, v, sep_set, lagged_data, ci_test, significance_level, **kwargs
    ):
        """
        Wrapper function to handle the mismatch between tuple-based variable names
        and what the CI test expects.

        Parameters
        ----------
        u, v : tuple
            The nodes to test, each is (variable, lag).
        sep_set : list
            List of conditioning variables (each is a tuple).
        lagged_data : pandas.DataFrame
            DataFrame with tuple column names.
        ci_test : callable
            The CI test function.
        significance_level : float
            Alpha level for CI tests.

        Returns
        -------
        bool
            True if u and v are conditionally independent given sep_set.
        """
        # Convert tuple column names to string column names for the CI test
        col_mapping = {}
        temp_data = lagged_data.copy()

        # Create string representations of tuple column names
        for col in lagged_data.columns:
            if isinstance(col, tuple):
                var_name, lag = col
                string_col = f"{var_name}_lag_{lag}"
                col_mapping[col] = string_col
            else:
                col_mapping[col] = str(col)

        # Rename columns in the temporary dataframe
        temp_data.columns = [col_mapping[col] for col in temp_data.columns]

        # Convert node tuples to their string representations
        u_str = col_mapping[u]
        v_str = col_mapping[v]
        sep_set_str = [col_mapping[s] for s in sep_set]

        # Call the CI test with string column names
        return ci_test(
            u_str,
            v_str,
            sep_set_str,
            data=temp_data,
            significance_level=significance_level,
            **kwargs,
        )

    def _orient_time_series_edges(self, skeleton, separating_sets, max_time_lag=3):
        """
        Orient edges in the skeleton based on time ordering and v-structures.

        Parameters
        ----------
        skeleton : networkx.Graph
            The undirected skeleton graph with time-lagged variables.

        separating_sets : dict
            Dictionary of separating sets for each pair of non-adjacent nodes.

        max_time_lag : int
            Maximum time lag considered.

        Returns
        -------
        ts_dag : TimeSeriesDAG
            A directed acyclic graph for time series data.
        """
        # Initialize a directed graph
        ts_dag = TimeSeriesDAG()
        ts_dag.add_nodes_from(skeleton.nodes())

        # First, orient edges based on time ordering (temporal constraint)
        for u, v in skeleton.edges():
            _, u_lag = u
            _, v_lag = v

            # Same time lag (contemporaneous) - leave unoriented for now
            if u_lag == v_lag:
                # For contemporaneous variables, we don't know the direction yet
                # Store as bidirectional edge for now (will be resolved later)
                ts_dag.add_edge(u, v)
                ts_dag.add_edge(v, u)
            else:
                # Different time lags - orient from higher lag (past) to lower lag (more recent/future)
                if u_lag > v_lag:  # u is in the past relative to v
                    ts_dag.add_edge(u, v)
                else:  # v is in the past relative to u
                    ts_dag.add_edge(v, u)

        # Next, orient v-structures (colliders)
        node_pairs = list(permutations(skeleton.nodes(), 2))

        for X, Y in node_pairs:
            if not skeleton.has_edge(X, Y):  # X and Y are not adjacent
                _, X_lag = X
                _, Y_lag = Y

                # Find common neighbors (potential colliders)
                common_neighbors = set(skeleton.neighbors(X)) & set(
                    skeleton.neighbors(Y)
                )

                for Z in common_neighbors:
                    _, Z_lag = Z

                    # Check if Z is not in the separating set of X and Y
                    if frozenset((X, Y)) in separating_sets:
                        sep_set = separating_sets[frozenset((X, Y))]
                        if Z not in sep_set:
                            # Temporal constraint check - a variable can only be a collider if it's
                            # at the same time or more recent than both parents
                            if Z_lag <= X_lag and Z_lag <= Y_lag:
                                # Orient edges to form a v-structure
                                if ts_dag.has_edge(Z, X):
                                    ts_dag.remove_edge(Z, X)
                                if ts_dag.has_edge(Z, Y):
                                    ts_dag.remove_edge(Z, Y)
                                ts_dag.add_edge(X, Z)
                                ts_dag.add_edge(Y, Z)

        # Remove cycles within the same time slice (if any)
        self._remove_cycles_within_time_slice(ts_dag)

        return ts_dag

    def _remove_cycles_within_time_slice(self, ts_dag):
        """
        Remove cycles within the same time slice by keeping only one direction.

        Parameters
        ----------
        ts_dag : TimeSeriesDAG
            The time series directed graph, possibly with cycles.
        """
        # Group nodes by time lag
        nodes_by_lag = {}
        for node in ts_dag.nodes():
            _, lag = node
            if lag not in nodes_by_lag:
                nodes_by_lag[lag] = []
            nodes_by_lag[lag].append(node)

        # Check for cycles within each time slice
        for lag, nodes in nodes_by_lag.items():
            subgraph = ts_dag.subgraph(nodes)

            # Find cycles within the time slice
            try:
                cycles = list(nx.simple_cycles(subgraph))

                # Remove one edge from each cycle
                for cycle in cycles:
                    if len(cycle) > 1:
                        # Remove the last edge in the cycle
                        ts_dag.remove_edge(cycle[-1], cycle[0])
            except nx.NetworkXNoCycle:
                # No cycles in this time slice
                pass

    def _run_mci_tests(
        self,
        ts_dag,
        ci_test,
        significance_level=0.05,
        max_cond_vars=5,
        n_jobs=-1,
        show_progress=True,
        **kwargs,
    ):
        """
        Run Momentary Conditional Independence tests to refine the causal links.

        The MCI test conditions on parents of both the source and the target variables
        to control for common causes and indirect paths.

        Parameters
        ----------
        ts_dag : TimeSeriesDAG
            The time series directed acyclic graph with potential causal links.

        ci_test : callable
            The conditional independence test to use.

        significance_level : float
            The significance level for the conditional independence test.

        max_cond_vars : int
            The maximum number of conditioning variables to consider for the conditional independence test.

        n_jobs : int
            The number of jobs to run in parallel.

        show_progress : bool
            Whether to show a progress bar during the estimation process.

        Returns
        -------
        ts_dag : TimeSeriesDAG
            The refined time series directed acyclic graph with causal links.
        """
        # Create lagged data for MCI tests
        max_time_lag = max([lag for _, lag in ts_dag.nodes()])
        lagged_data = self._create_lagged_data(self.data, max_time_lag)

        # Get all the edges to test
        edges_to_test = list(ts_dag.edges())

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=len(edges_to_test), desc="Running MCI tests")

        # Get copy of the DAG to avoid modifying it during iteration
        result_dag = ts_dag.copy()

        # Run MCI tests in parallel
        if n_jobs != 1 and len(edges_to_test) > 0:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._run_single_mci_test)(
                    ts_dag,
                    u,
                    v,
                    lagged_data,
                    ci_test,
                    significance_level,
                    max_cond_vars,
                    **kwargs,
                )
                for u, v in edges_to_test
            )

            # Remove the edges that fail the MCI test
            for edge, should_remove in zip(edges_to_test, results):
                if should_remove:
                    result_dag.remove_edge(*edge)
                if show_progress and config.SHOW_PROGRESS:
                    pbar.update(1)
        else:
            # Sequential implementation
            for u, v in edges_to_test:
                should_remove = self._run_single_mci_test(
                    ts_dag,
                    u,
                    v,
                    lagged_data,
                    ci_test,
                    significance_level,
                    max_cond_vars,
                    **kwargs,
                )
                if should_remove:
                    result_dag.remove_edge(u, v)
                if show_progress and config.SHOW_PROGRESS:
                    pbar.update(1)

        if show_progress and config.SHOW_PROGRESS:
            pbar.close()

        return result_dag

    def _run_single_mci_test(
        self,
        ts_dag,
        u,
        v,
        data,
        ci_test,
        significance_level,
        max_cond_vars,
        **kwargs,
    ):
        """
        Run a single Momentary Conditional Independence test for the edge (u, v).

        Parameters
        ----------
        ts_dag : TimeSeriesDAG
            The time series directed acyclic graph with potential causal links.

        u : tuple
            The source node of the edge.

        v : tuple
            The target node of the edge.

        data : pandas.DataFrame
            The lagged data for the MCI tests.

        ci_test : callable
            The conditional independence test to use.

        significance_level : float
            The significance level for the conditional independence test.

        max_cond_vars : int
            The maximum number of conditioning variables to consider for the conditional independence test.

        Returns
        -------
        should_remove : bool
            True if the edge should be removed, False otherwise.
        """
        # Get the parents of u and v in the current ts_dag, excluding each other
        parents_u = set(ts_dag.get_parents(u))
        parents_v = set(ts_dag.get_parents(v))

        # Exclude each other from parents
        parents_u.discard(v)
        parents_v.discard(u)

        # Create a list of all potential conditioning variables
        cond_vars = list(parents_u | parents_v)

        # Check if there are too many conditioning variables
        if len(cond_vars) > max_cond_vars:
            # Prioritize more recent parents and those with stronger causal effects
            # This is simplified; in practice, a more sophisticated selection might be used
            cond_vars = sorted(
                cond_vars, key=lambda x: x[1]
            )  # Sort by lag (more recent first)
            cond_vars = cond_vars[:max_cond_vars]

        # Run the MCI test using the wrapper function
        is_independent = self._ci_test_wrapper(
            u, v, cond_vars, data, ci_test, significance_level, **kwargs
        )

        return is_independent

    def _create_lagged_data(self, data, max_lag):
        """Create a lagged version of the input data up to the maximum lag.

        Parameters
        ---------
        data : pd.DataFrame
            Original Time Series data with columns as variable names and
            rows as time steps.
        max_lag : int
            Maximum lag to include in the lagged data.

        Returns
        -------
        lagged_data : pandas.DataFrame
            A DataFrame where each column corresponds to a (variable, lag) pair and rows are aligned
            such that time t in the new DataFrame corresponds to variables at time t in original data.
            NaNs are dropped for the rows with insufficient lag history.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        if max_lag < 0:
            raise ValueError("max_lag must be non-negative")

        lagged_data = {}

        # Create lagged versions of each variable
        for lag in range(max_lag + 1):
            lagged_df = data.shift(lag).copy()
            var_lag_tuples = [(col, lag) for col in data.columns]
            lagged_df.columns = var_lag_tuples
            lagged_data.update(lagged_df.to_dict("series"))

        # Convert to DataFrame
        lagged_df = pd.DataFrame(lagged_data)

        # Drop rows with NaN values from shifting
        lagged_df.dropna(inplace=True)

        # Reset index for clean DataFrame
        lagged_df.reset_index(drop=True, inplace=True)

        return lagged_df

    def get_parents(self, node):
        """
        Get the parents of a node in the time series graph.

        Parameters
        ----------
        node : tuple
            The node for which to get the parents. Should be a tuple (variable, lag).

        Returns
        -------
        list
            A list of parents of the node.
        """
        return list(self.predecessors(node))
