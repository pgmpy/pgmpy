#!/usr/bin/env python

from itertools import chain, combinations, permutations

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.CITests import get_ci_test
from pgmpy.global_vars import logger


class PCMCI(StructureEstimator):
    """Class for constraint-based esitmation of time series causal graphs using the PCMCI algorithm.

    PCMCI is a two-step procedure that first applies a variant of the PC algorithm to
    identify the potential causal parents and then uses the Momentary Conditional test to determine causal
    links while accounting for the time-lagged cofounder and autocorrelation effects.

    Parameters
    ----------
    data : pandas.DataFrame
        The time series data where each column represents one variable and each row represents
        one time point.

    References
    ----------
    reference1
    reference2
    reference3
    etc..
    """

    def __init__(self, data=None, independencies=None, **kwargs):
        super(PCMCI, self).__init__(data=data, independencies=independencies, **kwargs)

    def estimate(
        self,
        ci_test="pearsonr",
        return_type="pdag",
        significance_level=0.05,
        show_progress=True,
        n_jobs=-1,
        max_cond_vars=5,
        max_time_lag=3,
        **kwargs,
    ):
        """
        Estimate a time series causal graph from the given dataset using the PCMCI algorithm.

        Parameters
        ----------
        ci_test : str or callable
            The conditional independence test to use If string, should be one of:
                "pearsonr", "chi_square", "g_sq", etc.

        max_time_lag : int
            Maximum time lag to consider for causal relationships.

        significance_level : float
            Significance level for the conditional independence test.
            Default is 0.05.

        max_cond_vars : int
            Maximum number of conditioning variables to consider for the conditional independence test.
            Default is 5.

        show_progress : bool
            Whether to show a progress bar during the estimation process.
            Default is True.

        n_jobs : int
            Number of jobs to run in parallel. Default is -1, which means using all processors.
            If 1, no parallel computing is used.
            If -1, all processors are used.

        return_type : str (one of "ts_dag", "summary_graph", "pdag", "skeleton")
            The type of structure to return.


        Returns
        -------
        Estimated Model : TimeSeriesGraph
            The estimated time series causal graph.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import PCMCI
        >>> np.random.seed(42)
        >>> # Generate simeple AR process: X causes Y with lag 1
        >>> data = pd.DataFrame(np.random.randn(T, 2), columns=['X', 'Y'])
        >>> for t in range(1, T):
        ...             data.loc[t, 'Y'] += 0.5 * data.loc[t-1, 'X'] + 0.5 * data.loc[t-1, 'Y] + 0.1 * np.random.randn()
        >>> pcmci = PCMCI(data)
        >>> ts_tag = pcmci.estimate(max_time_lag=2)
        >>> print(ts_tag.edges())
        [(('X', 1), ('Y', 0)), (('Y', 1), ('Y', 0))]
        """

        # get the approproate CI test
        ci_test = get_ci_test(
            ci_test, full=True, data=self.data, independencies=self.independencies
        )

        # Run the PC algorithm to find the skeleton with time-lagged variables
        skeleton, separating_sets = self._build_time_series_skeleton(
            ci_test=ci_test,
            max_time_lag=max_time_lag,
            significance_level=significance_level,
            show_progress=show_progress,
            n_jobs=n_jobs,
            max_cond_vars=max_cond_vars,
            **kwargs,
        )

        if return_type == "skeleton":
            return skeleton, separating_sets

        # orient the edges based on the time order
        # Past cant predict the future
        ts_dag = self._orient_time_series_edges(
            skeleton,
            separating_sets,
            max_time_lag=max_time_lag,
        )

        # Run the MCI tests to further refine the causal links
        ts_dag = self._run_mci_tests(
            ts_dag,
            ci_test=ci_test,
            significance_level=significance_level,
            show_progress=show_progress,
            n_jobs=n_jobs,
            max_cond_vars=max_cond_vars,
            **kwargs,
        )

        if return_type == "ts_dag":
            return ts_dag
        else:
            raise ValueError(
                "Invalid return_type. Must be one of 'ts_dag', 'summary_graph', or 'pdag'."
            )

    def _build_time_series_skeleton(
        self,
        ci_test="pearsonr",
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

        # initialize the structures
        cond_set_size = 0
        separating_sets = {}
        ci_test = get_ci_test(ci_test, full=True, data=None)

        # get the list of variables from the data columns
        vars = list(self.data.columns)

        # create all possible time-lagged-variables
        time_lagged_variables = []
        for var in vars:
            time_lagged_variables.append((var, 0))
            for lag in range(1, max_time_lag + 1):
                time_lagged_variables.append((var, lag))

        # show the initial progress bar
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working on conditional variables")

        # create the initial skeleton graph with all time-lagged-variables as nodes
        graph = nx.Graph()
        graph.add_nodes_from(time_lagged_variables)

        # add edges between variables according to the temporal constraints
        # - Variables at current time can be connected to all past variables and other current variables
        # - Variables at past times can only be connected to vairiables at curren time
        # or other variables at the current time
        for (var1, lag1), (var2, lag2) in combinations(time_lagged_variables, 2):
            # Skip the self loops
            if var1 == var2 and lag1 == lag2:
                continue

            # Apply the temporal constraints
            if lag1 > 0 and lag2 > 0:
                # both are past variables
                if lag1 == lag2:
                    graph.add_edge((var1, lag1), (var2, lag2))

            else:
                graph.add_edge((var1, lag1), (var2, lag2))

        lagged_data = self._create_lagged_data(self.data, max_time_lag)

        # Now run the PC stable algorithm
        while not all(
            [
                len(list(graph.neighbors(var))) <= cond_set_size
                for var in time_lagged_variables
            ]
        ):
            self._run_pc(
                graph,
                time_lagged_variables,
                lagged_data,
                ci_test,
                separating_sets,
                cond_set_size,
                significance_level,
                **kwargs,
            )

            # Increase the conditional set size
            if cond_set_size >= max_cond_vars:
                logger.info(
                    f"Maximum number of conditional variables {max_cond_vars} reached. Stopping the search."
                )
                break
            cond_set_size += 1
            if show_progress and config.SHOW_PROGRESS:
                pbar.update(1)
                pbar.set_description(
                    f"Working on conditional variables: {cond_set_size}/{max_cond_vars}"
                )

        if show_progress and config.SHOW_PROGRESS:
            pbar.close()

        return graph, separating_sets

    def _run_pc(
        self,
        graph,
        variables,
        data,
        ci_test,
        separating_sets,
        cond_set_size,
        significance_level,
        **kwargs,
    ):
        """
        Run the original PC algorithm for time series data.
        """
        # Respec the temporal constraints
        for u, v in list(graph.edges()):
            u_var, u_lag = u
            v_var, v_lag = v

            # find the potential separating sets respecting temporal ordering
            for sep_set in self._get_potential_sepsets(u, v, graph, cond_set_size):
                if ci_test(
                    u_var,
                    v_var,
                    [s[0] for s in sep_set],
                    data=data,
                    time_lag_u=u_lag,
                    time_lag_v=v_lag,
                    time_lag_sep=[s[1] for s in sep_set],
                    significance_level=significance_level,
                    **kwargs,
                ):
                    separating_sets[frozenset((u, v))] = sep_set
                    graph.remove_edge(u, v)
                    break

    def _get_potential_sepsets(self, u, v, graph, cond_set_size):
        """
        Get potential separating sets for nodes u and v, respecting temporal constraints.
        """
        u_var, u_lag = u
        v_var, v_lag = v

        # Get neighbors of u and v
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))

        # Remove v from neighbors of u and vice versa
        neighbors_u.discard(v)
        neighbors_v.discard(u)

        # Apply temporal constraints
        # - For contemporaneous edges (same time lag), consider neighbors at that time or later
        # - For time-lagged edges, consider only neighbors that respect causality
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

        # Generate combinations of valid neighbors
        return chain(
            combinations(valid_neighbors_u, cond_set_size),
            combinations(valid_neighbors_v, cond_set_size),
        )

    def _get_potential_sepsets_from_neighbors(self, u, v, neighbors, cond_set_size):
        """
        Get potential separating sets from precomputed neighbors, respecting temporal constraints.
        """
        _, u_lag = u
        _, v_lag = v

        # Get neighbors of u and v
        neighbors_u = set(neighbors[u])
        neighbors_v = set(neighbors[v])

        # Remove v from neighbors of u and vice versa
        neighbors_u.discard(v)
        neighbors_v.discard(u)

        # Apply temporal constraints - same as in _get_potential_sepsets
        min_lag = min(u_lag, v_lag)

        valid_neighbors_u = set()
        for neigh in neighbors_u:
            _, neigh_lag = neigh
            if neigh_lag >= min_lag:
                valid_neighbors_u.add(neigh)

        valid_neighbors_v = set()
        for neigh in neighbors_v:
            _, neigh_lag = neigh
            if neigh_lag >= min_lag:
                valid_neighbors_v.add(neigh)

        # Generate combinations of valid neighbors
        return chain(
            combinations(valid_neighbors_u, cond_set_size),
            combinations(valid_neighbors_v, cond_set_size),
        )

    def _orient_time_series_edges(self, skeleton, separating_sets, max_time_lag):
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
            u_var, u_lag = u
            v_var, v_lag = v

            # Same time lag (contemporaneous) - leave unoriented for now
            if u_lag == v_lag:
                continue

            # Different time lags - orient from higher lag (past) to lower lag (more recent/future)
            if u_lag > v_lag:  # u is in the past relative to v
                ts_dag.add_edge(u, v)
            else:  # v is in the past relative to u
                ts_dag.add_edge(v, u)

        # Next, orient v-structures (colliders) like in the PC algorithm
        node_pairs = list(permutations(sorted(skeleton.nodes()), 2))

        for pair in node_pairs:
            X, Y = pair
            if not skeleton.has_edge(X, Y):  # X and Y are not adjacent
                X_var, X_lag = X
                Y_var, Y_lag = Y

                # Find common neighbors (potential colliders)
                common_neighbors = set(skeleton.neighbors(X)) & set(
                    skeleton.neighbors(Y)
                )

                for Z in common_neighbors:
                    Z_var, Z_lag = Z

                    # Check if Z is not in the separating set of X and Y
                    if Z not in separating_sets.get(frozenset((X, Y)), []):
                        # Temporal constraint check - a variable can only be a collider if it's
                        # at the same time or more recent than both parents
                        if Z_lag <= X_lag and Z_lag <= Y_lag:
                            # Add directed edges to form a v-structure
                            if not ts_dag.has_edge(Z, X) and not ts_dag.has_edge(Z, Y):
                                ts_dag.add_edge(X, Z)
                                ts_dag.add_edge(Y, Z)

        # Apply orientation rules (Meek rules) to orient remaining edges
        # This needs careful consideration for time series data
        return ts_dag

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
        to control the common causes and indirect paths

        Parameters
        ----------
        ts_dag : TimeSeriesDAG
            The time series directed acyclic graph with potential causal links.

        ci_test : callable
            The cinditional independence test to use.

        significance_level : float
            The significance level for the conditional independence test.

        max_cond_vars : int
            The maximum number of conditioning variables to consider for the conditional independence test.

        n_jobs : int
            The number of jobs to run in parallel. Default is -1, which means using all processors.
            If 1, no parallel computing is used.
            If -1, all processors are used.

        show_progress : bool
            Whether to show a progress bar during the estimation process.
            Default is True.

        Returns
        -------
        ts_dag : TimeSeriesDAG
            The refined time series directed acyclic graph with causal links.
        """

        # Create lagged data for mci_tests
        max_time_lag = max(abs(lag) for _, lag in ts_dag.nodes())
        lagged_data = self._create_lagged_data(self.data, max_time_lag)
        ci_test = get_ci_test(
            ci_test, full=True, data=self.data, independencies=self.independencies
        )

        # get all the edges to test
        edges_to_test = list(ts_dag.edges())

        if show_progress and config.SHOW_PROGRESS:
            edges_to_test = tqdm(edges_to_test, desc="Running MCI tests")

        # Run MCI tests in parallel
        if n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._single_mci_test)(
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

            # remove the egdes that fail the mci test
            for edge, should_remove in zip(edges_to_test, results):
                if should_remove:
                    ts_dag.remove_edge(*edge)

        else:
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
                    ts_dag.remove_edge(*edge)

        return ts_dag

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

        ci_test = get_ci_test(
            ci_test,
            full=True,
            data=data,
            independencies=self.independencies,
        )

        # Get the parents of u and v in the current ts_dag
        parents_u = set(ts_dag.predecessors(u))
        parents_v = set(ts_dag.predecessors(v))

        # Get the common neighbors of u and v in the current ts_dag
        common_neighbors = set(ts_dag.neighbors(u)) & set(ts_dag.neighbors(v))

        # Create a list of all potential conditioning variables
        cond_vars = list(parents_u | parents_v | common_neighbors)

        # Check if there are enough conditioning variables to test
        if len(cond_vars) > max_cond_vars:
            logger.warning(
                f"Too many conditioning variables ({len(cond_vars)}) for edge ({u}, {v}). "
                f"Skipping MCI test."
            )
            return False

        # Run the MCI test with the specified conditional variables
        return ci_test(
            u[0],
            v[0],
            [s[0] for s in cond_vars],
            data=data,
            time_lag_u=u[1],
            time_lag_v=v[1],
            time_lag_sep=[s[1] for s in cond_vars],
            significance_level=significance_level,
            **kwargs,
        )

    def _create_lagged_data(self, data, max_lag):
        """Create a lagged version of the input data upto the maximum lag.

        Paramters
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
        variable_names : list
            List of (variable, lag) tuples representing the columns in the lagged data.
        """
        lagged_data = {}
        variable_names = []  # List to store all variable names

        for lag in range(0, max_lag + 1):
            lagged_df = data.shift(lag).copy()
            var_lag_tuples = [(col, lag) for col in data.columns]
            lagged_df.columns = var_lag_tuples
            lagged_data.update(lagged_df.to_dict(orient="series"))
            variable_names.extend(var_lag_tuples)

        lagged_df = pd.DataFrame(lagged_data)
        lagged_df.dropna(inplace=True)
        lagged_df.reset_index(drop=True, inplace=True)

        return lagged_df