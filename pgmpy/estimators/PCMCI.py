#!/usr/bin/env python

from itertools import chain, combinations, permutations

import networkx as nx
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import PDAG, DAG, TimeSeriesDAG
from pgmpy.estimators import ExpertKnowledge, StructureEstimator
from pgmpy.estimators.CITests import get_ci_test
from pgmpy.global_vars import logger

class PCMCI(StructureEstimator, TimeSeriesDAG):
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

    def __init__(self, data=None, independencies=None,  **kwargs):
        super(PCMCI, self).__init__(data=data, independencies=independencies, **kwargs)

    def estimate(
        self,
        variant="parallel",
        ci_test="pearsonr",
        return_type="pdag",
        significance_level=0.05,
        show_progress=True,
        n_jobs=-1,
        expert_knowledge=None,
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

        expert_knowledge : pgmpy.estimators.ExpertKnowledge instance
            Expert knowledge to be used with the algorithm.

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
        >>> data = pd.DataFrame(np.random.randn(T, 2), columns=['X', 'Y])
        >>> for t in range(1, T):
        ...             data.loc[t, 'Y'] += 0.5 * data.loc[t-1, 'X'] + 0.5 * data.loc[t-1, 'Y] + 0.1 * np.random.randn()
        >>> pcmci = PCMCI(data)
        >>> ts_tag = pcmci.estimate(max_time_lag=2)
        >>> print(ts_tag.edges())
        [(('X', 1), ('Y', 0)), (('Y', 1), ('Y', 0))]
        """

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()
        
        # get the approproate CI test
        ci_test_function = get_ci_test(
            ci_test,
            full=True,
            data=self.data,
            independencies=self.independencies,
        )

        # Run the PC algorithm to find the skeleton with time-lagged variables
        skeleton, separating_sets = self._build_time_series_skeleton(
            ci_test=ci_test_function,
            max_time_lag=max_time_lag,
            significance_level=significance_level,
            show_progress=show_progress,
            n_jobs=n_jobs,
            expert_knowledge=expert_knowledge,
            max_cond_vars=max_cond_vars,
            **kwargs,
        )

        if return_type == "skeleton":
            return skeleton, separating_sets

        # orient the edges based on the time order
        # Past cant predict the future
        ts_dag = self._orient_edges(
            skeleton,
            separating_sets,
            max_time_lag=max_time_lag,
            expert_knowledge=expert_knowledge,
        )

        # Run the MCI tests to further refine the causal links
        ts_dag = self._run_mci_tests(
            ts_dag,
            ci_test=ci_test_function,
            significance_level=significance_level,
            show_progress=show_progress,
            n_jobs=n_jobs,
            expert_knowledge=expert_knowledge,
            max_cond_vars=max_cond_vars,
            **kwargs,
        )

        if return_type == "ts_dag":
            return ts_dag
        elif return_type == "summary_graph":
            return ts_dag.to_summary_graph()
        else:
            raise ValueError(
                "Invalid return_type. Must be one of 'ts_dag', 'summary_graph', or 'pdag'."
            )
        
    def _build_time_series_skeleton(
        self,
        ci_test,
        max_time_lag=3,
        significance_level=0.05,
        show_progress=True,
        n_jobs=-1,
        expert_knowledge=None,
        max_cond_vars=5,
        variant="parallel",
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
        lim_neighbors = 0
        separating_sets = {}

        # get the list of variables from the data columns
        vars = list(self.data.columns)
        
        # create all possible time-lagged-variables
        time_lagged_variables = []
        for var in vars:
            time_lagged_variables.append((var, 0))
            for lag in range(1, max_time_lag+1):
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
        for (var1, lag1) , (var2, lag2) in combinations(time_lagged_variables, 2):
            # Skip the self loops
            if var1 == var2 and lag1 == lag2:
                continue
                
            # Apply the temporal constraints
            if lag1 > 0 and lag2 > 0:
                # both are past variables
                if lag1 == lag2:
                    graph.add_edge((var1, lag1) , (var2, lag2))
            
            else:
                graph.add_edge((var1, lag1) , (var2, lag2))
            
        lagged_data = self._create_lagged_data(self.data, max_time_lag)

        # Now run the PC stable algorithm
        while not all(
            [len(list(graph.neighbors(var))) <= lim_neighbors for var in time_lagged_variables]

        ):
            # Implement the PC algorithm with temporal constraints
            if variant == "orig":
                self._run_pc_orig(graph, time_lagged_variables, lagged_data, ci_test, 
                                 separating_sets, lim_neighbors, significance_level,
                                 expert_knowledge, **kwargs)
            elif variant == "stable":
                self._run_pc_stable(graph, time_lagged_variables, lagged_data, ci_test, 
                                   separating_sets, lim_neighbors, significance_level,
                                   expert_knowledge, **kwargs)
            elif variant == "parallel":
                self._run_pc_parallel(graph, time_lagged_variables, lagged_data, ci_test, 
                                     separating_sets, lim_neighbors, significance_level,
                                     expert_knowledge, n_jobs, **kwargs)
            else:
                raise ValueError(
                    f"variant must be one of (orig, stable, parallel). Got: {variant}"
                )

            # Increase the conditional set size
            if lim_neighbors >= max_cond_vars:
                logger.info(
                    f"Maximum number of conditional variables {max_cond_vars} reached. Stopping the search."
                )
                break
            lim_neighbors += 1
            if show_progress and config.SHOW_PROGRESS:
                pbar.update(1)
                pbar.set_description(
                    f"Working on conditional variables: {lim_neighbors}/{max_cond_vars}"
                )
        
        if show_progress and config.SHOW_PROGRESS:
            pbar.close()
        
        return graph, separating_sets

    def _run_pc_orig(self, graph, variables, data, ci_test, seperating_sets,
                lim_neighbors, significance_level, expert_knowledge, **kwargs):
        """
        Run the original PC algorithm for time series data.
        """
        # Implement the original PC algorithm
        pass

    def _run_pc_stable(self, graph, variables, data, ci_test, seperating_sets,
                lim_neighbors, significance_level, expert_knowledge, **kwargs):
        """
        Run the stable PC algorithm for time series data.
        """
        # Implement the stable PC algorithm
        pass 

    def _run_pc_parallel(self, graph, variables, data, ci_test, seperating_sets,
                lim_neighbors, significance_level, expert_knowledge, n_jobs, **kwargs):
        """
        Run the parallel PC algorithm for time series data.
        """
        # Implement the parallel PC algorithm
        pass

    @staticmethod
    def _get_potential_sepsets(u, v, temporal_ordering, graph, lim_neighbors):
        """
        Return the temporally consistent superset of separating set of u, v.

        The temporal order (if specified) of the superset can only be smaller
        ("earlier") than the particular node. The neighbors of 'u' satisfying
        this condition are returned.

        Parameters
        ----------
        u: variable
            The node whose neighbors are being considered for separating set.

        v: variable
            The node along with u whose separating set is being calculated.

        temporal_ordering: dict
            The temporal ordering of variables according to prior knowledgee.

        graph: UndirectedGraph
            The graph where separating sets are being calculated for the edges.

        lim_neighbors: int
            The maximum number of neighbours (conditioning variables) for u, v.

        Returns
        --------
        separating_set: set
            Set containing the superset of separating set of u, v.
        """
        separating_set_u = set(graph.neighbors(u))
        separating_set_v = set(graph.neighbors(v))
        separating_set_u.discard(v)
        separating_set_v.discard(u)

        if temporal_ordering != dict():
            max_order = min(temporal_ordering[u], temporal_ordering[v])
            for neigh in list(separating_set_u):
                if temporal_ordering[neigh] > max_order:
                    separating_set_u.discard(neigh)

            for neigh in list(separating_set_v):
                if temporal_ordering[neigh] > max_order:
                    separating_set_v.discard(neigh)

        return chain(
            combinations(separating_set_u, lim_neighbors),
            combinations(separating_set_v, lim_neighbors),
        )

    def _orient_time_series_edges(self, skeleton, separating_sets, max_time_lag, expert_knowledge=None):
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
            
        expert_knowledge : pgmpy.estimators.ExpertKnowledge, optional
            Expert knowledge to be used for edge orientation.
        
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
                common_neighbors = set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y))
                
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
        
        # Apply expert knowledge if provided
        if expert_knowledge:
            # Convert expert knowledge edges to time-lagged format if needed
            # This would require extending ExpertKnowledge to handle time series data
            pass
        
        # Apply orientation rules (Meek rules) to orient remaining edges
        # This needs careful consideration for time series data
        return ts_dag

            