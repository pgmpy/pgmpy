from collections.abc import Iterable
from itertools import chain, combinations, product

import networkx as nx
import numpy as np
from networkx.algorithms.dag import descendants
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators.LinearModel import LinearEstimator
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import (
    DiscreteBayesianNetwork,
    FunctionalBayesianNetwork,
    LinearGaussianBayesianNetwork,
    SEMGraph,
)
from pgmpy.utils.sets import _powerset, _variable_or_iterable_to_set


class CausalInference(object):
    """
    This is an inference class for performing Causal Inference over Bayesian
    Networks or Structural Equation Models.

    Parameters
    ----------
    model: pgmpy.base.DAG | pgmpy.models.DiscreteBayesianNetwork | pgmpy.models.SEMGraph
        The model that we'll perform inference over.

    Examples
    --------
    Create a small Bayesian Network.

    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> game = DiscreteBayesianNetwork([('X', 'A'),
    ...                                 ('A', 'Y'),
    ...                                 ('A', 'B')])

    Load the graph into the CausalInference object to make causal queries.

    >>> from pgmpy.inference.CausalInference import CausalInference
    >>> inference = CausalInference(game)
    >>> inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
    >>> inference.get_all_frontdoor_adjustment_sets(X="X", Y="Y")

    References
    ----------
    'Causality: Models, Reasoning, and Inference' - Judea Pearl (2000)
    """

    def __init__(self, model):
        if not isinstance(
            model,
            (
                DiscreteBayesianNetwork,
                LinearGaussianBayesianNetwork,
                FunctionalBayesianNetwork,
                SEMGraph,
                DAG,
            ),
        ):
            raise NotImplementedError(
                "Causal Inference is only implemented for DAGs, BayesianNetworks, and SEMGraphs."
            )

        # Check if the variable names are strings. If not, raise an error.
        bad_variable = model._variable_name_contains_non_string()
        if bad_variable != False:
            raise NotImplementedError(
                f"Causal Inference is only implemented for a model with "
                "variable names with string type. "
                f"Found {bad_variable[0]} with type {bad_variable[1]}. "
                "Convert them to string to proceed."
            )

        # Initialize data structures.
        self.model = model

        if isinstance(model, SEMGraph):
            self.observed_variables = frozenset(model.observed)
            self.latent_variables = model.latents
            self.dag = DAG(
                model.full_graph_struct,
                latents=model.latents.union(
                    set(
                        [
                            var
                            for var in model.full_graph_struct.nodes()
                            if var.startswith(".")
                        ]
                    )
                ),
            )

        elif isinstance(model, (DiscreteBayesianNetwork, DAG)):
            self.observed_variables = frozenset(model.nodes()).difference(model.latents)
            self.latent_variables = model.latents
            self.dag = DAG(model.to_directed(), latents=model.latents)

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.observed_variables)))
        return f"{self.__class__.__name__}({variables})"

    def is_valid_backdoor_adjustment_set(self, X, Y, Z=[]):
        """
        Test whether Z is a valid backdoor adjustment set for estimating the causal impact of X on Y.

        Parameters
        ----------

        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable.

        Z: list (array-like)
            List of adjustment variables.

        Returns
        -------
        Is a valid backdoor adjustment set: bool
            True if Z is a valid backdoor adjustment set else False

        Examples
        --------
        >>> game1 = DiscreteBayesianNetwork([('X', 'A'),
        ...                          ('A', 'Y'),
        ...                          ('A', 'B')])
        >>> inference = CausalInference(game1)
        >>> inference.is_valid_backdoor_adjustment_set("X", "Y")
        True
        """
        Z_ = _variable_or_iterable_to_set(Z)

        observed = [X] + list(Z_)
        parents_d_sep = []
        for p in self.dag.predecessors(X):
            parents_d_sep.append(not self.dag.is_dconnected(p, Y, observed=observed))
        return all(parents_d_sep)

    def get_all_backdoor_adjustment_sets(self, X, Y):
        """
        Returns a list of all adjustment sets per the back-door criterion.

        A set of variables Z satisfies the back-door criterion relative
          to an ordered pair of variabies (Xi, Xj) in a DAG G if:
            (i) no node in Z is a descendant of Xi; and
            (ii) Z blocks every path between Xi and Xj that contains an arrow into Xi.

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable.

        Returns
        -------
        frozenset: A frozenset of frozensets

        Y: str
            Target Variable

        Examples
        --------
        >>> game1 = DiscreteBayesianNetwork([('X', 'A'),
        ...                          ('A', 'Y'),
        ...                          ('A', 'B')])
        >>> inference = CausalInference(game1)
        >>> inference.get_all_backdoor_adjustment_sets("X", "Y")
        frozenset()
        """
        try:
            assert X in self.observed_variables
            assert Y in self.observed_variables
        except AssertionError:
            raise AssertionError("Make sure both X and Y are observed.")

        if self.is_valid_backdoor_adjustment_set(X, Y, Z=frozenset()):
            return frozenset()

        possible_adjustment_variables = (
            set(self.observed_variables) - {X} - {Y} - set(nx.descendants(self.dag, X))
        )

        valid_adjustment_sets = []
        for s in _powerset(possible_adjustment_variables):
            super_of_complete = []
            for vs in valid_adjustment_sets:
                super_of_complete.append(vs.intersection(set(s)) == vs)
            if any(super_of_complete):
                continue
            if self.is_valid_backdoor_adjustment_set(X, Y, s):
                valid_adjustment_sets.append(frozenset(s))

        if len(valid_adjustment_sets) == 0:
            raise ValueError(f"No valid adjustment set found for {X} -> {Y}")

        return frozenset(valid_adjustment_sets)

    def is_valid_frontdoor_adjustment_set(self, X, Y, Z=None):
        """
        Test whether Z is a valid frontdoor adjustment set for estimating the causal impact of X on Y via the frontdoor
        adjustment formula.

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable.

        Z: list (array-like)
            List of adjustment variables.

        Returns
        -------
        Is valid frontdoor adjustment: bool
            True if Z is a valid frontdoor adjustment set.
        """
        Z = _variable_or_iterable_to_set(Z)

        # 0. Get all directed paths from X to Y.  Don't check further if there aren't any.
        directed_paths = list(nx.all_simple_paths(self.dag, X, Y))

        if directed_paths == []:
            return False

        # 1. Z intercepts all directed paths from X to Y
        unblocked_directed_paths = [
            path for path in directed_paths if not any(zz in path for zz in Z)
        ]

        if unblocked_directed_paths:
            return False

        # 2. there is no backdoor path from X to Z
        unblocked_backdoor_paths_X_Z = [
            zz for zz in Z if not self.is_valid_backdoor_adjustment_set(X, zz)
        ]

        if unblocked_backdoor_paths_X_Z:
            return False

        # 3. All back-door paths from Z to Y are blocked by X
        valid_backdoor_sets = []
        for zz in Z:
            valid_backdoor_sets.append(self.is_valid_backdoor_adjustment_set(zz, Y, X))
        if not all(valid_backdoor_sets):
            return False

        return True

    def get_all_frontdoor_adjustment_sets(self, X, Y):
        """
        Identify possible sets of variables, Z, which satisfy the front-door criterion relative to given X and Y.

        Z satisfies the front-door criterion if:
          (i)    Z intercepts all directed paths from X to Y
          (ii)   there is no backdoor path from X to Z
          (iii)  all back-door paths from Z to Y are blocked by X

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable

        Returns
        -------
        frozenset: a frozenset of frozensets
        """
        assert X in self.observed_variables
        assert Y in self.observed_variables

        possible_adjustment_variables = set(self.observed_variables) - {X} - {Y}

        valid_adjustment_sets = frozenset(
            [
                frozenset(s)
                for s in _powerset(possible_adjustment_variables)
                if self.is_valid_frontdoor_adjustment_set(X, Y, s)
            ]
        )

        return valid_adjustment_sets

    def get_scaling_indicators(self):
        """
        Returns a scaling indicator for each of the latent variables in the model.
        The scaling indicator is chosen randomly among the observed measurement
        variables of the latent variable.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(ebunch=[('xi1', 'eta1'), ('xi1', 'x1'), ('xi1', 'x2'),
        ...                          ('eta1', 'y1'), ('eta1', 'y2')],
        ...                  latents=['xi1', 'eta1'])
        >>> model.get_scaling_indicators()
        {'xi1': 'x1', 'eta1': 'y1'}

        Returns
        -------
        dict: Returns a dict with latent variables as the key and their value being the
                scaling indicator.
        """
        scaling_indicators = {}
        for node in self.latent_variables:
            for neighbor in self.dag.neighbors(node):
                if neighbor in self.observed_variables:
                    scaling_indicators[node] = neighbor
                    break
        return scaling_indicators

    def _iv_transformations(self, X, Y, scaling_indicators={}):
        """
        Transforms the graph structure of SEM so that the d-separation criterion is
        applicable for finding IVs. The method transforms the graph for finding MIIV
        for the estimation of X \rightarrow Y given the scaling indicator for all the
        parent latent variables.

        Parameters
        ----------
        X: node
            The explantory variable.

        Y: node
            The dependent variable.

        scaling_indicators: dict
            Scaling indicator for each latent variable in the model.

        Returns
        -------
        nx.DiGraph: The transformed full graph structure.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(ebunch=[('xi1', 'eta1'), ('xi1', 'x1'), ('xi1', 'x2'),
        ...                          ('eta1', 'y1'), ('eta1', 'y2')],
        ...                  latents=['xi1', 'eta1'])
        >>> model._iv_transformations('xi1', 'eta1',
        ...                           scaling_indicators={'xi1': 'x1', 'eta1': 'y1'})
        """
        full_graph = self.dag.copy()

        if not (X, Y) in full_graph.edges():
            raise ValueError(f"The edge from {X} -> {Y} doesn't exist in the graph")

        if (X in self.observed_variables) and (Y in self.observed_variables):
            full_graph.remove_edge(X, Y)
            return full_graph, Y

        elif Y in self.latent_variables:
            full_graph.add_edge("." + Y, scaling_indicators[Y])
            dependent_var = scaling_indicators[Y]
        else:
            dependent_var = Y

        # This check is to not remove edges from error terms to the variable. Specifically for SEMs.
        variable_parents = [
            var for var in self.dag.predecessors(Y) if not var.startswith(".")
        ]

        for parent_y in variable_parents:
            full_graph.remove_edge(parent_y, Y)
            if parent_y in self.latent_variables:
                full_graph.add_edge("." + scaling_indicators[parent_y], dependent_var)

        return full_graph, dependent_var

    def get_ivs(self, X, Y, scaling_indicators={}):
        """
        Returns the Instrumental variables(IVs) for the relation X -> Y

        Parameters
        ----------
        X: node
            The variable name (observed or latent)

        Y: node
            The variable name (observed or latent)

        scaling_indicators: dict (optional)
            A dict representing which observed variable to use as scaling indicator for
            the latent variables.
            If not given the method automatically selects one of the measurement variables
            at random as the scaling indicator.

        Returns
        -------
        set: {str}
            The set of Instrumental Variables for X -> Y.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(ebunch=[('I', 'X'), ('X', 'Y')],
        ...                  latents=[],
        ...                  err_corr=[('X', 'Y')])
        >>> model.get_ivs('X', 'Y')
        {'I'}
        """
        if not scaling_indicators:
            scaling_indicators = self.get_scaling_indicators()

        if (X in scaling_indicators.keys()) and (scaling_indicators[X] == Y):
            logger.warning(
                f"{Y} is the scaling indicator of {X}. Please specify `scaling_indicators`"
            )

        transformed_graph, dependent_var = self._iv_transformations(
            X, Y, scaling_indicators=scaling_indicators
        )

        if X in self.latent_variables:
            explanatory_var = scaling_indicators[X]
        else:
            explanatory_var = X

        d_connected_x = transformed_graph.active_trail_nodes([explanatory_var])[
            explanatory_var
        ]

        # Compute the d-connected nodes to Y except any variable connected through X.
        transformed_graph_copy = transformed_graph.copy()
        transformed_graph_copy.remove_edges_from(
            list(transformed_graph_copy.in_edges(explanatory_var))
        )
        d_connected_y = transformed_graph_copy.active_trail_nodes([dependent_var])[
            dependent_var
        ]

        # Remove {X, Y} because they can't be IV for X -> Y
        return d_connected_x - d_connected_y - {dependent_var, explanatory_var}

    def get_conditional_ivs(self, X, Y, scaling_indicators={}):
        """
        Returns the conditional IVs for the relation X -> Y

        Parameters
        ----------
        X: node
            The observed variable's name

        Y: node
            The oberved variable's name

        scaling_indicators: dict (optional)
            A dict representing which observed variable to use as scaling indicator for
            the latent variables.
            If not provided, automatically finds scaling indicators by randomly selecting
            one of the measurement variables of each latent variable.

        Returns
        -------
        set: Set of 2-tuples representing tuple[0] is an IV for X -> Y given tuple[1].

        References
        ----------
        .. [1] Van Der Zander, B., Textor, J., & Liskiewicz, M. (2015, June). Efficiently finding
               conditional instruments for causal inference. In Twenty-Fourth International Joint
               Conference on Artificial Intelligence.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(ebunch=[('I', 'X'), ('X', 'Y'), ('W', 'I')],
        ...                  latents=[],
        ...                  err_corr=[('W', 'Y')])
        >>> model.get_ivs('X', 'Y')
        [('I', {'W'})]
        """
        if not scaling_indicators:
            scaling_indicators = self.get_scaling_indicators()

        if (X in scaling_indicators.keys()) and (scaling_indicators[X] == Y):
            logger.warning(
                f"{Y} is the scaling indicator of {X}. Please specify `scaling_indicators`"
            )

        transformed_graph, dependent_var = self._iv_transformations(
            X, Y, scaling_indicators=scaling_indicators
        )
        if (X, Y) in transformed_graph.edges:
            G_c = transformed_graph.remove_edge(X, Y)
        else:
            G_c = transformed_graph

        instruments = []
        for Z in self.observed_variables - {X, Y}:
            W = self._nearest_separator(G_c, Y, Z)
            # Condition to check if W d-separates Y from Z
            if (not W) or (W.intersection(descendants(G_c, Y))) or (X in W):
                continue

            # Condition to check if X d-connected to I after conditioning on W.
            elif X in self.model.active_trail_nodes([Z], observed=W)[Z]:
                instruments.append((Z, W))
            else:
                continue
        return instruments

    def get_total_conditional_ivs(self, X, Y, scaling_indicators={}):
        all_paths = list(nx.all_simple_paths(self.dag, X, Y))
        nodes_on_paths = set([node for path in all_paths for node in path])
        nodes_on_paths = nodes_on_paths - {X, Y}

        transformed_graph, dependent_var = self._iv_transformations(
            X, Y, scaling_indicators=scaling_indicators
        )

        if (X, Y) in transformed_graph.edges():
            transformed_graph.remove_edge(X, Y)

        instruments = []
        for Z in self.observed_variables - {X, Y}:
            W = self._nearest_separator(transformed_graph, Y, Z)

            # Check if W contains any nodes on paths from X to Y
            if W and W.intersection(nodes_on_paths):
                # Skip this instrument if it requires conditioning on nodes in paths
                continue

            # Regular conditions from get_conditional_ivs
            if (
                (not W)
                or (W.intersection(descendants(transformed_graph, Y)))
                or (X in W)
            ):
                continue
            elif X in self.model.active_trail_nodes([Z], observed=W)[Z]:
                instruments.append((Z, W))
            else:
                continue

        return instruments

    def identification_method(self, X, Y):
        """
        Automatically identifies a valid method for estimating the causal effect from X to Y.

        Parameters
        ----------
        X: str
            The treatment/exposure variable
        Y: str
            The outcome variable

        Returns
        -------
        dict
            A dictionary containing keys as method and value as the corresponding result.
        """
        result = {}

        try:
            backdoor_sets = self.get_all_backdoor_adjustment_sets(X, Y)
            if len(backdoor_sets) > 0:
                result["backdoor set"] = backdoor_sets
        except:
            pass

        try:
            frontdoor_sets = self.get_all_frontdoor_adjustment_sets(X, Y)
            if len(frontdoor_sets) > 0:
                result["frontdoor set"] = frontdoor_sets
        except:
            pass

        try:
            instruments = self.get_ivs(X, Y)
            if len(instruments) > 0:
                result["instrumental variables"] = instruments
        except:
            pass

        try:
            conditional_ivs = self.get_conditional_ivs(X, Y)
            if len(conditional_ivs) > 0:
                result["conditional instrumental variables"] = conditional_ivs
        except:
            pass

        try:
            total_conditional_ivs = self.get_total_conditional_ivs(X, Y)
            if len(total_conditional_ivs) > 0:
                result["total conditional instrumental variables"] = (
                    total_conditional_ivs
                )
        except:
            pass

        return result

    def _nearest_separator(self, G, Y, Z):
        """
        Finds the set of the nearest separators for `Y` and `Z` in `G`.

        Parameters
        ----------
        G: nx.DiGraph instance
            The graph in which to the find the nearest separation for `Y` and `Z`.

        Y: str
            The variable name for which the separators are needed.

        Z: str
            The other variable for which the separators are needed.

        Returns
        -------
        set or None: If there is a nearest separator returns the set of separators else returns None.
        """
        W = set()
        ancestral_G = G.subgraph(
            nx.ancestors(G, Y).union(nx.ancestors(G, Z)).union({Y, Z})
        ).copy()

        if isinstance(self.model, SEMGraph):
            # Optimization: Remove all error nodes which don't have
            #  any correlation as it doesn't add any new path.
            #  If not removed it can create a lot of
            # extra paths resulting in a much higher runtime.
            err_nodes_to_remove = set(self.model.err_graph.nodes()) - set(
                [node for edge in self.model.err_graph.edges() for node in edge]
            )
            ancestral_G.remove_nodes_from(["." + node for node in err_nodes_to_remove])

        M = ancestral_G.moralize()
        visited = set([Y])
        to_visit = list(M.neighbors(Y))

        # Another optimization over the original algo. Rather than going through all the paths does
        # a DFS search to find a markov blanket of observed variables. This doesn't ensure minimal observed
        # set.
        while to_visit:
            node = to_visit.pop()
            if node == Z:
                return None
            visited.add(node)
            if node in self.observed_variables:
                W.add(node)
            else:
                to_visit.extend(
                    [node for node in M.neighbors(node) if node not in visited]
                )
        # for path in nx.all_simple_paths(M, Y, Z):
        #     path_set = set(path)
        #     if (len(path) >= 3) and not (W & path_set):
        #         for index in range(1, len(path)-1):
        #             if path[index] in self.observed:
        #                 W.add(path[index])
        #                 break
        if Y not in G.active_trail_nodes([Z], observed=W)[Z]:
            return W
        else:
            return None

    def _simple_decision(self, adjustment_sets=[]):
        """
        Selects the smallest set from provided adjustment sets.

        Parameters
        ----------
        adjustment_sets: iterable
            A frozenset or list of valid adjustment sets

        Returns
        -------
        frozenset
        """
        adjustment_list = list(adjustment_sets)
        if adjustment_list == []:
            return frozenset([])
        return adjustment_list[np.argmin(adjustment_list)]

    def estimate_ate(
        self,
        X,
        Y,
        data,
        estimand_strategy="smallest",
        estimator_type="linear",
        **kwargs,
    ):
        """
        Estimate the average treatment effect (ATE) of X on Y.

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable

        data: pandas.DataFrame
            All observed data for this Bayesian Network.

        estimand_strategy: str or frozenset
            Either specify a specific backdoor adjustment set or a strategy.
            The available options are:
                smallest:
                    Use the smallest estimand of observed variables
                all:
                    Estimate the ATE from each identified estimand

        estimator_type: str
            The type of model to be used to estimate the ATE.
            All of the linear regression classes in statsmodels are available including:
                * GLS: generalized least squares for arbitrary covariance
                * OLS: ordinary least square of i.i.d. errors
                * WLS: weighted least squares for heteroskedastic error
            Specify them with their acronym (e.g. "OLS") or simple "linear" as an alias for OLS.

        **kwargs: dict
            Keyward arguments specific to the selected estimator.
            linear:
              missing: str
                Available options are "none", "drop", or "raise"

        Returns
        -------
        The average treatment effect: float

        Examples
        --------
        >>> import pandas as pd
        >>> game1 = DiscreteBayesianNetwork([('X', 'A'),
        ...                          ('A', 'Y'),
        ...                          ('A', 'B')])
        >>> data = pd.DataFrame(np.random.randint(2, size=(1000, 4)), columns=['X', 'A', 'B', 'Y'])
        >>> inference = CausalInference(model=game1)
        >>> inference.estimate_ate("X", "Y", data=data, estimator_type="linear")
        """
        valid_estimators = ["linear"]
        try:
            assert estimator_type in valid_estimators
        except AssertionError:
            print(
                f"{estimator_type} if not a valid estimator_type.  Please select from {valid_estimators}"
            )
        all_simple_paths = nx.all_simple_paths(self.model, X, Y)
        all_path_effects = []
        for path in all_simple_paths:
            causal_effect = []
            for x1, x2 in zip(path, path[1:]):
                if isinstance(estimand_strategy, frozenset):
                    adjustment_set = frozenset({estimand_strategy})
                    assert self.is_valid_backdoor_adjustment_set(
                        x1, x2, Z=adjustment_set
                    )
                elif estimand_strategy in ["smallest", "all"]:
                    adjustment_sets = self.get_all_backdoor_adjustment_sets(x1, x2)
                    if estimand_strategy == "smallest":
                        adjustment_sets = frozenset(
                            {self._simple_decision(adjustment_sets)}
                        )

                if estimator_type == "linear":
                    self.estimator = LinearEstimator(self.model)

                ate = [
                    self.estimator.fit(X=x1, Y=x2, Z=s, data=data, **kwargs)._get_ate()
                    for s in adjustment_sets
                ]
                causal_effect.append(np.mean(ate))
            all_path_effects.append(np.prod(causal_effect))
        return np.sum(all_path_effects)

    def get_proper_backdoor_graph(self, X, Y, inplace=False):
        """
        Returns a proper backdoor graph for the exposure `X` and outcome `Y`.
        A proper backdoor graph is a graph which remove the first edge of every
        proper causal path from `X` to `Y`.

        Parameters
        ----------
        X: list (array-like)
            A list of exposure variables.

        Y: list (array-like)
            A list of outcome variables

        inplace: boolean
            If inplace is True, modifies the object itself. Otherwise retuns
            a modified copy of self.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import CausalInference
        >>> model = DiscreteBayesianNetwork([("x1", "y1"), ("x1", "z1"), ("z1", "z2"),
        ...                        ("z2", "x2"), ("y2", "z2")])
        >>> c_infer = CausalInference(model)
        >>> c_infer.get_proper_backdoor_graph(X=["x1", "x2"], Y=["y1", "y2"])
        <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork at 0x7fba501ad940>

        References
        ----------
        [1] Perkovic, Emilija, et al.
         "Complete graphical characterization and construction of
         adjustment sets in Markov equivalence classes of ancestral graphs."
           The Journal of Machine Learning Research 18.1 (2017): 8132-8193.
        """
        if isinstance(X, str):
            X = [X]
        if isinstance(Y, str):
            Y = [Y]

        for var in chain(X, Y):
            if var not in self.dag.nodes():
                raise ValueError(f"{var} not found in the model.")

        model = self.dag if inplace else self.dag.copy()
        edges_to_remove = []
        for source in X:
            paths = nx.all_simple_edge_paths(model, source, Y)
            for path in paths:
                edges_to_remove.append(path[0])
        model.remove_edges_from(edges_to_remove)
        return model

    def is_valid_adjustment_set(self, X, Y, adjustment_set):
        """
        Method to test whether `adjustment_set` is a valid adjustment set for
        identifying the causal effect of `X` on `Y`.

        Parameters
        ----------
        X: list (array-like)
            The set of cause variables.

        Y: list (array-like)
            The set of predictor variables.

        adjustment_set: list (array-like)
            The set of variables for which to test whether they satisfy the
            adjustment set criteria.

        Returns
        -------
        Is valid adjustment set: bool
            Returns True if `adjustment_set` is a valid adjustment set for
            identifying the effect of `X` on `Y`. Else returns False.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import CausalInference
        >>> model = DiscreteBayesianNetwork([("x1", "y1"), ("x1", "z1"), ("z1", "z2"),
        ...                        ("z2", "x2"), ("y2", "z2")])
        >>> c_infer = CausalInference(model)
        >>> c_infer.is_valid_adjustment_set(X=['x1', 'x2'], Y=['y1', 'y2'], adjustment_set=['z1', 'z2'])
        True

        References
        ----------
        [1] Perkovic, Emilija, et al.
          "Complete graphical characterization and construction of
            adjustment sets in Markov equivalence classes of ancestral graphs."
              The Journal of Machine Learning Research 18.1 (2017): 8132-8193.
        """
        if isinstance(X, str):
            X = [X]
        if isinstance(Y, str):
            Y = [Y]

        backdoor_graph = self.get_proper_backdoor_graph(X, Y, inplace=False)
        for x, y in zip(X, Y):
            if backdoor_graph.is_dconnected(start=x, end=y, observed=adjustment_set):
                return False
        return True

    def get_minimal_adjustment_set(self, X, Y):
        """
        Returns a minimal adjustment set for
        identifying the causal effect of `X` on `Y`.

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable

        Returns
        -------
        Minimal adjustment set: set or None
            A set of variables which are the minimal possible adjustment set. If
            None, no adjustment set is possible.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import CausalInference
        >>> dag = DiscreteBayesianNetwork([("X_1", "X_2"), ("Z", "X_1"), ("Z", "X_2")])
        >>> infer = CausalInference(dag)
        >>> infer.get_minimal_adjustment_set("X_1", "X_2")
        {'Z'}

        References
        ----------
        [1] Perkovic, Emilija, et al.
          "Complete graphical characterization and construction of
            adjustment sets in Markov equivalence classes of ancestral graphs."
              The Journal of Machine Learning Research 18.1 (2017): 8132-8193.
        """
        backdoor_graph = self.get_proper_backdoor_graph([X], [Y], inplace=False)
        return backdoor_graph.minimal_dseparator(X, Y)

    def query(
        self,
        variables,
        do=None,
        evidence=None,
        adjustment_set=None,
        inference_algo="ve",
        show_progress=True,
        **kwargs,
    ):
        """
        Performs a query on the model of the form :math:`P(X | do(Y), Z)` where :math:`X`
        is `variables`, :math:`Y` is `do` and `Z` is the `evidence`.

        Parameters
        ----------
        variables: list
            list of variables in the query i.e. `X` in :math:`P(X | do(Y), Z)`.

        do: dict (default: None)
            Dictionary of the form {variable_name: variable_state} representing
            the variables on which to apply the do operation i.e. `Y` in
            :math:`P(X | do(Y), Z)`.

        evidence: dict (default: None)
            Dictionary of the form {variable_name: variable_state} repesenting
            the conditional variables in the query i.e. `Z` in :math:`P(X |
            do(Y), Z)`.

        adjustment_set: str or list (default=None)
            Specifies the adjustment set to use. If None, uses the parents of the
            do variables as the adjustment set.

        inference_algo: str or pgmpy.inference.Inference instance
            The inference algorithm to use to compute the probability values.
            String options are: 1) ve: Variable Elimination 2) bp: Belief
            Propagation.

        kwargs: Any
            Additional paramters which needs to be passed to inference
            algorithms.  Please refer to the pgmpy.inference.Inference for
            details.

        Returns
        -------
        Queried distribution: pgmpy.factor.discrete.DiscreteFactor
            A factor object representing the joint distribution over the variables in `variables`.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model('alarm')
        >>> infer = CausalInference(model)
        >>> infer.query(['HISTORY'], do={'CVP': 'LOW'}, evidence={'HR': 'LOW'})
        <DiscreteFactor representing phi(HISTORY:2) at 0x7f4e0874c2e0>
        """
        # Step 1: Check if all the arguments are valid and get them to uniform types.
        if (not isinstance(variables, Iterable)) or (isinstance(variables, str)):
            raise ValueError(
                f"variables much be a list (array-like). Got type: {type(variables)}."
            )
        elif not all([node in self.model.nodes() for node in variables]):
            raise ValueError(
                f"Some of the variables in `variables` are not in the model."
            )
        else:
            variables = list(variables)

        if do is None:
            do = {}
        elif not isinstance(do, dict):
            raise ValueError(
                "`do` must be a dict of the form: {variable_name: variable_state}"
            )
        if evidence is None:
            evidence = {}
        elif not isinstance(evidence, dict):
            raise ValueError(
                "`evidence` must be a dict of the form: {variable_name: variable_state}"
            )

        if do:
            for var, do_var in product(variables, do):
                if do_var in nx.descendants(self.dag, var):
                    raise ValueError(
                        f"Invalid causal query: There is a direct edge from the query variable"
                        f" '{var}' to the intervention variable '{do_var}'. "
                        f"In causal inference, you can typically only query the effect on variables"
                        f" that are descendants of the intervention."
                    )

        from pgmpy.inference import Inference

        if inference_algo == "ve":
            from pgmpy.inference import VariableElimination

            inference_algo = VariableElimination
        elif inference_algo == "bp":
            from pgmpy.inference import BeliefPropagation

            inference_algo = BeliefPropagation
        elif not isinstance(inference_algo, Inference):
            raise ValueError(
                f"inference_algo must be one of: 've', 'bp', or an "
                f"instance of pgmpy.inference.Inference. Got: {inference_algo}"
            )

        # Step 2: Check if adjustment set is provided, otherwise try calculating it.
        if adjustment_set is None:
            do_vars = [var for var, state in do.items()]
            adjustment_set = set(
                chain(*[self.model.predecessors(var) for var in do_vars])
            )
            if len(adjustment_set.intersection(self.model.latents)) != 0:
                raise ValueError(
                    "Not all parents of do variables are observed. Please specify an adjustment set."
                )

        infer = inference_algo(self.model)

        # Step 3.1: If no do variable specified, do a normal probabilistic inference.
        if do == {}:
            return infer.query(variables, evidence, show_progress=False)
        # Step 3.2: If no adjustment is required, do a normal probabilistic
        #           inference with do variables as the evidence.
        elif len(adjustment_set) == 0:
            evidence = {**evidence, **do}
            return infer.query(variables, evidence, show_progress=False)

        # Step 4: For other cases, compute \sum_{z} p(variables | do, z) p(z)
        values = []

        # Step 4.1: Compute p_z and states of z to iterate over.
        # For computing p_z, if evidence variables also in adjustment set,
        # manually do reduce else inference will throw error.
        evidence_adj_inter = {
            var: state
            for var, state in evidence.items()
            if var in adjustment_set.intersection(evidence.keys())
        }
        if len(evidence_adj_inter) != 0:
            p_z = infer.query(adjustment_set, show_progress=False).reduce(
                [(key, value) for key, value in evidence_adj_inter.items()],
                inplace=False,
            )
            # Since we are doing reduce over some of the variables, they are
            # going to be removed from the factor but would be required to get
            # values later. A hackish solution to reintroduce those variables in p_z

            if set(p_z.variables) != adjustment_set:
                p_z = DiscreteFactor(
                    p_z.variables + list(evidence_adj_inter.keys()),
                    list(p_z.cardinality) + [1] * len(evidence_adj_inter),
                    p_z.values,
                    state_names={
                        **p_z.state_names,
                        **{var: [state] for var, state in evidence_adj_inter.items()},
                    },
                )
        else:
            p_z = infer.query(adjustment_set, evidence=evidence, show_progress=False)

        adj_states = []
        for var in adjustment_set:
            if var in evidence_adj_inter.keys():
                adj_states.append([evidence_adj_inter[var]])
            else:
                adj_states.append(self.model.get_cpds(var).state_names[var])

        # Step 4.2: Iterate over states of adjustment set and compute values.
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=np.prod([len(states) for states in adj_states]))

        for state_comb in product(*adj_states):
            adj_evidence = {
                var: state for var, state in zip(adjustment_set, state_comb)
            }
            evidence = {**do, **adj_evidence}
            values.append(
                infer.query(variables, evidence=evidence, show_progress=False)
                * p_z.get_value(**adj_evidence)
            )

            if show_progress and config.SHOW_PROGRESS:
                pbar.update(1)

        return sum(values).normalize(inplace=False)
