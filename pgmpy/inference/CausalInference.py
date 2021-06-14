from collections.abc import Iterable
from itertools import chain, product

import numpy as np
import networkx as nx
from tqdm import tqdm

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.estimators.LinearModel import LinearEstimator
from pgmpy.global_vars import SHOW_PROGRESS
from pgmpy.utils.sets import _powerset, _variable_or_iterable_to_set


class CausalInference(object):
    """
    This is an inference class for performing Causal Inference over Bayesian Networks or Structural Equation Models.

    This class will accept queries of the form: P(Y | do(X)) and utilize its methods to provide an estimand which:
     * Identifies adjustment variables
     * Backdoor Adjustment
     * Front Door Adjustment
     * Instrumental Variable Adjustment

    Parameters
    ----------
    model: CausalGraph
        The model that we'll perform inference over.

    set_nodes: list[node:str] or None
        A list (or set/tuple) of nodes in the Bayesian Network which have been set to a specific value per the
        do-operator.

    Examples
    --------
    Create a small Bayesian Network.
    >>> from pgmpy.models.BayesianModel import BayesianModel
    >>> game = BayesianModel([('X', 'A'),
    ...                       ('A', 'Y'),
    ...                       ('A', 'B')])

    Load the graph into the CausalInference object to make causal queries.
    >>> from pgmpy.inference.CausalInference import CausalInference
    >>> inference = CausalInference(game)
    >>> inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
    >>> inference.get_all_frontdoor_adjustment_sets(X="X", Y="Y")

    References
    ----------
    'Causality: Models, Reasoning, and Inference' - Judea Pearl (2000)

    Many thanks to @ijmbarr for their implementation of Causal Graphical models available. It served as an invaluable
    reference. Available on GitHub: https://github.com/ijmbarr/causalgraphicalmodels
    """

    def __init__(self, model, set_nodes=None):
        if not isinstance(model, BayesianModel):
            raise NotImplementedError(
                "Causal Inference is only implemented for BayesianModels at this time."
            )
        self.model = model
        self.set_nodes = _variable_or_iterable_to_set(set_nodes)
        self.observed_variables = frozenset(self.model.nodes()).difference(
            model.latents
        )

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.observed_variables)))
        return f"{self.__class__.__name__}({variables})"

    def is_valid_backdoor_adjustment_set(self, X, Y, Z=[]):
        """
        Test whether Z is a valid backdoor adjustment set for estimating the causal impact of X on Y.

        Parameters
        ----------
        X: str
            Intervention Variable

        Y: str
            Target Variable

        Z: str or set[str]
            Adjustment variables

        Returns
        -------
        boolean: True if Z is a valid backdoor adjustment set.

        Examples
        --------
        >>> game1 = BayesianModel([('X', 'A'),
        ...                        ('A', 'Y'),
        ...                        ('A', 'B')])
        >>> inference = CausalInference(game1)
        >>> inference.is_valid_backdoor_adjustment_set("X", "Y")
        True
        """
        Z_ = list(Z)
        observed = [X] + Z_
        parents_d_sep = []
        for p in self.model.predecessors(X):
            parents_d_sep.append(not self.model.is_dconnected(p, Y, observed=observed))
        return all(parents_d_sep)

    def get_all_backdoor_adjustment_sets(self, X, Y):
        """
        Returns a list of all adjustment sets per the back-door criterion.

        A set of variables Z satisfies the back-door criterion relative to an ordered pair of variabies (Xi, Xj) in a DAG G if:
            (i) no node in Z is a descendant of Xi; and
            (ii) Z blocks every path between Xi and Xj that contains an arrow into Xi.

        TODO:
          * Backdoors are great, but the most general things we could implement would be Ilya Shpitser's ID and
            IDC algorithms. See [his Ph.D. thesis for a full explanation]
            (https://ftp.cs.ucla.edu/pub/stat_ser/shpitser-thesis.pdf). After doing a little reading it is clear
            that we do not need to immediatly implement this.  However, in order for us to truly account for
            unobserved variables, we will need not only these algorithms, but a more general implementation of a DAG.
            Most DAGs do not allow for bidirected edges, but it is an important piece of notation which Pearl and
            Shpitser use to denote graphs with latent variables.

        Parameters
        ----------
        X: str
            Intervention Variable

        Returns
        -------
        frozenset: A frozenset of frozensets

        Y: str
            Target Variable

        Examples
        --------
        >>> game1 = BayesianModel([('X', 'A'),
        ...                        ('A', 'Y'),
        ...                        ('A', 'B')])
        >>> inference = CausalInference(game1)
        >>> inference.get_all_backdoor_adjustment_sets("X", "Y")
        frozenset()

        References
        ----------
        "Causality: Models, Reasoning, and Inference", Judea Pearl (2000). p.79.
        """
        try:
            assert X in self.observed_variables
            assert Y in self.observed_variables
        except AssertionError:
            raise AssertionError("Make sure both X and Y are observed.")

        if self.is_valid_backdoor_adjustment_set(X, Y, Z=frozenset()):
            return frozenset()

        possible_adjustment_variables = (
            set(self.observed_variables)
            - {X}
            - {Y}
            - set(nx.descendants(self.model, X))
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

        return frozenset(valid_adjustment_sets)

    def is_valid_frontdoor_adjustment_set(self, X, Y, Z=None):
        """
        Test whether Z is a valid frontdoor adjustment set for estimating the causal impact of X on Y via the frontdoor
        adjustment formula.

        Parameters
        ----------
        X: str
            Intervention Variable

        Y: str
            Target Variable

        Z: set
            Adjustment variables

        Returns
        -------
        boolean: True if Z is a valid frontdoor adjustment set.
        """
        Z = _variable_or_iterable_to_set(Z)

        # 0. Get all directed paths from X to Y.  Don't check further if there aren't any.
        directed_paths = list(nx.all_simple_paths(self.model, X, Y))

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
        Identify possible sets of variables, Z, which satisify the front-door criterion relative to given X and Y.

        Z satisifies the front-door critierion if:
          (i)    Z intercepts all directed paths from X to Y
          (ii)   there is no backdoor path from X to Z
          (iii)  all back-door paths from Z to Y are blocked by X

        Returns
        -------
        frozenset: a frozenset of frozensets

        References
        ----------
        Causality: Models, Reasoning, and Inference, Judea Pearl (2000). p.82.
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

    def get_distribution(self):
        """
        Returns a string representing the factorized distribution implied by the CGM.
        """
        products = []
        for node in nx.topological_sort(self.model):
            if node in self.set_nodes:
                continue

            parents = list(self.model.predecessors(node))
            if not parents:
                p = f"P({node})"
            else:
                parents = [
                    f"do({n})" if n in self.set_nodes else str(n) for n in parents
                ]
                p = f"P({node}|{','.join(parents)})"
            products.append(p)
        return "".join(products)

    def simple_decision(self, adjustment_sets=[]):
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
        X: str
            Intervention Variable

        Y: str
            Target Variable

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
        float: The average treatment effect

        Examples
        --------
        >>> import pandas as pd
        >>> game1 = BayesianModel([('X', 'A'),
        ...                        ('A', 'Y'),
        ...                        ('A', 'B')])
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

        if isinstance(estimand_strategy, frozenset):
            adjustment_set = frozenset({estimand_strategy})
            assert self.is_valid_backdoor_adjustment_set(X, Y, Z=adjustment_set)
        elif estimand_strategy in ["smallest", "all"]:
            adjustment_sets = self.get_all_backdoor_adjustment_sets(X, Y)
            if estimand_strategy == "smallest":
                adjustment_sets = frozenset({self.simple_decision(adjustment_sets)})

        if estimator_type == "linear":
            self.estimator = LinearEstimator(self.model)

        ate = [
            self.estimator.fit(X=X, Y=Y, Z=s, data=data, **kwargs)._get_ate()
            for s in adjustment_sets
        ]
        return np.mean(ate)

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
        pgmpy.factor.DiscreteFactor: A factor object representing the joint distribution
            over the variables in `variables`.

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

        from pgmpy.inference import Inference

        if inference_algo == "ve":
            from pgmpy.inference import VariableElimination

            inference_algo = VariableElimination
        elif inference_algo == "bp":
            from pgmpy.inference import BeliefPropagation

            inference_algo = BeliefPropagation
        elif not isinstance(inference_algo, Inference):
            raise ValueError(
                f"inference_algo must be one of: 've', 'bp', or an instance of pgmpy.inference.Inference. Got: {inference_algo}"
            )

        # Step 2: Check if adjustment set is provided, otherwise calcualte it.
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

        # Step 3: If no do variable specified, do a normal probabilistic inference.
        if do == {}:
            return infer.query(variables, evidence, show_progress=False)

        # Step 3: Compute \sum_{z} p(variables | do, z) p(z)
        values = []
        p_z = infer.query(adjustment_set, evidence=evidence, show_progress=False)
        adj_states = [
            self.model.get_cpds(var).state_names[var] for var in adjustment_set
        ]

        if show_progress and SHOW_PROGRESS:
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

            if show_progress and SHOW_PROGRESS:
                pbar.update(1)

        return sum(values).normalize(inplace=False)
