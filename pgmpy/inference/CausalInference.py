import numpy as np
import networkx as nx

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.estimators.LinearModel import LinearEstimator
from pgmpy.utils.sets import _powerset, _variable_or_iterable_to_set


class CausalInference(object):
    """
    This is an inference class for performing Causal Inference over Bayesian Networks or Strucural Equation Models.

    This class will accept queries of the form: P(Y | do(X)) and utilize it's methods to provide an estimand which:
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
    >>> game = CausalGraph([('X', 'A'),
                            ('A', 'Y'),
                            ('A', 'B')])
    Load the graph into the CausalInference object to make causal queries.
    >>> from pgmpy.inference.causal_inferece import CausalInference
    >>> inference = CausalInference(game)
    >>> inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
    >>> inference.get_all_frontdoor_adjustment_sets(X="X", Y="Y")

    References
    ----------
    'Causality: Models, Reasoning, and Inference' - Judea Pearl (2000)

    Many thanks to @ijmbarr for their implementation of Causal Graphical models available. It served as an invaluable
    reference. Available on GitHub: https://github.com/ijmbarr/causalgraphicalmodels
    """

    def __init__(self, model, latent_vars=None, set_nodes=None):
        if not isinstance(model, BayesianModel):
            raise NotImplementedError(
                "Causal Inference is only implemented for BayesianModels at this time."
            )
        self.dag = model
        self.graph = self.dag.to_undirected()
        self.latent_variables = _variable_or_iterable_to_set(latent_vars)
        self.set_nodes = _variable_or_iterable_to_set(set_nodes)
        self.observed_variables = frozenset(self.dag.nodes()).difference(
            self.latent_variables
        )

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.observed_variables)))
        return "{classname}({vars})".format(self.__class__.__name__, variables)

    def _is_d_separated(self, X, Y, Z=None):
        return not self.dag.is_active_trail(X, Y, observed=Z)

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
                                   ('A', 'Y'),
                                   ('A', 'B')])
        >>> inference = CausalInference(game1)
        >>> inference.is_valid_backdoor_adjustment_set("X", "Y")
        True
        """
        Z_ = list(Z)
        observed = [X] + Z_
        parents_d_sep = []
        for p in self.dag.predecessors(X):
            parents_d_sep.append(self._is_d_separated(p, Y, Z=observed))
        return all(parents_d_sep)

    def get_all_backdoor_adjustment_sets(self, X, Y):
        """
        Returns a list of all adjustment sets per the back-door criterion.

        A set of variables Z satisfies the back-door criterion relative to an ordered pair ofvariabies (Xi, Xj) in a DAG G if:
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
                                   ('A', 'Y'),
                                   ('A', 'B')])
        >>> inference = CausalInference(game1)
        >>> inference.get_all_backdoor_adjustment_sets("X", "Y")
        frozenset([])

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
        for node in nx.topological_sort(self.dag):
            if node in self.set_nodes:
                continue

            parents = list(self.dag.predecessors(node))
            if not parents:
                p = "P({})".format(node)
            else:
                parents = [
                    "do({})".format(n) if n in self.set_nodes else str(n)
                    for n in parents
                ]
                p = "P({}|{})".format(node, ",".join(parents))
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
        **kwargs
    ):
        """
        Estimate the average treatment effect (ATE) of X on Y.

        Parameters
        ----------
        X: str
            Intervention Variable

        Y: str
            Target Variable

        data: pandas DataFrame
            All observed data for this Bayesian Network.

        estimand_strategy: str or frozenset
            Either specify a specific backdoor adjustment set or a strategy.  The available options are:
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
        >>> game1 = BayesianModel([('X', 'A'),
                                   ('A', 'Y'),
                                   ('A', 'B')])
        >>> data = pd.DataFrame(np.random.randint(2, size=(1000, 4)), columns=['X', 'A', 'B', 'Y'])
        >>> inference = CausalInference(model=game1)
        >>> inference.estimate_ate("X", "Y", data=data, estimator_type="linear")
        """
        valid_estimators = ["linear"]
        try:
            assert estimator_type in valid_estimators
        except AssertionError:
            print(
                "{} if not a valid estimator_type.  Please select from {}".format(
                    estimator_type, valid_estimators
                )
            )

        if isinstance(estimand_strategy, frozenset):
            adjustment_set = frozenset({estimand_strategy})
            assert self.is_valid_backdoor_adjustment_set(X, Y, Z=adjustment_set)
        elif estimand_strategy in ["smallest", "all"]:
            adjustment_sets = self.get_all_backdoor_adjustment_sets(X, Y)
            if estimand_strategy == "smallest":
                adjustment_sets = frozenset({self.simple_decision(adjustment_sets)})

        if estimator_type == "linear":
            self.estimator = LinearEstimator(self.dag)

        ate = [
            self.estimator.fit(X=X, Y=Y, Z=s, data=data, **kwargs)._get_ate()
            for s in adjustment_sets
        ]
        return np.mean(ate)
