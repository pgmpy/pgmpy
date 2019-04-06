#!/usr/bin/env python3

from collections import Iterable
from itertools import combinations, chain

import networkx as nx

from pgmpy.inference import Inference
from pgmpy.models.BayesianModel import BayesianModel


class CausalInference(Inference):
    """
    This is an inference class for performing Causal Inference over Bayesian Networks or Strucural Equation Models.

    This class will accept queries of the form: P(Y | do(X)) and utilize it's methods to provide an estimand which:
     * Identifies adjustment variables
     * Backdoor Adjustment
     * Front Door Adjustment
     * Instrumental Variable Adjustment

    Parameters
    ----------
    model : BayesianModel or SEM class
        The model that we'll perform inference over.
    latent_vars : set or list[node:str] or None
        A list (or set/tuple) of nodes in the Bayesian Network that are unobserved.
    set_nodes : list[node:str] or None
        A list (or set/tuple) of nodes in the Bayesian Network which have been set to a specific value per the
        do-operator.

    References
    ----------
    'Causality: Models, Reasoning, and Inference' - Judea Pearl (2000)

    Many thanks to @ijmbarr for their implementation of Causal Graphical models available. It served as a valuable
    reference. Available on GitHub: https://github.com/ijmbarr/causalgraphicalmodels
    """
    def __init__(self, model, latent_vars=None, set_nodes=None):
        # Leaving this out for now.  Inference seems to be requiring CPDs to be associated with each factor, which
        # isn't actually a requirement I want to enforce.
        # super(CausalInference, self).__init__(model)
        if set_nodes is None:
            self.set_nodes = frozenset()
        else:
            self.set_nodes = frozenset(set_nodes)

        if latent_vars is None:
            self.latent_vars = frozenset()
        else:
            self.latent_vars = frozenset(latent_vars)

        assert isinstance(model, BayesianModel)
        self.dag = model
        self.unobserved_variables = frozenset(self.latent_vars)
        self.observed_variables = frozenset(self.dag.nodes()).difference(self.unobserved_variables)

        for set_node in self.set_nodes:
            # Nodes are set with the do-operator and thus cannont have parents
            assert not nx.ancestors(self.dag, set_node)

        self.graph = self.dag.to_undirected()

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.observed_variables)))
        return ("{classname}({vars})"
                .format(classname=self.__class__.__name__,
                        vars=variables))

    def get_distribution(self):
        """
        Returns a string representing the factorized distribution implied by
        the CGM.
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

    def do(self, node):
        """
        Applies the do operator to the graph and returns a new Inference object with the transformed graph.

        Defined by Pearl in Causality on p. 70, the do-operator, do(X = x) has the effect of removing all edges from
        the parents of X and setting X to the given value x.

        Parameters
        ----------
        node : string
            The name of the node to apply the do-operator to.
        """
        assert node in self.observed_variables
        set_nodes = self.set_nodes | frozenset([node])
        edges = [(a, b) for a, b in self.dag.edges() if b != node]
        dag_do_x = BayesianModel(edges)
        for n in self.dag.nodes():
            # Make sure isolated nodes aren't lost when new graph is created.
            if n not in dag_do_x.nodes():
                dag_do_x.add_node(n)
        return CausalInference(model=dag_do_x, latent_vars=self.unobserved_variables, set_nodes=set_nodes)

    def is_d_separated(self, X, Y, Z=None):
        return self.dag.is_active_trail(X, Y, observed=Z)

    def _check_d_separation(self, path, Z=None):
        """
        This function and ._classify_three_structure are a little duplicative with .active_trail_nodes in the DAG
        class.  However, because we're leveraging nx.all_simple_paths we want a version of this function which
        can operate directly on paths.

        Parameters
        ----------
        path : networkx path object
            This will typically be an output from nx.all_simple_paths
        """
        Z = _variable_or_iterable_to_set(Z)

        if len(path) < 3:
            return False

        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            structure = self._classify_three_structure(a, b, c)

            if structure in ("chain", "fork") and b in Z:
                return True

            if structure == "collider":
                descendants = (nx.descendants(self.dag, b) | {b})
                if not descendants & set(Z):
                    return True

        return False

    def _classify_three_structure(self, a, b, c):
        """
        Classify three structure as a chain, fork or collider.
        """
        if self.dag.has_edge(a, b) and self.dag.has_edge(b, c):
            return "chain"

        if self.dag.has_edge(c, b) and self.dag.has_edge(b, a):
            return "chain"

        if self.dag.has_edge(a, b) and self.dag.has_edge(c, b):
            return "collider"

        if self.dag.has_edge(b, a) and self.dag.has_edge(b, c):
            return "fork"

        raise ValueError("Unsure how to classify ({},{},{})".format(a, b, c))

    def get_all_backdoor_paths(self, X, Y):
        """
        Returns all backdoor paths from X to Y.
        """
        return [
            path
            for path in nx.all_simple_paths(self.graph, X, Y)
            if len(path) > 2
            and path[1] in self.dag.predecessors(X)
        ]

    def is_valid_backdoor_adjustment_set(self, X, Y, Z):
        """
        Test whether Z is a valid backdoor deconfoudning set for estimating the causal impact of X on Y.

        Parameters
        ----------
        X: str
            Intervention Variable
        Y: str
            Target Variable
        Z: str or set[str]
            Adjustment variables
        """
        z = _variable_or_iterable_to_set(Z)

        assert X in self.observed_variables
        assert Y in self.observed_variables
        assert X not in z
        assert Y not in z

        if any([zz in nx.descendants(self.dag, X) for zz in Z]):
            return False

        unblocked_backdoor_paths = [
            path
            for path in self.get_all_backdoor_paths(X, Y)
            if not self._check_d_separation(path, Z)
        ]

        if unblocked_backdoor_paths:
            return False

        return True

    def _has_active_backdoors(self, X, Y):
        return all([
            not self.dag.is_active_trail(p, Y, observed=X)
            for p in self.dag.predecessors(X)
        ])

    def get_all_backdoor_deconfounders(self, X, Y):
        """
        Return a list of all possible of deconfounding sets by backdoor adjustment per Pearl, "Causality: Models,
        Reasoning, and Inference", p.79 up to sets of size maxdepth.

        TODO:
          * Backdoors are great, but the most general things we could implement would be Ilya Shpitser's ID and
            IDC algorithms. See [his Ph.D. thesis for a full explanation]
            (https://ftp.cs.ucla.edu/pub/stat_ser/shpitser-thesis.pdf). After doing a little reading it is clear
            that we do not need to immediatly implement this.  However, in order for us to truly account for
            unobserved variables, we will need not only these algorithms, but a more general implementation of a DAG.
            Most DAGs do not allow for bidirected edges, but it is an important piece of notation which Pearl and
            Shpitser use to denote graphs with latent variables.  So we would probably need to add a new model class
            in order to fully capture these models.
          * However, way prior to that we should just implement Backdoor, Frontdoor and Instrumental Variable
            adjustment. This combination of tools is very powerful by itself and if we simply assume that all varibles
            in the graph are observed then we never have to work about non-identifiable graphs.
          * Users probably don't want to choose their estimand themselves, but actually want a default decision rule
            implement.  Likely something like, "choose the estimand with the smallest number of observed factors."
            Or we could just fit an estimator to all of them.  This would be interesting in that it gives a natural
            opportunity for robustness checks.

        Parameters
        ----------
        X : string
            The name of the variable we perform an intervention on.
        Y : string
            The name of the variable we want to measure given out intervention on X.
        """
        assert X in self.observed_variables
        assert Y in self.observed_variables

        if self._has_active_backdoors(X=X, Y=Y):
            return frozenset([])

        possible_adjustment_variables = (
            set(self.observed_variables)
            - {X} - {Y}
            - set(nx.descendants(self.dag, X))
        )

        valid_adjustment_sets = []
        for s in _powerset(possible_adjustment_variables):
            if sum([
                vs.intersection(set(s)) == vs
                for vs in valid_adjustment_sets
            ]) > 0:
                continue
            if self.is_valid_backdoor_adjustment_set(X, Y, s):
                valid_adjustment_sets.append(frozenset(s))

        return frozenset(valid_adjustment_sets)

    def get_frontdoor_deconfounders(self, X, Y):
        """
        Identify possible sets of variables, Z, which satisify the front-door criterion relative to given X and Y.

        Per *Causality* by Pearl, the Z satisifies the front-door critierion if:
          (i)    Z intercepts all directed paths from X to Y
          (ii)   there is no back-door path from X to Z
          (iii)  all back-door paths from Z to Y are blocked by X
        """
        pass


def _variable_or_iterable_to_set(x):
    """
    Convert variable or iterable x to a frozenset.

    If x is None, returns the empty set.

    Parameters
    ---------
    x : None, str or Iterable[str]
    """
    if x is None:
        return frozenset([])

    if isinstance(x, str):
        return frozenset([x])

    if not isinstance(x, Iterable) or not all(isinstance(xx, str) for xx in x):
        raise ValueError(
            "{} is expected to be either a string or an iterable of strings"
            .format(x))

    return frozenset(x)


def _powerset(iterable):
    """
    https://docs.python.org/3/library/itertools.html#recipes
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
