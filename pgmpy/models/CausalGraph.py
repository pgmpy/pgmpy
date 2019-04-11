from collections import Iterable
from itertools import combinations, chain

import networkx as nx

from pgmpy.base import DAG


class CausalGraph(DAG):
    """
    The CausalGraph class is the primary interface for creating and querying CausalGraphs.  Extends the DAG class by
    giving support for identifying if an estimand given a model and a query.

    Importing from existing models is supported, though the model is assumed to either be a DAG or be compatiable with
    a DAG (such as a structured equation model).

    Parameters
    ----------
    ebunch : ebunch
        The set of directional edges given in the following form:
        >>> edges = [('X', 'A'),
        >>>          ('A', 'Y'),
        >>>          ('A', 'B')]
    latent_vars : iterable
        The set of unobserved variables.  Must be a subset of existing nodes. 
    """
    def __init__(self, ebunch=None, latent_vars=None, set_nodes=None):
        super(CausalGraph, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)

        self.latent_variables = _variable_or_iterable_to_set(latent_vars)
        self.set_nodes = _variable_or_iterable_to_set(set_nodes)
        self.observed_variables = frozenset(self.nodes()).difference(self.latent_variables)
        self.graph = self.to_undirected()

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.observed_variables)))
        return ("{}({})".format(self.__class__.__name__, variables))

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
        edges = [(a, b) for a, b in self.edges() if b != node]
        dag_do_x = CausalGraph(edges, latent_vars=self.latent_variables, set_nodes=set_nodes)
        # Make sure disconnected nodes aren't lost
        [dag_do_x.add_node(n) for n in self.nodes() if n not in dag_do_x.nodes()]
        return dag_do_x

    def _is_d_separated(self, X, Y, Z=None):
        return not self.is_active_trail(X, Y, observed=Z)

    def is_valid_backdoor_adjustment_set(self, X, Y, Z=None):
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
        """
        observed = [X]+list(Z) if Z else [X]
        return all([
            # Are all parents of X d-separated from Y given X and Z?
            self._is_d_separated(p, Y, Z=observed)
            for p in self.predecessors(X)
        ])

    def get_all_backdoor_adjustment_sets(self, X, Y):
        """
        Returns a list of all adjustment sets per the back-door criterion. 
        
        Pearl defined the back-door criterion this way in "Causality: Models, Reasoning, and Inference", p.79:
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
        Y: str
            Target Variable
        """
        assert X in self.observed_variables
        assert Y in self.observed_variables

        if self.is_valid_backdoor_adjustment_set(X, Y, Z=frozenset()):
            return frozenset()

        possible_adjustment_variables = (
            set(self.observed_variables)
            - {X} - {Y}
            - set(nx.descendants(self, X))
        )

        valid_adjustment_sets = []
        for s in _powerset(possible_adjustment_variables):
            super_of_complete = any([
                vs.intersection(set(s)) == vs
                for vs in valid_adjustment_sets
            ])
            if super_of_complete:
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
        """
        Z = _variable_or_iterable_to_set(Z)

        # 0. Get all directed paths from X to Y.  Don't check further if there aren't any.
        directed_paths = list(nx.all_simple_paths(self, X, Y))

        if directed_paths == []:
            return False

        # 1. Z intercepts all directed paths from X to Y
        unblocked_directed_paths = [
            path for path in
            directed_paths
            if not any(zz in path for zz in Z)
        ]

        if unblocked_directed_paths:
            return False

        # 2. there is no backdoor path from X to Z
        unblocked_backdoor_paths_X_Z = [
            zz
            for zz in Z
            if not self.is_valid_backdoor_adjustment_set(X, zz)
        ]

        if unblocked_backdoor_paths_X_Z:
            return False

        # 3. All back-door paths from Z to Y are blocked by X
        if not all([self.is_valid_backdoor_adjustment_set(zz, Y, X) for zz in Z]):
            return False

        return True

    def get_all_frontdoor_adjustment_sets(self, X, Y):
        """
        Identify possible sets of variables, Z, which satisify the front-door criterion relative to given X and Y.

        Per *Causality* p.82 by Pearl, Z satisifies the front-door critierion if:
          (i)    Z intercepts all directed paths from X to Y
          (ii)   there is no backdoor path from X to Z
          (iii)  all back-door paths from Z to Y are blocked by X
        """
        assert X in self.observed_variables
        assert Y in self.observed_variables

        possible_adjustment_variables = (
            set(self.observed_variables)
            - {X} - {Y}
        )

        valid_adjustment_sets = frozenset([           
                frozenset(s)
                for s in _powerset(possible_adjustment_variables)
                if self.is_valid_frontdoor_adjustment_set(X, Y, s)
            ])

        return valid_adjustment_sets

    def get_distribution(self):
        """
        Returns a string representing the factorized distribution implied by the CGM.
        """
        products = []
        for node in nx.topological_sort(self):
            if node in self.set_nodes:
                continue

            parents = list(self.predecessors(node))
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

    def from_BayesianModel(self):
        pass

    def from_SEM(self):
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
