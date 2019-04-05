#!/usr/bin/env python3

from itertools import combinations

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

    def check_active_backdoors(self, X, Y):
        """
        Checks each backdoor path to see if it's active.  Also provides (ideally) a complete set of nodes in the
        backdoor path so that we can induce a subgraph on it.

        Parameters
        ----------
        X : string
            The name of the variable we perform an intervention on.
        Y : string
            The name of the variable we want to measure given out intervention on X.
        """
        active_backdoor_nodes = set()
        bdroots = set(self.dag.get_parents(X))
        for node in bdroots:
            active_backdoor_nodes = active_backdoor_nodes.union(
                self.dag.active_trail_nodes(node, observed=X)[node])
        has_active_bdp = Y in active_backdoor_nodes
        bdg = self.dag.subgraph(active_backdoor_nodes)
        return has_active_bdp, bdg, bdroots

    def get_possible_deconfounders(self, possible_nodes, maxdepth=None):
        """
        Generates the set of possible combinations of deconfounding variables up to a certain depth.

        Parameters
        ----------
        possible_nodes : set
            The set of nodes which we will draw our deconfounding sets from.
        """
        possible_combinations = []
        if maxdepth is None:
            for i in range(1, len(possible_nodes)+1):
                possible_combinations += combinations(possible_nodes, i)
        else:
            # Just in case the depth is greater than what's possible, we norm the term to be the number of possible
            # nodes at most.
            maxdepth = min(len(possible_nodes), maxdepth)
            for i in range(1, maxdepth+1):
                possible_combinations += combinations(possible_nodes, i)
        return possible_combinations

    def check_deconfounders(self, bdgraph, bdroots, X, Y, maxdepth=None):
        """This function explores each possible deconfounding set and determines if it deactivates all backdoor paths.

        We will want this to take into account observed/unobserved variables.

        Parameters
        ----------
        bdgraph : CausalGraph
            The subgraph induced on all nodes present in the backdoor paths from x to y.
        bdroots : set
            The parents of x which are also the roots of all backdoor paths.
        X : string
            The name of the variable we perform an intervention on.
        Y : string
            The name of the variable we want to measure given out intervention on X.
        maxdepth : int
            The maximum number of variables in a set of deconfounders. This should be larger than the number of
            possible variables, but error catching will prvent it from being too large.
        """
        nodes = set(bdgraph.nodes())
        complete_sets = set()
        possible_deconfounders = self.get_possible_deconfounders(
            nodes.difference({Y}), maxdepth=maxdepth)
        for deconfounder in possible_deconfounders:
            seenbefore = False
            for cs in complete_sets:
                overlap = cs.intersection(set(deconfounder))
                if overlap == cs:
                    seenbefore = True
            if seenbefore:
                continue
            active = {}
            for bd in bdroots:
                a = int(Y in bdgraph.active_trail_nodes(bd, observed=deconfounder)[bd])
                active[bd] = active.get(bd, 0) + a
            still_active = sum([val > 0 for val in active.values()])
            if still_active == 0:
                complete_sets.add(frozenset(deconfounder))
        return complete_sets

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

    def get_backdoor_deconfounders(self, X, Y, maxdepth=None):
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
        has_active_bdp, bdg, bdroots = self.check_active_backdoors(X, Y)
        if has_active_bdp:
            deconfounding_set = self.check_deconfounders(bdg, bdroots, X, Y, maxdepth=maxdepth)
        else:
            deconfounding_set = set()
        return deconfounding_set

    def get_frontdoor_deconfounders(self, X, Y):
        """
        Identify possible sets of variables, Z, which satisify the front-door criterion relative to given X and Y. 

        Per *Causality* by Pearl, the Z satisifies the front-door critierion if:
          (i)    Z intercepts all directed paths from X to Y
          (ii)   there is no back-door path from X to Z
          (iii)  all back-door paths from Z to Y are blocked by X       
        """
        pass
