#!/usr/bin/env python3

from itertools import combinations

from pgmpy.inference import Inference


class CausalInference(Inference):
    """
    This is an inference class for performing Causal Inference over Bayesian Networks or Strucural Equation Models.

    This class will accept queries of the form: P(Y | do(X)) and utilize it's method to provide an estimand which
    executes :
     * Identifying adjustment variables
     * Backdoor Adjustment
     * Front Door Adjustment

    Parameters
    ----------
    model : instance of pgmpy Bayesian Network or SEM class
        The model that we'll perform inference over.
    """
    def __init__(self, model=None):
        # Leaving this out for now.  Inference seems to be requiring CPDs to be associated with each factor, which
        # isn't actually a requirement I want to enforce.
        # super(CausalInference, self).__init__(model)
        self.model = model

    def check_active_backdoors(self, X, Y):
        """
        Checks each backdoor path to see if it's active.  Also provides (ideally) a complete set of nodes in the
        backdoor path so that we can induce a subgraph on it.

        TODO:
          * Our current method for getting the set of nodes in the backdoor path uses the .active_trail_nodes
            method from the graph.

        Parameters
        ----------
        X : string
            The name of the variable we perform an intervention on.
        Y : string
            The name of the variable we want to measure given out intervention on X.
        """
        active_backdoor_nodes = set()
        bdroots = set(self.model.get_parents(X))
        for node in bdroots:
            active_backdoor_nodes = active_backdoor_nodes.union(
                self.model.active_trail_nodes(node, observed=X)[node])
        has_active_bdp = Y in active_backdoor_nodes
        bdg = self.model.subgraph(active_backdoor_nodes)
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

    def get_deconfounders(self, X, Y, maxdepth=None):
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
