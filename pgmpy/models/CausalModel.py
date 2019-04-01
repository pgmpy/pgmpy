#!/usr/bin/env python3

import itertools
from itertools import combinations
from collections import defaultdict
import logging
from operator import mul

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.models.BayesianModel import BayesianModel


class CausalModel(BayesianModel):
    """
    Base class for causal models.

    A models stores nodes and edges with conditional probability
    distribution (cpd) and other attributes.

    models hold directed edges.  Self loops are not allowed neither
    multiple (parallel) edges.

    Nodes can be any hashable python object.

    Edges are represented as links between nodes.

    Parameters
    ----------
    data : input graph
        Data to initialize graph.  If data=None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.

    This class extends BayesianModel will some of the essential features
    in Causal models including:
     * Identifying adjustment variables
     * Backdoor Adjustment
     * Front Door Adjustment
    """
    def __init__(self, ebunch=None):
        super(CausalModel, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)

    def check_active_backdoors(self, treatment, outcome):
        """
        Checks each backdoor path to see if it's active.  Also
        proves a complete set of nodes in the backdoor path so that
        we can induce a subgraph on it.

        Parameters
        ----------
        treatment : string
            The name of the varaible we want to consider as the treatment.
            We probably will want to eventually meausure the causal effect
            of the treatment on the outcome.
        outcomes : string
            The name of the variable we want to treat as the outcome.
        """
        active_backdoor_nodes = set()
        bdroots = set(self.graph.get_parents(treatment))
        for node in bdroots:
            active_backdoor_nodes = active_backdoor_nodes.union(
                self.graph.active_trail_nodes(node, observed=treatment)[node])
        has_active_bdp = outcome in active_backdoor_nodes
        bdg = self.graph.subgraph(active_backdoor_nodes)
        return has_active_bdp, bdg, bdroots

    def get_possible_deconfounders(self, possible_nodes, maxdepth=None):
        """
        Generates the set of possible combinations of deconfounding variables
        up to a certain depth.

        Parameters
        ----------
        possible_nodes : set
            The set of nodes which we will draw our deconfounding sets from.
        outcomes : string
            The name of the variable we want to treat as the outcome.
        """
        possible_combinations = []
        if maxdepth is None:
            for i in range(1, len(possible_nodes)+1):
                possible_combinations += combinations(possible_nodes, i)
        else:
            # Just in case the depth is greater than what's possible, we
            # norm the term to be the number of possible nodes at most.
            maxdepth = min(len(possible_nodes), maxdepth)
            for i in range(1, maxdepth+1):
                possible_combinations += combinations(possible_nodes, i)
        return possible_combinations

    def check_deconfounders(self, bdgraph, bdroots, treatment, outcome, maxdepth=None):
        """This function explores each possible deconfounding set and determines
        if it deactivates all backdoor paths.

        We will want this to take into account observed/unobserved variables.

        Parameters
        ----------
        bdgraph : CausalGraph
            The subgraph induced on all nodes present in the backdoor paths
            from the treatment to the outcome variable.
        bdroots : set
            The parents of the treatment variable which are also the roots of
            all backdoor paths.
        treatment : string
            The name of the varaible we want to consider as the treatment.
            We probably will want to eventually meausure the causal effect
            of the treatment on the outcome.
        outcomes : string
            The name of the variable we want to treat as the outcome.
        maxdepth : int
            The maximum number of variables in a set of deconfounders. This
            should be larger than the number of possible variables, but error
            catching will prvent it from being too large.
        """
        nodes = set(bdgraph.nodes)
        complete = []
        possible_deconfounders = self.get_possible_deconfounders(
            nodes.difference({'Y'}), maxdepth=maxdepth)
        for deconfounder in possible_deconfounders:
            active = {}
            for bd in bdroots:
                a = int("Y" in bdgraph.active_trail_nodes(bd, observed=deconfounder)[bd])
                active[bd] = active.get(bd, 0) + a
            still_active = sum([val > 0 for val in active.values()])
            if still_active == 0:
                complete.append(deconfounder)
        return complete

    def get_deconfounders(self, treatment, outcome, maxdepth=None):
        """
        Return a list of all possible of deconfounding sets by backdoor
        adjustment per Pearl, "Causality: Models, Reasoning, and Inference", 
        p.79 up to sets of size maxdepth.

        Parameters
        ----------
        treatment
        """
        has_active_bdp, bdg, bdroots = self.check_active_backdoors(self.graph)
        if has_active_bdp:
            deconfounding_set = self.check_deconfounders(bdg, bdroots, maxdepth=maxdepth)
        else:
            deconfounding_set = []
        return deconfounding_set
