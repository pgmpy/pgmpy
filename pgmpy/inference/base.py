#!/usr/bin/env python3

from collections import defaultdict
from itertools import chain

from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel
from pgmpy.models import FactorGraph
from pgmpy.models import JunctionTree
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


class Inference(object):
    """
    Base class for all inference algorithms.

    Converts BayesianModel and MarkovModel to a uniform representation so that inference
    algorithms can be applied. Also it checks if all the associated CPDs / Factors are
    consistent with the model.

    Initialize inference for a model.

    Parameters
    ----------
    model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
        model for which to initialize the inference object.

    Examples
    --------
    >>> from pgmpy.inference import Inference
    >>> from pgmpy.models import BayesianModel
    >>> from pgmpy.factors.discrete import TabularCPD
    >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
    >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
    >>> intel_cpd = TabularCPD('intel', 2, [[0.3], [0.7]])
    >>> grade_cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1],
    ...                                     [0.1, 0.1, 0.1, 0.1],
    ...                                     [0.8, 0.8, 0.8, 0.8]],
    ...                        evidence=['diff', 'intel'], evidence_card=[2, 2])
    >>> student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
    >>> model = Inference(student)

    >>> from pgmpy.models import MarkovModel
    >>> from pgmpy.factors.discrete import DiscreteFactor
    >>> import numpy as np
    >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
    ...                        ('Charles', 'Debbie'), ('Debbie', 'Alice')])
    >>> factor_a_b = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2],
    ...                             values=np.random.rand(4))
    >>> factor_b_c = DiscreteFactor(['Bob', 'Charles'], cardinality=[2, 2],
    ...                             values=np.random.rand(4))
    >>> factor_c_d = DiscreteFactor(['Charles', 'Debbie'], cardinality=[2, 2],
    ...                             values=np.random.rand(4))
    >>> factor_d_a = DiscreteFactor(['Debbie', 'Alice'], cardinality=[2, 2],
    ...                             values=np.random.rand(4))
    >>> student.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)
    >>> model = Inference(student)
    """

    def __init__(self, model):
        self.model = model
        model.check_model()

    def _initialize_structures(self):
        """
        Initializes all the data structures which will
        later be used by the inference algorithms.
        """
        if isinstance(self.model, JunctionTree):
            self.variables = set(chain(*self.model.nodes()))
        else:
            self.variables = self.model.nodes()

        self.cardinality = {}
        self.factors = defaultdict(list)

        if isinstance(self.model, BayesianModel):
            self.state_names_map = {}
            for node in self.model.nodes():
                cpd = self.model.get_cpds(node)
                if isinstance(cpd, TabularCPD):
                    self.cardinality[node] = cpd.variable_card
                    cpd = cpd.to_factor()
                for var in cpd.scope():
                    self.factors[var].append(cpd)
                self.state_names_map.update(cpd.no_to_name)

        elif isinstance(self.model, (MarkovModel, FactorGraph, JunctionTree)):
            self.cardinality = self.model.get_cardinality()

            for factor in self.model.get_factors():
                for var in factor.variables:
                    self.factors[var].append(factor)

        elif isinstance(self.model, DynamicBayesianNetwork):
            self.start_bayesian_model = BayesianModel(self.model.get_intra_edges(0))
            self.start_bayesian_model.add_cpds(*self.model.get_cpds(time_slice=0))
            cpd_inter = [
                self.model.get_cpds(node) for node in self.model.get_interface_nodes(1)
            ]
            self.interface_nodes = self.model.get_interface_nodes(0)
            self.one_and_half_model = BayesianModel(
                self.model.get_inter_edges() + self.model.get_intra_edges(1)
            )
            self.one_and_half_model.add_cpds(
                *(self.model.get_cpds(time_slice=1) + cpd_inter)
            )

    def _prune_bayesian_model(self, variables, evidence):
        """
        Prunes unnecessary nodes from the model to optimize the computation.

        Parameters
        ----------
        variables: list
            The variables on which the query is done i.e. the variables whose
            values we are interested in.

        evidence: dict (default: None)
            The variables whose values we know. The values can be specified as
            {variable: state}.

        Returns
        -------
        instance of pgmpy.models.BayesianModel: The pruned model.

        Examples
        --------
        >>>
        >>>

        References
        ----------
        [1] Baker, M., & Boult, T. E. (2013). Pruning Bayesian networks for efficient computation. arXiv preprint arXiv:1304.1112.
        """
        bn = self.model.copy()
        evidence = {} if evidence is None else evidence
        variables = list(self.model.nodes()) if len(variables) == 0 else list(variables)

        # Step 1: Remove all the variables that are d-separated from `variables` when conditioned
        #         on `evidence`
        d_connected = bn.active_trail_nodes(
            variables=variables, observed=list(evidence.keys())
        )
        d_connected = set.union(*d_connected.values()).union(evidence.keys())
        bn = bn.subgraph(d_connected)
        evidence = {var: state for var, state in evidence.items() if var in d_connected}

        # Step 2: Reduce the model to ancestral graph of [`variables` + `evidence`]
        bn = bn.get_ancestral_graph(list(variables) + list(evidence.keys()))

        # Step 3: Since all the CPDs are lost, add them back.
        bn.add_cpds(*[self.model.get_cpds(var) for var in bn.nodes()])

        return bn, evidence
