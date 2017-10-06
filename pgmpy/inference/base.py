#!/usr/bin/env python3

from collections import defaultdict
from itertools import chain

from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel
from pgmpy.models import FactorGraph
from pgmpy.models import JunctionTree
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.utils import StateNameInit
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
    >>> diff_cpd = TabularCPD('diff', 2, [[0.2, 0.8]])
    >>> intel_cpd = TabularCPD('intel', 2, [[0.3, 0.7]])
    >>> grade_cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1],
    ...                                     [0.1, 0.1, 0.1, 0.1],
    ...                                     [0.8, 0.8, 0.8, 0.8]],
    ...                        evidence=['diff', 'intel'], evidence_card=[2, 2])
    >>> student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
    >>> model = Inference(student)

    >>> from pgmpy.models import MarkovModel
    >>> from pgmpy.factors import DiscreteFactor
    >>> import numpy as np
    >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
    ...                        ('Charles', 'Debbie'), ('Debbie', 'Alice')])
    >>> factor_a_b = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2], value=np.random.rand(4))
    >>> factor_b_c = DiscreteFactor(['Bob', 'Charles'], cardinality=[2, 2], value=np.random.rand(4))
    >>> factor_c_d = DiscreteFactor(['Charles', 'Debbie'], cardinality=[2, 2], value=np.random.rand(4))
    >>> factor_d_a = DiscreteFactor(['Debbie', 'Alice'], cardinality=[2, 2], value=np.random.rand(4))
    >>> student.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)
    >>> model = Inference(student)
    """

    @StateNameInit()
    def __init__(self, model):
        self.model = model
        model.check_model()

        if isinstance(model, JunctionTree):
            self.variables = set(chain(*model.nodes()))
        else:
            self.variables = model.nodes()

        self.cardinality = {}
        self.factors = defaultdict(list)

        if isinstance(model, BayesianModel):
            for node in model.nodes():
                cpd = model.get_cpds(node)
                if isinstance(cpd, TabularCPD):
                    self.cardinality[node] = cpd.variable_card
                    cpd = cpd.to_factor()
                for var in cpd.scope():
                    self.factors[var].append(cpd)

        elif isinstance(model, (MarkovModel, FactorGraph, JunctionTree)):
            self.cardinality = model.get_cardinality()

            for factor in model.get_factors():
                for var in factor.variables:
                    self.factors[var].append(factor)

        elif isinstance(model, DynamicBayesianNetwork):
            self.start_bayesian_model = BayesianModel(model.get_intra_edges(0))
            self.start_bayesian_model.add_cpds(*model.get_cpds(time_slice=0))
            cpd_inter = [model.get_cpds(node) for node in model.get_interface_nodes(1)]
            self.interface_nodes = model.get_interface_nodes(0)
            self.one_and_half_model = BayesianModel(model.get_inter_edges() + model.get_intra_edges(1))
            self.one_and_half_model.add_cpds(*(model.get_cpds(time_slice=1) + cpd_inter))
