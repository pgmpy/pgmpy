#!/usr/bin/env python3
import numpy as np
from pgmpy.BayesianModel import BayesianModel
from pgmpy.MarkovModel import MarkovModel
from pgmpy.Factor import Factor


class Inference:
    """
    Base class for all Inference algorithms.
    """
    def __init__(self, model):
        self.variables = model.nodes()
        if isinstance(model, BayesianModel):
            self.factors = {}
            for node in model.nodes():
                cpd = model.get_cpd(node)
                factor = Factor(cpd.get_variables(), cpd.get_cardinality(), cpd.values)
                for var in cpd.get_variables():
                    try:
                        self.factors[var].append(factor)
                    except KeyError:
                        self.factors[var] = [factor]
                self.factors.append(factor)

        elif isinstance(model, MarkovModel):
            self.factors = {}
            for factor in model.get_factors():
                for var in factor.variables:
                    try:
                        self.factors[var].append(factor)
                    except KeyError:
                        self.factors[var] = [factor]
