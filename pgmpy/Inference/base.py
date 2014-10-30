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
        if isinstance(model, BayesianModel):
            factors = []
            for node in model.nodes():
                cpd = model.get_cpd(node)
                factor = Factor(cpd.get_variables(), cpd.get_cardinality(), cpd.values)
                factors.append(factor)
        if isinstance(model, MarkovModel):
            factors = model.get_factors()

