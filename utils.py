#!/usr/bin/env python

"""
Utility functions for pgmpy, including example model definitions.
"""

import numpy as np

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


def get_example_model(model_name):
    """
    Get a predefined example model.

    Parameters
    ----------
    model_name : str
        Name of the example model to retrieve.
        Currently supported: 'cancer'

    Returns
    -------
    model : BayesianNetwork
        The requested example model with fitted CPDs.

    Raises
    ------
    ValueError
        If the requested model is not available.
    """

    if model_name.lower() == "cancer":
        return _get_cancer_model()
    else:
        raise ValueError(
            f"Example model '{model_name}' not available. "
            f"Supported models: ['cancer']"
        )


def _get_cancer_model():
    """
    Create the classic Cancer Bayesian Network example.

    Structure:
    Cancer → Smoking
    Cancer → XRay
    Smoking → Dyspnoea

    Returns
    -------
    model : BayesianNetwork
        Cancer model with fitted CPDs
    """

    # Define the network structure
    model = BayesianNetwork(
        [("Cancer", "Smoking"), ("Cancer", "XRay"), ("Smoking", "Dyspnoea")]
    )

    # Define CPDs (Conditional Probability Distributions)

    # Cancer (root node) - P(Cancer)
    cpd_cancer = TabularCPD(
        variable="Cancer",
        variable_card=2,
        values=[[0.99], [0.01]],  # P(Cancer=0)=0.99, P(Cancer=1)=0.01
    )

    # Smoking given Cancer - P(Smoking|Cancer)
    cpd_smoking = TabularCPD(
        variable="Smoking",
        variable_card=2,
        values=[
            [0.7, 0.2],  # P(Smoking=0|Cancer=0), P(Smoking=0|Cancer=1)
            [0.3, 0.8],  # P(Smoking=1|Cancer=0), P(Smoking=1|Cancer=1)
        ],
        evidence=["Cancer"],
        evidence_card=[2],
    )

    # XRay given Cancer - P(XRay|Cancer)
    cpd_xray = TabularCPD(
        variable="XRay",
        variable_card=2,
        values=[
            [0.95, 0.1],  # P(XRay=0|Cancer=0), P(XRay=0|Cancer=1)
            [0.05, 0.9],  # P(XRay=1|Cancer=0), P(XRay=1|Cancer=1)
        ],
        evidence=["Cancer"],
        evidence_card=[2],
    )

    # Dyspnoea given Smoking - P(Dyspnoea|Smoking)
    cpd_dyspnoea = TabularCPD(
        variable="Dyspnoea",
        variable_card=2,
        values=[
            [0.9, 0.3],  # P(Dyspnoea=0|Smoking=0), P(Dyspnoea=0|Smoking=1)
            [0.1, 0.7],  # P(Dyspnoea=1|Smoking=0), P(Dyspnoea=1|Smoking=1)
        ],
        evidence=["Smoking"],
        evidence_card=[2],
    )

    # Add CPDs to the model
    model.add_cpds(cpd_cancer, cpd_smoking, cpd_xray, cpd_dyspnoea)

    # Verify model consistency
    assert model.check_model()

    return model
