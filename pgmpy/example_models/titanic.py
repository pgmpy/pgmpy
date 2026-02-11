"""
Titanic Bayesian Network.

Source: bnRep R package (https://github.com/manueleleonelli/bnRep)
Model: Titanic

This example is based on the Titanic Bayesian network distributed
with the bnRep repository.
"""

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def get_model():
    model = DiscreteBayesianNetwork([
        ("Class", "Survived"),
        ("Sex", "Survived"),
    ])

    # Class prior
    cpd_class = TabularCPD(
        variable="Class",
        variable_card=4,
        values=np.array([[0.25], [0.25], [0.25], [0.25]]),
        state_names={"Class": ["1st", "2nd", "3rd", "Crew"]},
    )

    # Sex prior
    cpd_sex = TabularCPD(
        variable="Sex",
        variable_card=2,
        values=np.array([[0.5], [0.5]]),
        state_names={"Sex": ["Male", "Female"]},
    )

    # Survived CPD (4 Class × 2 Sex = 8 columns)
    cpd_survived = TabularCPD(
        variable="Survived",
        variable_card=2,
        values=np.array([
            # Survived = Yes
            [0.62, 0.47, 0.24, 0.11, 0.88, 0.73, 0.14, 0.86],
            # Survived = No
            [0.38, 0.53, 0.76, 0.89, 0.12, 0.27, 0.86, 0.14],
        ]),
        evidence=["Class", "Sex"],
        evidence_card=[4, 2],
        state_names={
            "Survived": ["Yes", "No"],
            "Class": ["1st", "2nd", "3rd", "Crew"],
            "Sex": ["Male", "Female"],
        },
    )

    model.add_cpds(cpd_class, cpd_sex, cpd_survived)

    model.check_model()

    return model

