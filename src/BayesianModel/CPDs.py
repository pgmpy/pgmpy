#!/usr/bin/env python3
"""Contains the different formats of CPDs used in PGM"""

import numpy as np


class TabularCPD():
    """Represents the CPD of a node in tabular form"""
    def __init__(self, cpd):
        self.table = np.array(cpd)
