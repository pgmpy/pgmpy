"""Parameterizations: models of one variable given its parents, and the distributions they return.

This module is experimental: its API may change in any release without a deprecation period.
"""

from pgmpy.parameterization._base import BaseParameter
from pgmpy.parameterization.cpds import LinearGaussianCPD, TabularCPD
from pgmpy.utils._warnings import ExperimentalWarning, _warn_external

_warn_external(
    "pgmpy.parameterization is experimental: its API may change in any release without a deprecation period.",
    ExperimentalWarning,
)

__all__ = ["BaseParameter", "LinearGaussianCPD", "TabularCPD"]
