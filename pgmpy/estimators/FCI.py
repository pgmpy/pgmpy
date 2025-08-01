"""
Implementation of the Fast Causal Inference (FCI) algorithm."""

from itertools import chain, combinations, permutations
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Collection,
    Tuple,
    Union,
)

import networkx as nx
import pandas as pd


from pgmpy.estimators.PC import PC
from pgmpy.estimators.CITests import get_callable_ci_test
from pgmpy.base import ADMG, DAG, UndirectedGraph
