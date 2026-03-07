from .ApproxInference import ApproxInference
from .base import Inference
from .CausalInference import CausalInference
from .dbn_inference import DBNInference
from .ExactInference import (
    BeliefPropagation,
    BeliefPropagationWithMessagePassing,
    VariableElimination,
)
from .mplp import Mplp

__all__ = [
    "Inference",
    "VariableElimination",
    "DBNInference",
    "BeliefPropagation",
    "BeliefPropagationWithMessagePassing",
    "BayesianModelSampling",
    "CausalInference",
    "ApproxInference",
    "GibbsSampling",
    "Mplp",
    "continuous",
]


def __getattr__(name):
    if name == "ApproxInference":
        from .ApproxInference import ApproxInference

        return ApproxInference
    elif name == "Inference":
        from .base import Inference

        return Inference
    elif name == "CausalInference":
        from .CausalInference import CausalInference

        return CausalInference
    elif name == "DBNInference":
        from .dbn_inference import DBNInference

        return DBNInference
    elif name == "BeliefPropagation":
        from .ExactInference import BeliefPropagation

        return BeliefPropagation
    elif name == "BeliefPropagationWithMessagePassing":
        from .ExactInference import BeliefPropagationWithMessagePassing

        return BeliefPropagationWithMessagePassing
    elif name == "VariableElimination":
        from .ExactInference import VariableElimination

        return VariableElimination
    elif name == "Mplp":
        from .mplp import Mplp

        return Mplp
    raise AttributeError(f"module {__name__} has no attribute {name}")
