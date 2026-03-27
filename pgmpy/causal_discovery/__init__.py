from pgmpy.causal_discovery.ExpertKnowledge import ExpertKnowledge
from pgmpy.causal_discovery.GES import GES
from pgmpy.causal_discovery.HillClimbSearch import HillClimbSearch
from pgmpy.causal_discovery.PC import PC

__all__ = [
    "CASTLE",
    "ExpertKnowledge",
    "GES",
    "HillClimbSearch",
    "PC",
]


def __getattr__(name):
    if name == "CASTLE":
        from pgmpy.causal_discovery.castle import CASTLE

        return CASTLE
    raise AttributeError(f"module 'pgmpy.causal_discovery' has no attribute {name!r}")
