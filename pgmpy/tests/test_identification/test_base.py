import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.identification import BaseIdentification


@pytest.fixture
def cg():
    edges = [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")]
    roles = {"exposure": "X", "outcome": "Y"}
    return DAG(ebunch=edges, roles=roles)


class DummyIdentification(BaseIdentification):
    """Sorts non-exposure and non-outcome nodes in the graph and assigns the
    first or the last one as adjustment node depending on the `variant`.
    """

    def __init__(self, variant=None):
        self.variant = variant
        self.supported_graph_types = (DAG, PDAG)

    def _identify(self, causal_graph):
        if self.variant == "first":
            adjustment_node = sorted(
                set(causal_graph.nodes())
                - set(
                    causal_graph.get_role("exposure") + causal_graph.get_role("outcome")
                )
            )[0]
            return causal_graph.with_role("adjustment", [adjustment_node]), True
        elif self.variant == "last":
            adjustment_node = sorted(
                set(causal_graph.nodes())
                - set(
                    causal_graph.get_role("exposure") + causal_graph.get_role("outcome")
                )
            )[-1]
            return causal_graph.with_role("adjustment", [adjustment_node]), True
        else:
            return causal_graph, False


class TestBaseIdentification:
    def test_base_identification_first(self, cg):
        identifier = DummyIdentification(variant="first")
        identified_cg, is_identified = identifier(causal_graph=cg)

        assert is_identified == True
        assert identified_cg.get_role_dict() == {
            "exposure": ["X"],
            "outcome": ["Y"],
            "adjustment": ["M"],
        }

    def test_base_identification_last(self, cg):
        identifier = DummyIdentification(variant="last")
        identified_cg, is_identified = identifier(causal_graph=cg)

        assert is_identified == True
        assert identified_cg.get_role_dict() == {
            "exposure": ["X"],
            "outcome": ["Y"],
            "adjustment": ["U"],
        }

    def test_base_identification_gibberish(self, cg):
        identifier = DummyIdentification(variant="gibberish")
        identified_cg, is_identified = identifier(causal_graph=cg)

        assert is_identified == False
        assert identified_cg.get_role_dict() == {"exposure": ["X"], "outcome": ["Y"]}
