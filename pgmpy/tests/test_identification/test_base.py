import pytest

from pgmpy.base import ADMG, DAG, PDAG
from pgmpy.identification import BaseFormulaIdentification, BaseGraphicalIdentification
from pgmpy.identification.probability_expression import ProbabilityExpressionTree, ProbabilityNode


@pytest.fixture
def cg():
    edges = [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")]
    roles = {"exposures": "X", "outcomes": "Y"}
    return DAG(ebunch=edges, roles=roles)


@pytest.fixture
def admg_frontdoor():
    """ADMG with X->Z->Y and X<->Y, identifiable by the frontdoor criterion."""
    return ADMG(
        edge_list=[("X", "Z", "->"), ("Z", "Y", "->"), ("X", "Y", "<>")],
        roles={"exposures": "X", "outcomes": "Y"},
    )


@pytest.fixture
def admg_no_roles():
    """ADMG with no roles assigned — should fail validation."""
    return ADMG(edge_list=[("X", "Z", "->"), ("Z", "Y", "->"), ("X", "Z", "<>")])


@pytest.fixture
def admg_with_conditioning():
    """ADMG with exposures, outcomes, and a conditioning variable Z."""
    return ADMG(
        edge_list=[("X", "Z", "->"), ("Z", "Y", "->"), ("X", "Y", "->"), ("X", "Y", "<>")],
        roles={"exposures": "X", "outcomes": "Y", "conditioning": "Z"},
    )


class DummyIdentification(BaseGraphicalIdentification):
    """Sorts non-exposure and non-outcome nodes in the graph and assigns the
    first or the last one as adjustment node depending on the `variant`.
    """

    def __init__(self, variant=None):
        self.variant = variant
        self.supported_graph_types = (DAG, PDAG)

    def _identify(self, causal_graph):
        if self.variant == "first":
            adjustment_node = sorted(
                set(causal_graph.nodes()) - set(causal_graph.get_role("exposures") + causal_graph.get_role("outcomes"))
            )[0]
            return causal_graph.with_role("adjustment", [adjustment_node]), True
        elif self.variant == "last":
            adjustment_node = sorted(
                set(causal_graph.nodes()) - set(causal_graph.get_role("exposures") + causal_graph.get_role("outcomes"))
            )[-1]
            return causal_graph.with_role("adjustment", [adjustment_node]), True
        else:
            return causal_graph, False


class TestBaseGraphicalIdentification:
    def test_base_identification_first(self, cg):
        identifier = DummyIdentification(variant="first")
        identified_cg, is_identified = identifier(causal_graph=cg)

        assert is_identified == True
        assert identified_cg.get_role_dict() == {
            "exposures": ["X"],
            "outcomes": ["Y"],
            "adjustment": ["M"],
        }

    def test_base_identification_last(self, cg):
        identifier = DummyIdentification(variant="last")
        identified_cg, is_identified = identifier(causal_graph=cg)

        assert is_identified == True
        assert identified_cg.get_role_dict() == {
            "exposures": ["X"],
            "outcomes": ["Y"],
            "adjustment": ["U"],
        }

    def test_base_identification_gibberish(self, cg):
        identifier = DummyIdentification(variant="gibberish")
        identified_cg, is_identified = identifier(causal_graph=cg)

        assert is_identified == False
        assert identified_cg.get_role_dict() == {"exposures": ["X"], "outcomes": ["Y"]}


class DummyFormulaIdentification(BaseFormulaIdentification):
    """Returns a query expression to exercise the base-class wrapper."""

    supported_graph_types = (ADMG, DAG)

    def _identify(self, causal_graph):
        return ProbabilityExpressionTree(
            ProbabilityNode(
                causal_graph.get_role("outcomes"),
                do=causal_graph.get_role("exposures"),
                cond=causal_graph.get_role("conditioning"),
            )
        )


class DummyFailingFormulaIdentification(DummyFormulaIdentification):
    """Fails for the bow-arc effect and succeeds for observational queries."""

    supported_graph_types = (ADMG,)

    def _identify(self, causal_graph):
        if causal_graph.get_role("exposures"):
            self.hedge_ = (causal_graph, causal_graph.get_subgraph(causal_graph.get_role("outcomes")))
            return False
        return super()._identify(causal_graph)


class DummyRequiredRoleFormulaIdentification(DummyFormulaIdentification):
    """Exercises subclass-specific requirements for an additional role."""

    required_roles = ("outcomes", "conditioning")


class IncompleteFormulaIdentification(BaseFormulaIdentification):
    """Does not override _identify - should raise NotImplementedError."""

    supported_graph_types = (ADMG, DAG)


class TestBaseFormulaIdentification:
    def test_identify_success(self, admg_frontdoor, cg):
        identifier = DummyFormulaIdentification()
        expected = ProbabilityExpressionTree(ProbabilityNode({"Y"}, do={"X"}))
        assert identifier.identify(admg_frontdoor) == expected
        assert identifier(cg) == expected
        assert identifier(admg_frontdoor.without_role("exposures")) == ProbabilityExpressionTree(ProbabilityNode({"Y"}))
        assert identifier.hedge_ is None

    def test_identify_failure_sets_hedge(self):
        admg = ADMG(
            edge_list=[("X", "Y", "->"), ("X", "Y", "<>")],
            roles={"exposures": "X", "outcomes": "Y"},
        )
        identifier = DummyFailingFormulaIdentification()
        result = identifier.identify(admg)

        assert result is False
        assert identifier.hedge_[0] is admg
        assert set(identifier.hedge_[1]) == {"Y"}
        assert identifier(admg.without_role("exposures")) == ProbabilityExpressionTree(ProbabilityNode({"Y"}))
        assert identifier.hedge_ is None

    def test_identify_wrong_graph_type(self, admg_frontdoor):
        class DAGOnly(DummyFormulaIdentification):
            supported_graph_types = (DAG,)

        with pytest.raises(ValueError, match="must be an instance of"):
            DAGOnly().identify(admg_frontdoor)

    def test_identify_missing_roles(self, admg_no_roles, admg_frontdoor):
        with pytest.raises(ValueError, match="outcomes"):
            DummyFormulaIdentification().identify(admg_no_roles)

        with pytest.raises(ValueError, match="conditioning"):
            DummyRequiredRoleFormulaIdentification().identify(admg_frontdoor)

    def test_identify_with_all_required_roles(self, admg_with_conditioning):
        expected = ProbabilityExpressionTree(ProbabilityNode({"Y"}, do={"X"}, cond={"Z"}))
        assert DummyRequiredRoleFormulaIdentification().identify(admg_with_conditioning) == expected
        assert DummyFormulaIdentification().identify(admg_with_conditioning) == expected

    def test_identify_not_implemented(self, admg_frontdoor):
        with pytest.raises(NotImplementedError):
            IncompleteFormulaIdentification().identify(admg_frontdoor)
