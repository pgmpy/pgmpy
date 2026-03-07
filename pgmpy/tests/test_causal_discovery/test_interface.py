from pgmpy.causal_discovery import PC
from pgmpy.utils.test_causal_discovery_checks import (
    check_causal_discovery_interface,
)


def test_pc_interface():
    check_causal_discovery_interface(PC(return_type="dag"))
