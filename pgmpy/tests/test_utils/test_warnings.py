import runpy
import sys
from pathlib import Path

import pytest

from pgmpy.base import PDAG
from pgmpy.utils import _warnings


@pytest.mark.parametrize("category", [UserWarning, FutureWarning])
def test_warn_external_reports_test_caller(category):
    with pytest.warns(category, match="Check this result") as recorded:
        line = sys._getframe().f_lineno + 1
        _warnings._warn_external("Check this result", category)

    assert len(recorded) == 1
    assert Path(recorded[0].filename).resolve() == Path(__file__).resolve()
    assert recorded[0].lineno == line
    assert recorded[0].category is category


@pytest.mark.parametrize("method", ["to_dag", "to_cpdag"])
def test_pdag_warning_reports_test_caller(method):
    pdag = PDAG(edge_list=[("A", "B", "--"), ("B", "C", "--"), ("C", "D", "--"), ("D", "A", "--")])

    with pytest.warns(UserWarning, match="PDAG has no consistent extension") as recorded:
        line = sys._getframe().f_lineno + 1
        getattr(pdag, method)()

    assert len(recorded) == 1
    assert Path(recorded[0].filename).resolve() == Path(__file__).resolve()
    assert recorded[0].lineno == line


def test_pdag_warning_reports_external_script(tmp_path):
    script = tmp_path / "caller.py"
    script.write_text(
        "from pgmpy.base import PDAG\n"
        "graph = PDAG(edge_list=[('A', 'B', '--'), ('B', 'C', '--'), ('C', 'D', '--'), ('D', 'A', '--')])\n"
        "graph.to_cpdag()\n"
    )

    with pytest.warns(UserWarning, match="PDAG has no consistent extension") as recorded:
        runpy.run_path(str(script))

    assert len(recorded) == 1
    assert Path(recorded[0].filename).resolve() == script.resolve()
    assert recorded[0].lineno == 3
