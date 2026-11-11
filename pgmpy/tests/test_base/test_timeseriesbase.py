from pgmpy.base._timeseriesbase import _TimeSeriesCoreGraph


class TestTimeseriesCoreGraph:
    def test_tau_max(self):
        assert _TimeSeriesCoreGraph(edge_list=[(("A", -3), ("B", 0), "->")])._tau_max == 3
        graph = _TimeSeriesCoreGraph(edge_list=[(("A", -1), ("B", 0), "->")], tau_max=4)
        assert graph.tau_max == 4
