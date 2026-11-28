from pgmpy.base._timeseriesbase import _TimeSeriesCoreGraph


class TimeseriesDAG(_TimeSeriesCoreGraph):
    """
    Class for representing time-series directed acyclic graphs (DAGs) in pgmpy.
    This class inherits from the base class ``_TimeSeriesCoreGraph`.
    Docs are in progress.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-"])
