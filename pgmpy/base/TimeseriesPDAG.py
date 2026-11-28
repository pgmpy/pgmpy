from pgmpy.base._timeseriesbase import _TimeSeriesCoreGraph


class TimeseriesPDAG(_TimeSeriesCoreGraph):
    """
    Class for representing time-series partially directed acyclic graphs (PDAGs) in pgmpy.
    This class inherits from the base class ``_TimeSeriesCoreGraph``.
    Docs are in progress.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "--"])
