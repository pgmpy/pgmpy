from pgmpy.base._timeseriesbase import _TimeSeriesCoregraph


class TimeseriesPDAG(_TimeSeriesCoregraph):
    """
    Class for representing time-series partially directed acyclic graphs (PDAGs) in pgmpy.
    This class inherits from the base class ``_TimeSeriesCoregraph`.
    Docs are in progress.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "--"])
