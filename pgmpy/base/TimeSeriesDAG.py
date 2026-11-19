from pgmpy.base._timeseriesbase import _TimeSeriesCoregraph


class TimeseriesDAG(_TimeSeriesCoregraph):
    """
    Class for representing time-series directed acyclic graphs (DAGs) in pgmpy.
    This class inherits from the base class ``_TimeSeriesCoregraph`.
    Docs are in progress.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-"])
