"""Timeseries utilities bridging sktime <-> pgmpy DynamicBayesianNetwork data formats.

This module provides helper functions to convert between the *panel* dataframe
representation that sktime uses for multivariate/panel forecasting tasks and
the wide *sample × (variable,time_slice)* representation that pgmpy expects for
DynamicBayesianNetwork ``fit`` / ``simulate`` APIs.

The chosen conventions keep the implementation intentionally simple and do **not**
try to cover every exotic sktime data container. They should, however, be
sufficient for the vast majority of real-world use-cases.

Key assumptions
---------------
1. **Panel dataframe input (sktime)**
   ``df`` is either

   - a pandas ``DataFrame`` whose index is a 2-level ``MultiIndex`` of the
     form ``(instance, time)``. ``instance`` identifies the *case* / *row* in
     the pgmpy sample dimension, while ``time`` can be any sortable key
     (``DateTime``, integer, etc.).
   - *or* a wide dataframe with any index, plus an ``instance_col`` that holds
     the instance id. In that case the index itself is interpreted as the time
     axis.

   Missing combinations are filled with ``NaN`` so that every instance has the
   same number of time steps.

2. **pgmpy DBN dataframe**
   A *wide* dataframe where **columns** are tuples ``(variable, time_int)`` and
   **rows** correspond to individual instances / trajectories. ``time_int`` is
   always a contiguous ``range(n_time_slices)`` starting at ``0``.

Round-tripping through both functions should therefore preserve the original
values (barring dtype up-casts caused by ``NaN`` filling).
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _canonical_time_order(index: pd.Index) -> list[Hashable]:
    """Return *sorted* list of unique time labels."""
    if isinstance(index, pd.MultiIndex):
        time_labels = index.get_level_values(-1)
    else:
        time_labels = index
    return sorted(pd.unique(time_labels))


def _build_time_mapping(time_labels: list[Hashable]):
    """Map arbitrary time labels -> contiguous ints starting at 0."""
    return {lbl: i for i, lbl in enumerate(time_labels)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_sktime_to_dbn(df: pd.DataFrame, *, instance_col: str | None = None) -> pd.DataFrame:  # noqa: D401
    """Convert an *sktime* style panel dataframe to pgmpy-compatible DBN format.

    Parameters
    ----------
    df : pd.DataFrame
        Either a dataframe with a two-level ``MultiIndex`` (instance, time) or
        a regular dataframe together with *instance_col*.
    instance_col : str, optional
        Name of the column that contains the instance id when *df* does **not**
        have a multi-index. Ignored when *df.index* is a ``MultiIndex``.

    Returns
    -------
    pd.DataFrame
        Wide dataframe whose columns are ``(variable, time_int)`` tuples and
        rows correspond to distinct instances.
    """
    # ---------------------------------------------------------------------
    # Normalise input to: multi-index (instance, time) + variable columns
    # ---------------------------------------------------------------------
    if isinstance(df.index, pd.MultiIndex):
        if df.index.nlevels != 2:
            raise ValueError("df.index must have exactly 2 levels (instance, time) or provide instance_col.")
        panel_df = df.copy()
        panel_df.index.set_names(["instance", "time"], inplace=True)
    else:
        if instance_col is None:
            # Single trajectory – treat entire df as instance 0
            panel_df = df.copy()
            panel_df["__instance__"] = 0
            panel_df.set_index(["__instance__", panel_df.index], inplace=True)
            panel_df.index.set_names(["instance", "time"], inplace=True)
        else:
            if instance_col not in df.columns:
                raise KeyError(f"instance_col '{instance_col}' not found in df.columns")
            panel_df = df.copy()
            panel_df.set_index([instance_col, panel_df.index], inplace=True)
            panel_df.index.set_names(["instance", "time"], inplace=True)

    # ---------------------------------------------------------------------
    # Build canonical time mapping shared by *all* instances
    # ---------------------------------------------------------------------
    time_labels = _canonical_time_order(panel_df.index)
    time_map = _build_time_mapping(time_labels)
    reverse_time_map = {v: k for k, v in time_map.items()}

    # Pre-allocate list of per-instance wide rows
    wide_rows = []
    row_index = []

    # Iterate over instances
    for inst, grp in panel_df.groupby(level="instance"):
        grp = grp.droplevel("instance")  # now index only time
        # Ensure time rows are sorted
        grp = grp.sort_index()

        # Collect row values
        row_dict = {}
        for t_lbl in time_labels:
            t_int = time_map[t_lbl]
            if t_lbl in grp.index:
                # Existing time point – extract values as 1-row Series
                vals = grp.loc[t_lbl]
                for var, val in vals.items():
                    row_dict[(var, t_int)] = val
            else:
                # Missing – fill with NaN
                for var in panel_df.columns:
                    row_dict[(var, t_int)] = np.nan
        wide_rows.append(row_dict)
        row_index.append(inst)

    wide_df = pd.DataFrame(wide_rows, index=row_index)

    # Ensure deterministic column order: sort by time then variable
    wide_df = wide_df.reindex(sorted(wide_df.columns, key=lambda x: (x[1], x[0])), axis=1)

    # Persist mapping so we can reconstruct original labels later
    wide_df.attrs["_time_reverse_map"] = reverse_time_map

    return wide_df


def from_dbn_to_sktime(dbn_df: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
    """Convert pgmpy DBN dataframe back to an *sktime* style panel dataframe.

    The returned dataframe always has a ``MultiIndex`` of ``(instance, time)``
    and regular variable columns which is the common denominator supported by
    sktime. If you prefer a different container you can, of course, post-process
    the result.

    Parameters
    ----------
    dbn_df : pd.DataFrame
        A *wide* dataframe whose columns are tuples ``(variable, time_int)``.

    Returns
    -------
    pd.DataFrame
        Panel dataframe with ``MultiIndex`` (instance, time) and variable
        columns.
    """
    if not all(isinstance(c, tuple) and len(c) == 2 for c in dbn_df.columns):
        raise ValueError("All columns of dbn_df must be 2-tuples (variable, time_int).")

    # Build proper MultiIndex columns
    tmp = dbn_df.copy()
    tmp.columns = pd.MultiIndex.from_tuples(dbn_df.columns, names=["variable", "time_int"])

    # Stack time dimension -> becomes row index level
    long = tmp.stack(level="time_int", future_stack=True)
    long.index.set_names(["instance", "time_int"], inplace=True)

    # Restore original labels if mapping present
    rev_map = dbn_df.attrs.get("_time_reverse_map")
    if rev_map is not None:
        long.index = long.index.set_levels(
            [long.index.levels[0], [rev_map.get(v, v) for v in long.index.levels[1]]],
            level=[0, 1],
        )
    long.index.rename(["instance", "time"], inplace=True)
    long.columns.name = None  # stack() leaves "variable" as name
    # Sort for nice human-readable order
    long = long.sort_index()

    return long
