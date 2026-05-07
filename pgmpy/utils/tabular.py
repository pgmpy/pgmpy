from collections.abc import Iterable

import numpy as np
import pandas as pd


def collect_state_names(data: pd.DataFrame, variable: str) -> list:
    """Return the sorted observed states for `variable` in `data`."""
    return sorted(list(data.loc[:, variable].dropna().unique()))


def build_state_names(data: pd.DataFrame, state_names: dict | None = None) -> dict:
    """Build a complete state-name mapping for all variables in `data`."""
    variables = list(data.columns.values)

    if not isinstance(state_names, dict):
        return {var: collect_state_names(data, var) for var in variables}

    inferred_state_names = {}
    for var in variables:
        observed_states = collect_state_names(data, var)
        if var in state_names:
            if not set(observed_states) <= set(state_names[var]):
                raise ValueError(f"Data contains unexpected states for variable: {var}.")
            inferred_state_names[var] = state_names[var]
        else:
            inferred_state_names[var] = observed_states

    return inferred_state_names


def get_state_counts(
    data: pd.DataFrame,
    state_names: dict,
    variable: str,
    parents=(),
    sample_weight: np.ndarray | None = None,
    reindex: bool = True,
) -> pd.DataFrame:
    """Return counts for `variable`, optionally conditioned on `parents`.

    If `sample_weight` is provided, it must be an array-like of length `len(data)`
    aligned to `data`'s row order; counts become weighted sums. Length and dtype
    validation is the caller's responsibility.
    """
    parents = list(parents)

    if sample_weight is None:
        if not parents:
            state_count_data = data.loc[:, variable].value_counts()
            return state_count_data.reindex(state_names[variable]).fillna(0).to_frame()
        state_count_data = data.groupby([variable] + parents, observed=True).size().unstack(parents)
    else:
        weights = pd.Series(np.asarray(sample_weight), index=data.index)
        groupers = [data[variable]] + [data[p] for p in parents]
        if not parents:
            state_count_data = weights.groupby(groupers, observed=True).sum()
            return state_count_data.reindex(state_names[variable]).fillna(0).to_frame()
        state_count_data = weights.groupby(groupers, observed=True).sum().unstack(parents)

    if not isinstance(state_count_data.columns, pd.MultiIndex):
        state_count_data.columns = pd.MultiIndex.from_arrays([state_count_data.columns])

    if reindex:
        row_index = state_names[variable]
        column_index = pd.MultiIndex.from_product([state_names[p] for p in parents], names=parents)
        return state_count_data.reindex(index=row_index, columns=column_index).fillna(0)

    return state_count_data.fillna(0)


def encode_columns(data: pd.DataFrame, state_names: dict) -> tuple[dict, dict]:
    """Encode columns of `data` as integer codes aligned to `state_names`.

    Returns two dicts keyed by column name: `codes[col]` is an int64 numpy array of
    length `len(data)` holding the index of each value within `state_names[col]`
    (or -1 for missing/unknown values), and `cardinalities[col]` is `len(state_names[col])`.

    Used by `get_state_counts_array` to build contingency tables via `np.bincount`
    without going through pandas groupby on the hot path.
    """
    codes = {}
    cardinalities = {}
    for col in data.columns:
        cats = state_names[col]
        cat = pd.Categorical(data[col], categories=cats)
        codes[col] = np.asarray(cat.codes, dtype=np.int64)
        cardinalities[col] = len(cats)
    return codes, cardinalities


def get_state_counts_array(
    codes: dict,
    cardinalities: dict,
    variable: str,
    parents: Iterable = (),
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Return a `(var_card, n_parent_strata)` contingency table as a numpy array.

    Builds the table in a single `np.bincount` over a flat product index, using the
    pre-encoded integer codes returned by `encode_columns`. Rows with a missing
    value (code < 0) in `variable` or any of the `parents` are excluded, matching
    the behavior of `get_state_counts`'s pandas groupby (which silently drops NaN
    rows). `n_parent_strata` is the product of `cardinalities[p]` over `parents`,
    or 1 if `parents` is empty.

    Parameters
    ----------
    codes, cardinalities : dict
        Output of `encode_columns`.
    variable : str
        The conditioned-on variable.
    parents : iterable of str
        Conditioning parents. The column-axis layout is the row-major flattening
        of `parents` cardinalities (first parent is the slowest-varying axis).
    sample_weight : np.ndarray, optional
        If provided, must be an array of length `len(codes[variable])` aligned to
        the original row order. Counts become weighted sums.
    """
    parents = tuple(parents)
    v_codes = codes[variable]
    kv = cardinalities[variable]

    valid = v_codes >= 0
    for p in parents:
        valid &= codes[p] >= 0

    if valid.all():
        v = v_codes
        p_codes_list = [codes[p] for p in parents]
        w = sample_weight
    else:
        v = v_codes[valid]
        p_codes_list = [codes[p][valid] for p in parents]
        w = sample_weight[valid] if sample_weight is not None else None

    if not parents:
        counts = np.bincount(v, weights=w, minlength=kv).astype(np.float64, copy=False)
        return counts.reshape(kv, 1)

    parent_idx = np.zeros(len(v), dtype=np.int64)
    n_parents_total = 1
    for p_codes, p in zip(p_codes_list, parents):
        parent_idx = parent_idx * cardinalities[p] + p_codes
        n_parents_total *= cardinalities[p]

    flat = parent_idx * kv + v
    counts = np.bincount(flat, weights=w, minlength=n_parents_total * kv).astype(np.float64, copy=False)
    return counts.reshape(n_parents_total, kv).T
