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
    weighted: bool = False,
    reindex: bool = True,
) -> pd.DataFrame:
    """Return counts for `variable`, optionally conditioned on `parents`."""
    parents = list(parents)

    if weighted and ("_weight" not in data.columns):
        raise ValueError("data must contain a `_weight` column if weighted=True")

    if not parents:
        if weighted:
            state_count_data = data.groupby([variable], observed=True)["_weight"].sum()
        else:
            state_count_data = data.loc[:, variable].value_counts()

        return state_count_data.reindex(state_names[variable]).fillna(0).to_frame()

    parents_states = [state_names[parent] for parent in parents]
    if weighted:
        state_count_data = data.groupby([variable] + parents, observed=True)["_weight"].sum().unstack(parents)
    else:
        state_count_data = data.groupby([variable] + parents, observed=True).size().unstack(parents)

    if not isinstance(state_count_data.columns, pd.MultiIndex):
        state_count_data.columns = pd.MultiIndex.from_arrays([state_count_data.columns])

    if reindex:
        row_index = state_names[variable]
        column_index = pd.MultiIndex.from_product(parents_states, names=parents)
        return state_count_data.reindex(index=row_index, columns=column_index).fillna(0)

    return state_count_data.fillna(0)
