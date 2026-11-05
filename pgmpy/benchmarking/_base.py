from __future__ import annotations

from dataclasses import replace

import pandas as pd

from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.datasets import load_dataset
from pgmpy.datasets._base import BaseSimulatedDataset, Dataset


def _get_dataset_labels(datasets: list[str | BaseSimulatedDataset | Dataset | pd.DataFrame]) -> list[str]:
    """
    Return a unique label for every entry of ``datasets``.

    Parameters
    ----------
    datasets : list of str, BaseSimulatedDataset, Dataset, or pd.DataFrame
        The datasets passed to the benchmark.

    Returns
    -------
    list of str
        One label per entry, in order. An entry whose label is already taken by an earlier one is suffixed with
        its position in ``datasets``.

    Raises
    ------
    TypeError
        If an entry is not a str, BaseSimulatedDataset, Dataset, or pd.DataFrame.
    """
    labels: list[str] = []
    seen: set[str] = set()

    for position, dataset in enumerate(datasets):
        if isinstance(dataset, str):
            label = dataset
        elif isinstance(dataset, pd.DataFrame):
            label = f"DataFrame_{position}({dataset.shape[0]}x{dataset.shape[1]})"
        elif isinstance(dataset, Dataset):
            label = dataset.name
        elif isinstance(dataset, BaseSimulatedDataset):
            label = " ".join(repr(dataset).split())
        else:
            received = f"the class {dataset.__name__}" if isinstance(dataset, type) else type(dataset).__name__
            raise TypeError(
                f"Each entry in datasets must be a str, BaseSimulatedDataset, Dataset, or pd.DataFrame, got {received}."
            )

        while label in seen:
            label = f"{label}_{position}"
        seen.add(label)
        labels.append(label)

    return labels


def _resolve_dataset(
    dataset: str | BaseSimulatedDataset | Dataset | pd.DataFrame,
    label: str,
    n_samples: int | None,
    seed: int | None,
    ground_truth: dict[str, DAG | PDAG | ADMG | MAG] | None = None,
) -> Dataset:
    """
    Load one benchmark dataset as a ``Dataset``.

    Parameters
    ----------
    dataset : str, BaseSimulatedDataset, Dataset, or pd.DataFrame
        A registered dataset name, an initialized simulator, or caller-prepared data. ``Dataset`` and
        ``pd.DataFrame`` inputs are used as they are; ``n_samples`` doesn't resample them.

    label : str
        The label of ``dataset`` from ``_get_dataset_labels``.

    n_samples : int or None
        Number of samples to load or simulate.

    seed : int or None
        Seed for registered datasets. Initialized simulators keep their own seed.

    ground_truth : dict or None, default=None
        Ground-truth graphs keyed by label, used only for datasets that have none.

    Returns
    -------
    Dataset
        The resolved dataset. The caller's ``Dataset`` or ``pd.DataFrame`` is never modified.

    Raises
    ------
    ValueError
        If ``dataset`` is a name that isn't registered.
    """
    if isinstance(dataset, str):
        resolved = load_dataset(dataset, n_samples=n_samples, seed=seed)
        resolved.tags["n_samples"], resolved.tags["n_variables"] = resolved.data.shape
    elif isinstance(dataset, BaseSimulatedDataset):
        tags = dataset.get_tags()
        df = dataset.load_dataframe(n_samples=n_samples)
        tags["n_samples"], tags["n_variables"] = df.shape
        resolved = Dataset(name=label, data=df, ground_truth=dataset.load_ground_truth(), tags=tags)
    elif isinstance(dataset, Dataset):
        resolved = dataset
    else:
        resolved = Dataset(name=label, data=dataset)

    if resolved.ground_truth is None and ground_truth is not None and label in ground_truth:
        new_tags = dict(resolved.tags) if resolved.tags is not None else {}
        new_tags["has_ground_truth"] = True
        resolved = replace(resolved, ground_truth=ground_truth[label], tags=new_tags)

    return resolved
