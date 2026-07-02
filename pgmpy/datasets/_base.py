from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.utils.hf_hub import read_hf_file


@dataclass
class Dataset:
    name: str
    data: pd.DataFrame
    expert_knowledge: ExpertKnowledge | None = None
    ground_truth: DAG | PDAG | ADMG | MAG | None = None

    tags: dict[str, Any] = None

    def __str__(self) -> str:
        return (
            f"Dataset(name={self.name}, \n data=DataFrame of size: {self.data.shape}, \n "
            f"expert_knowledge={self.expert_knowledge}, \n ground_truth={self.ground_truth}, \n tags={self.tags})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class BaseDataset(BaseObject):
    """
    Base class for all datasets in pgmpy.
    Inherits from skbase.base.BaseObject to utilize its tag and lookup functionality.
    """

    # define tags
    _tags = {
        "name": None,
        "n_variables": None,
        "n_samples": None,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = ""
    repo_id = "pgmpy/example_datasets"
    repo_type = "dataset"
    revision = "main"

    @staticmethod
    def _parse_expert_knowledge(raw_expert_knowledge: bytes) -> ExpertKnowledge:
        """
        Helper method to parse expert knowledge from raw bytes.
        """
        text = raw_expert_knowledge.decode("utf-8-sig", errors="ignore")

        temporal: list[list[str]] = []
        forbids: list[tuple[str, str]] = []
        requires: list[tuple[str, str]] = []

        section = None

        for raw_line in text.splitlines():
            stripped = raw_line.strip()

            if not stripped:
                continue

            lower = stripped.lower()

            # Section headers
            if lower == "/knowledge":
                section = None
                continue
            if lower == "addtemporal":
                section = "addtemporal"
                continue
            if lower == "forbiddirect":
                section = "forbiddirect"
                continue
            if lower in ("requiredirect", "requireddirect"):
                section = "requiredirect"
                continue

            # Content, depending on current section
            if section == "addtemporal":
                # Treat lines that are just an integer as placeholders for empty lines
                if stripped.isdigit():
                    temporal.append([])
                else:
                    tokens = stripped.split()
                    # Skip the first token; its the line number
                    temporal.append(tokens[1:])

            elif section == "forbiddirect":
                tokens = stripped.split()
                forbids.append((tokens[0], tokens[1]))

            elif section == "requiredirect":
                tokens = stripped.split()
                requires.append((tokens[0], tokens[1]))

        return ExpertKnowledge(forbidden_edges=forbids, required_edges=requires, temporal_order=temporal)

    @classmethod
    def _get_raw_data(cls, filename) -> bytes:
        """
        Fetches a dataset file from the Hugging Face Hub cache.
        """
        return read_hf_file(
            repo_id=cls.repo_id,
            filename=f"{cls.base_url}/{filename}",
            repo_type=cls.repo_type,
            revision=cls.revision,
        )

    @classmethod
    def load_dataframe(cls, n_samples=None, seed=None) -> pd.DataFrame:
        """
        Fetches/reads from cache the data associated with the dataset.

        Parameters
        ----------
        n_samples : int, optional
            If provided, return a random subsample of this size. Capped at the
            dataset size.
        seed : int, optional
            Random seed for reproducible subsampling.
        """
        raw_data = cls._get_raw_data(cls.data_url)
        df = pd.read_csv(io.BytesIO(raw_data), sep=getattr(cls, "sep", "\t"))
        if cls.get_class_tag("has_missing_data"):
            df.replace(cls.missing_values_marker, pd.NA, inplace=True)
        if cls.get_class_tag("has_index_col"):
            df.drop(df.columns[0], axis=1, inplace=True)
        if len(cls.categorical_variables) > 0:
            for col in cls.categorical_variables:
                df[col] = df[col].astype("category")
        if len(cls.ordinal_variables) > 0:
            for col, order in cls.ordinal_variables.items():
                cat_type = pd.CategoricalDtype(categories=order, ordered=True)
                df[col] = df[col].astype(cat_type)
        if n_samples is not None:
            if n_samples > len(df):
                warnings.warn(
                    f"Requested {n_samples} samples but dataset only has {len(df)}. Returning all {len(df)} rows."
                )
            else:
                df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
        return df

    @classmethod
    def load_expert_knowledge(cls) -> ExpertKnowledge:
        """Fetches/reads from cache the expert knowledge associated with the dataset."""
        if not cls.get_class_tag("has_expert_knowledge"):
            return None

        raw_data = cls._get_raw_data(cls.expert_knowledge_url)
        expert_knowledge = cls._parse_expert_knowledge(raw_data)
        return expert_knowledge

    @classmethod
    def load_ground_truth(cls, **kwargs) -> DAG | None:
        """Fetches/reads from cache the ground truth graph associated with the dataset.

        Parameters
        ----------
        **kwargs
            Absorbed for call-signature compatibility with ``load_dataset()``.
            Static datasets ignore all forwarded arguments.
        """
        if not cls.get_class_tag("has_ground_truth"):
            return None

        raw_data = cls._get_raw_data(cls.ground_truth_url).decode("utf-8-sig", errors="ignore")
        return DAG.from_dagitty(raw_data)


class BaseCovarianceDataset(BaseDataset):
    """
    Base class for datasets defined by a covariance matrix.

    Instead of loading a static data file, ``load_dataframe`` generates samples from a multivariate normal distribution
    parameterized by the dataset's covariance matrix.
    """

    _tags = {"is_simulated": True}

    @classmethod
    def _load_covariance_matrix(cls) -> pd.DataFrame:
        """
        Fetches the data and creates a covariance matrix DataFrame.
        """
        raw_data = cls._get_raw_data(cls.data_url).decode("utf-8-sig", errors="ignore")

        lines = raw_data.strip().splitlines()
        # First replace multiple spaces with a single space and then split the line on either \t or space. Datasets are
        # not uniform.
        names = re.split(r"\t|\ ", re.sub(r"\s+", " ", lines[1].strip()))

        mat = np.zeros((len(names), len(names)), dtype=float)

        for i, line in enumerate(lines[2 : 2 + len(names)]):
            vals = np.fromstring(line, sep="\t", dtype=float)
            mat[i, : i + 1] = vals
            mat[: i + 1, i] = vals

        return pd.DataFrame(mat, columns=names, index=names)

    @classmethod
    def load_dataframe(cls, n_samples=None, seed=None) -> pd.DataFrame:
        """Generate data from a covariance matrix.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate.  Defaults to the class tag
            ``n_samples`` when not provided.
        seed : int, optional
            Random seed for reproducible generation.  When ``None``, the
            existing unseeded behavior is preserved.
        """
        cov_matrix = cls._load_covariance_matrix()
        mean = [0] * cls.get_class_tag("n_variables")
        actual_n = n_samples if n_samples is not None else cls.get_class_tag("n_samples")
        rng = np.random.default_rng(seed) if seed is not None else np.random
        data = pd.DataFrame(
            rng.multivariate_normal(mean, cov_matrix.values, size=actual_n),
            columns=cov_matrix.columns,
        )
        return data


class BaseTubingenDataset(BaseDataset):
    """
    Base class for benchmark datasets that consist of multiple independent cause-effect pairs/files.
    URL: https://webdav.tuebingen.mpg.de/cause-effect/
    """

    @classmethod
    def load_dataframe(cls, pair_id: int) -> pd.DataFrame:
        raw_data = cls._get_raw_data(f"pair{pair_id:04}.txt")
        return pd.read_csv(io.BytesIO(raw_data), sep=r"\s+", header=None, names=["x", "y"])

    @classmethod
    def load_ground_truth(cls, pair_id: int) -> DAG:
        raw_data = cls._get_raw_data(f"pair{pair_id:04}_graph.txt")
        content = raw_data.decode("utf-8-sig", errors="ignore")
        return DAG.from_dagitty(content)


class BaseSimulatedDataset(BaseDataset):
    """
    Base class for simulated datasets.

    Concrete subclasses build the model and its ground-truth graph once in ``__init__`` and expose them through the
    instance methods ``load_dataframe()`` and ``load_ground_truth()``, so a single ``load_dataset`` call reuses one
    model for both the data and the graph.
    """

    _tags = {"is_simulated": True}

    def load_dataframe(self, n_samples=None) -> pd.DataFrame:
        """Generate and return simulated data. Must be implemented by each simulator."""
        raise NotImplementedError(f"{type(self).__name__} must implement load_dataframe().")

    def load_ground_truth(self) -> DAG | PDAG | ADMG | MAG:
        """Construct and return the ground-truth graph. Must be implemented by each simulator."""
        raise NotImplementedError(f"{type(self).__name__} must implement load_ground_truth().")


def load_dataset(
    name: str,
    n_samples: int | None = None,
    seed: int | None = None,
    **sim_kwargs,
) -> Dataset:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset to load.
    n_samples : int, optional
        For static datasets, return a random subsample of this size (capped
        at the dataset size).  For simulated datasets, the number of samples
        to generate.
    seed : int, optional
        Random seed for reproducible subsampling or simulation.
    **sim_kwargs : dict, optional
        Additional keyword arguments forwarded to the simulator's
        ``load_dataframe()`` and ``load_ground_truth()`` methods. Passing
        simulator kwargs to a static dataset raises ``TypeError``. For Tubingen
        datasets, these kwargs are ignored with a warning.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> dataset = load_dataset("sachs_mixed")
    >>> df = dataset.data
    >>> ground_truth = dataset.ground_truth

    Subsample a static dataset:

    >>> dataset = load_dataset("sachs_continuous", n_samples=100, seed=42)
    """
    all_datasets = all_objects(object_types=BaseDataset, package_name="pgmpy.datasets", return_names=False)
    if name.startswith("tubingen"):
        name_parts = name.split("/")
        if len(name_parts) == 2 and name_parts[1].isdigit():
            pair_id = int(name_parts[1])

            if not (1 <= pair_id <= 108):
                raise ValueError(f"Tubingen pair ID must be between 1 and 108. Got {pair_id}.")
            if sim_kwargs:
                warnings.warn(
                    "Tubingen datasets ignore simulator kwargs.",
                    UserWarning,
                )
            target_cls = next(
                (cls for cls in all_datasets if cls.get_class_tag("name") == "tubingen"),
                None,
            )
            df = target_cls.load_dataframe(pair_id)
            gt = target_cls.load_ground_truth(pair_id)

            if n_samples is not None:
                if n_samples > len(df):
                    warnings.warn(
                        f"Requested {n_samples} samples but dataset only has {len(df)}. Returning all {len(df)} rows."
                    )
                else:
                    df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

            tags = target_cls.get_class_tags()
            tags["n_samples"] = df.shape[0]
            tags["has_missing_data"] = bool(df.isnull().any().any())

            return Dataset(
                name=name,
                data=df,
                expert_knowledge=None,
                ground_truth=gt,
                tags=tags,
            )
        else:
            raise ValueError(f"Invalid dataset name format: '{name}'. For Tubingen datasets, use 'tubingen/<pair_id>'.")

    target_cls = None
    for cls in all_datasets:
        if cls.get_class_tag("name") == name:
            target_cls = cls
            break
    if target_cls is None:
        raise ValueError(f"Dataset with name '{name}' not found. Please use list_datasets() to see available datasets.")

    if issubclass(target_cls, BaseSimulatedDataset):
        # Build the model once and reuse it for both the data and the ground-truth graph.
        simulator = target_cls(seed=seed, **sim_kwargs)
        df = simulator.load_dataframe(n_samples=n_samples)
        tags = target_cls.get_class_tags()
        tags["n_samples"], tags["n_variables"] = df.shape
        return Dataset(
            name=name,
            data=df,
            expert_knowledge=None,
            ground_truth=simulator.load_ground_truth(),
            tags=tags,
        )

    return Dataset(
        name=name,
        data=target_cls.load_dataframe(n_samples=n_samples, seed=seed, **sim_kwargs),
        expert_knowledge=target_cls.load_expert_knowledge(),
        ground_truth=target_cls.load_ground_truth(seed=seed, **sim_kwargs),
        tags=target_cls.get_class_tags(),
    )


def list_datasets(**filter_tags) -> list[str]:
    """
    Returns a list of all available datasets, optionally filtered by a query string.

    Parameters
    ----------
    **filter_tags : optional arguments
        If specified, returns only datasets matching the provided tag filters. Any dataset tag can be used as a filter.
        Available tags:
            - n_variables
            - n_samples
            - has_ground_truth
            - has_expert_knowledge
            - has_missing_data
            - is_simulated
            - is_interventional
            - is_discrete
            - is_continuous
            - is_mixed
            - is_ordinal

    Returns
    -------
    list of str
        A sorted list of available dataset names.

    Examples
    --------
    >>> from pgmpy.datasets import list_datasets
    >>> list_datasets()
    ['abalone_continuous', 'abalone_mixed', ..., 'sachs_continuous', ...]

    >>> list_datasets(is_discrete=True, has_ground_truth=True)
    ['sachs_discrete']
    """
    valid_tags = set(BaseDataset._tags.keys())

    if invalid_tags := set(filter_tags.keys()) - valid_tags:
        raise ValueError(
            f"Unrecognized filter argument(s): {sorted(invalid_tags)}. Valid filter tags are: {sorted(valid_tags)}."
        )

    all_datasets = all_objects(
        object_types=BaseDataset,
        package_name="pgmpy.datasets",
        return_names=False,
        filter_tags=filter_tags,
    )

    dataset_names = [cls.get_class_tag("name") for cls in all_datasets if cls.get_class_tag("name") is not None]

    return sorted(dataset_names)
