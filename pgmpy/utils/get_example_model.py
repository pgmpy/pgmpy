"""
Utility functions for loading example datasets from the pgmpy/example_datasets
repository.

GitHub: https://github.com/pgmpy/example_datasets
"""

import os
import urllib.request

import pandas as pd



DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/pgmpy/example_datasets/main/"
)


def get_example_dataset(name, base_url=DEFAULT_BASE_URL):
    """
    Fetches an example dataset from the pgmpy/example_datasets repository.

    Parameters
    ----------
    name : str
        The name of the dataset to load. Must correspond to a directory name
        inside the `real/` or `simulated/` folders of the example_datasets
        repository. For example, ``"sachs"`` or ``"auto_mpg"``.

    base_url : str, optional
        The base URL pointing to the root of the raw example_datasets
        repository. Defaults to::

            "https://raw.githubusercontent.com/pgmpy/example_datasets/main/"

        Override this parameter to load datasets from a fork or a local
        mirror, for example during testing::

            base_url="https://raw.githubusercontent.com/<fork>/example_datasets/main/"

        .. versionadded:: 1.1.0
            The ``base_url`` parameter was added to fix issue #2567, where the
            dataset template had no way to specify the raw file location.

    Returns
    -------
    data : pd.DataFrame
        A pandas DataFrame containing the dataset.

    Raises
    ------
    ValueError
        If ``name`` does not match any known dataset in the repository.
    urllib.error.URLError
        If the dataset cannot be fetched from the resolved URL.

    Examples
    --------
    Load the sachs dataset with the default base URL:

    >>> from pgmpy.utils import get_example_dataset
    >>> df = get_example_dataset("sachs")
    >>> df.shape
    (7466, 11)

    Load from a custom fork (useful for testing or offline mirrors):

    >>> df = get_example_dataset(
    ...     "sachs",
    ...     base_url="https://raw.githubusercontent.com/myfork/example_datasets/main/"
    ... )

    Notes
    -----
    Dataset files are downloaded at runtime and not cached. For repeated use,
    consider saving the returned DataFrame locally:

    >>> df.to_csv("sachs.csv", index=False)
    """
    if not base_url.endswith("/"):
        base_url = base_url + "/"

    # Try real/ first, then simulated/ — mirrors the repo folder structure.
    for category in ("real", "simulated"):
        url = f"{base_url}{category}/{name}/{name}.csv"
        try:
            data = pd.read_csv(url)
            return data
        except Exception:
            continue

    raise ValueError(
        f"Could not find dataset '{name}' at base_url='{base_url}'. "
        f"Check that the dataset exists under real/ or simulated/ in "
        f"https://github.com/pgmpy/example_datasets"
    )