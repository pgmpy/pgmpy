from collections.abc import Hashable, Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype

from pgmpy import logger


def _column_types(data: pd.DataFrame) -> dict[Hashable, str]:
    """
    Type code of each column of `data`, inferred from its dtype.

    The codes are ``"N"`` (numeric), ``"B"`` (boolean, usable as either discrete or continuous), ``"C"`` (unordered
    categorical: category, object or string columns) and ``"O"`` (ordered categorical).

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset to classify. It is not modified.

    Returns
    -------
    dict
        Mapping from column name to type code.

    Raises
    ------
    ValueError
        If a column has a dtype of none of these kinds, such as datetime.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.utils.data_types import _column_types
    >>> data = pd.DataFrame({"A": [1.5, 2.0], "B": [True, False], "C": ["x", "y"]})
    >>> _column_types(data)
    {'A': 'N', 'B': 'B', 'C': 'C'}
    """
    types = {}
    for col, dtype in data.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            types[col] = "O" if dtype.ordered else "C"
        elif is_bool_dtype(dtype):
            types[col] = "B"
        elif is_numeric_dtype(dtype):
            types[col] = "N"
        elif is_object_dtype(dtype) or is_string_dtype(dtype):
            types[col] = "C"
        else:
            raise ValueError(
                f"Couldn't infer datatype of column: {col} from data. Try specifying the appropriate datatype to the "
                "column."
            )
    return types


def _resolve_booleans(types: dict[Hashable, str], data_types: Sequence[str] | None = None) -> dict[Hashable, str]:
    """
    Replace the boolean code by ``"N"`` when every other column is numeric, and by ``"C"`` when some is not.

    When every column is boolean, the code is ``"N"`` only for a consumer whose `data_types` support continuous but not
    discrete data.
    """
    others = {code for code in types.values() if code != "B"}
    if others:
        boolean_code = "N" if others == {"N"} else "C"
    else:
        continuous_only = data_types is not None and "continuous" in data_types and "discrete" not in data_types
        boolean_code = "N" if continuous_only else "C"
    return {col: boolean_code if code == "B" else code for col, code in types.items()}


def get_dataset_type(data: pd.DataFrame) -> str:
    """
    Returns continuous, discrete or mixed depending on the types of the columns of `data`.

    Boolean columns take the type of the other columns: they count as continuous when every other column is numeric,
    and as discrete otherwise, including when every column is boolean.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset to classify.

    Returns
    -------
    str
        ``"continuous"``, ``"discrete"`` or ``"mixed"``.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.utils import get_dataset_type
    >>> get_dataset_type(pd.DataFrame({"A": [1.5, 2.0], "B": [True, False]}))
    'continuous'
    >>> get_dataset_type(pd.DataFrame({"A": [1.5, 2.0], "C": ["x", "y"]}))
    'mixed'
    """
    kinds = {"continuous" if code == "N" else "discrete" for code in _resolve_booleans(_column_types(data)).values()}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def preprocess_data(
    data: pd.DataFrame, data_types: Sequence[str] | None = None
) -> tuple[pd.DataFrame, dict[Hashable, str]]:
    """
    Copy of `data` in the representation that pgmpy's algorithms expect, and the type code of each column.

    Every column gets the code ``"N"`` (numeric), ``"C"`` (unordered categorical) or ``"O"`` (ordered categorical), and
    the matching representation: numeric columns stay numeric, with integer columns without missing values as
    ``int64``, and categorical columns become ``category`` columns, with ordered categoricals staying ordered.

    A boolean column is numeric (``0``/``1`` integers) when every other column is numeric, and unordered categorical
    when some other column is not. When every column is boolean, it is numeric only for a consumer whose `data_types`
    support continuous but not discrete data.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset to preprocess. It is not modified.
    data_types : list of str, optional
        Dataset types supported by the method that will use the data, among ``"discrete"``, ``"continuous"`` and
        ``"mixed"``.

    Returns
    -------
    data : pandas.DataFrame
        The preprocessed copy.
    types : dict
        Mapping from column name to type code.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.utils import preprocess_data
    >>> data, types = preprocess_data(pd.DataFrame({"A": [1, 2], "B": [True, False], "C": ["x", "y"]}))
    >>> types
    {'A': 'N', 'B': 'C', 'C': 'C'}
    >>> data.dtypes.astype(str).to_dict()
    {'A': 'int64', 'B': 'category', 'C': 'category'}
    >>> preprocess_data(pd.DataFrame({"B": [True, False]}), data_types=["continuous"])[0].dtypes.astype(str).to_dict()
    {'B': 'int64'}
    """
    raw_types = _column_types(data)
    types = _resolve_booleans(raw_types, data_types)
    data = data.copy()
    for col, code in types.items():
        column = data[col]
        if code == "C" and not isinstance(column.dtype, pd.CategoricalDtype):
            data[col] = column.astype("category")
        elif code == "N" and (raw_types[col] == "B" or is_integer_dtype(column.dtype)):
            data[col] = column.astype("Int64" if column.isna().any() else "int64")

    logger.debug(f"Inferred variable types (N=numerical, C=unordered categorical, O=ordered categorical): {types}")
    return data, types


def is_discrete_type(code: str) -> bool:
    """
    Whether a type code returned by :func:`preprocess_data` is a discrete one, ``"C"`` or ``"O"``.

    Parameters
    ----------
    code : str
        Type code of a column.

    Returns
    -------
    bool

    Examples
    --------
    >>> from pgmpy.utils import is_discrete_type
    >>> is_discrete_type("O"), is_discrete_type("N")
    (True, False)
    """
    return code in ("C", "O")


def split_by_type(variables: Sequence[Hashable], types: dict[Hashable, str]) -> tuple[list, list]:
    """
    Split `variables` into continuous and discrete ones using the type codes from :func:`preprocess_data`.

    Parameters
    ----------
    variables : sequence of hashable
        Variables to split. Their order is kept within each part.
    types : dict
        Mapping from variable to type code, as returned by :func:`preprocess_data`.

    Returns
    -------
    continuous : list
        The variables with a continuous type code.
    discrete : list
        The variables with a discrete type code.

    Examples
    --------
    >>> from pgmpy.utils import split_by_type
    >>> split_by_type(["A", "B", "C"], {"A": "N", "B": "C", "C": "N"})
    (['A', 'C'], ['B'])
    """
    continuous = [var for var in variables if not is_discrete_type(types[var])]
    discrete = [var for var in variables if is_discrete_type(types[var])]
    return continuous, discrete
