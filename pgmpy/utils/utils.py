import gzip
import warnings

import pandas as pd

try:
    from importlib.resources import files
except ImportError:
    # For python 3.8 and lower
    from importlib_resources import files

from pgmpy import logger


def get_example_model(model: str):
    """
    Fetches the specified model from bnlearn repository and returns a
    pgmpy.model instance.

    Parameter
    ---------
    model: str
        Any model from bnlearn repository (http://www.bnlearn.com/bnrepository)
          and dagitty (https://www.dagitty.net/)
        Discrete Bayesian Network Options:
            Small Networks: asia, cancer, earthquake, sachs, survey
            Medium Networks: alarm, barley, child, insurance, mildew, water
            Large Networks: hailfinder, hepar2, win95pts
            Very Large Networks: andes, diabetes, link, munin1, munin2, munin3,
            munin4, pathfinder, pigs, munin
        Gaussian Bayesian Network Options: ecoli70,
        magic-niab, magic-irri, arth150
        Conditional Linear Gaussian Bayesian Network Options: sangiovese, mehra
        DAG Options: M-bias, confounding, mediator, paths,
          Sebastiani_2005, Polzer_2012,
          Schipf_2010, Shrier_2008, Acid_1996,
            Thoemmes_2013, Kampen_2014, Didelez_2010

    Example
    -------
    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model(model="asia")
    >>> model

    Returns
    -------
    pgmpy.models instance: An instance of
      one of the model classes in pgmpy.models
                           depending on the type of dataset.
    """
    warnings.warn(
        "`get_example_model` is deprecated. Please use `pgmpy.example_models.load_model` instead.",
        FutureWarning,
        stacklevel=2,
    )
    cat_models = {
        "asia",
        "cancer",
        "earthquake",
        "sachs",
        "survey",
        "alarm",
        "barley",
        "child",
        "insurance",
        "mildew",
        "water",
        "hailfinder",
        "hepar2",
        "win95pts",
        "andes",
        "diabetes",
        "link",
        "munin1",
        "munin2",
        "munin3",
        "munin4",
        "pathfinder",
        "pigs",
        "munin",
    }

    cont_models = {
        "ecoli70",
        "magic-niab",
        "magic-irri",
        "arth150",
    }

    hybrid_models = {
        "sangiovese",
        "mehra",
    }

    # Took the shorthand names from
    #  https://github.com/jtextor/dagitty/blob/master/r/man/getExample.Rd +
    #  year
    dag_models = {
        "M-bias",
        "confounding",
        "mediator",
        "paths",
        "Sebastiani_2005",
        "Polzer_2012",
        "Schipf_2010",
        "Shrier_2008",
        "Acid_1996",
        "Thoemmes_2013",
        "Kampen_2014",
        "Didelez_2010",
    }

    filenames = {
        "asia": "utils/example_models/asia.bif.gz",
        "cancer": "utils/example_models/cancer.bif.gz",
        "earthquake": "utils/example_models/earthquake.bif.gz",
        "sachs": "utils/example_models/sachs.bif.gz",
        "survey": "utils/example_models/survey.bif.gz",
        "alarm": "utils/example_models/alarm.bif.gz",
        "barley": "utils/example_models/barley.bif.gz",
        "child": "utils/example_models/child.bif.gz",
        "insurance": "utils/example_models/insurance.bif.gz",
        "mildew": "utils/example_models/mildew.bif.gz",
        "water": "utils/example_models/water.bif.gz",
        "hailfinder": "utils/example_models/hailfinder.bif.gz",
        "hepar2": "utils/example_models/hepar2.bif.gz",
        "win95pts": "utils/example_models/win95pts.bif.gz",
        "andes": "utils/example_models/andes.bif.gz",
        "diabetes": "utils/example_models/diabetes.bif.gz",
        "link": "utils/example_models/link.bif.gz",
        "munin1": "utils/example_models/munin1.bif.gz",
        "munin2": "utils/example_models/munin2.bif.gz",
        "munin3": "utils/example_models/munin3.bif.gz",
        "munin4": "utils/example_models/munin4.bif.gz",
        "pathfinder": "utils/example_models/pathfinder.bif.gz",
        "pigs": "utils/example_models/pigs.bif.gz",
        "munin": "utils/example_models/munin.bif.gz",
        "ecoli70": "utils/example_models/ecoli70.json",
        "magic-niab": "utils/example_models/magic-niab.json",
        "magic-irri": "utils/example_models/magic-irri.json",
        "arth150": "utils/example_models/arth150.json",
        "sangiovese": "",
        "mehra": "",
        "M-bias": "utils/example_models/M-bias.txt",
        "confounding": "utils/example_models/confounding.txt",
        "mediator": "utils/example_models/mediator.txt",
        "paths": "utils/example_models/paths.txt",
        "Sebastiani_2005": "utils/example_models/Sebastiani_2005.txt",
        "Polzer_2012": "utils/example_models/Polzer_2012.txt",
        "Schipf_2010": "utils/example_models/Schipf_2010.txt",
        "Shrier_2008": "utils/example_models/Shrier_2008.txt",
        "Acid_1996": "utils/example_models/Acid_1996.txt",
        "Thoemmes_2013": "utils/example_models/Thoemmes_2013.txt",
        "Kampen_2014": "utils/example_models/Kampen_2014.txt",
        "Didelez_2010": "utils/example_models/Didelez_2010.txt",
    }

    if model not in filenames:
        raise ValueError(f"Unknown model name: {model}. Please refer documentation for valid model names.")

    path = filenames[model]

    # Determine the model type
    if model in cat_models:
        if path.endswith(".bif.gz"):
            from pgmpy.readwrite import BIFReader

            ref = files("pgmpy") / path
            with gzip.open(ref) as f:
                content = f.read()
            reader = BIFReader(string=content.decode("utf-8"))
            return reader.get_model()

    elif model in cont_models:
        from pgmpy.models import LinearGaussianBayesianNetwork

        full_path = str(files("pgmpy") / path)
        return LinearGaussianBayesianNetwork.load(full_path)

    elif model in dag_models:
        from pgmpy.base import DAG

        fullpath = files("pgmpy") / path
        return DAG.from_dagitty(filename=fullpath)

    elif model in hybrid_models:
        raise ValueError("Hybrid models aren't supported yet.")


def discretize(data, cardinality, labels=dict(), method="rounding"):
    """
    Discretizes a given continuous dataset.

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset to discretize. All columns must have continuous values.

    cardinality: dict
        A dictionary of the form (str: int) representing the number of bins
        to create for each of the variables.

    labels: dict (default: None)
        A dictionary of the form (str: list) representing the label names for
        each variable in the discretized dataframe.

    method: rounding or quantile
        If rounding, equal width bins are created and
          data is discretized into these bins.
          Refer pandas.cut for more details.
        If quantile, creates bins such that each
          bin has an equal number of datapoints.
            Refer pandas.qcut for more details.

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.utils import discretize
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal(1000)
    >>> Y = 0.2 * X + rng.standard_normal(1000)
    >>> Z = 0.4 * X + 0.5 * Y + rng.standard_normal(1000)
    >>> df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    >>> df_disc = discretize(
    ...     df,
    ...     cardinality={"X": 3, "Y": 3, "Z": 3},
    ...     labels={
    ...         "X": ["low", "mid", "high"],
    ...         "Y": ["low", "mid", "high"],
    ...         "Z": ["low", "mid", "high"],
    ...     },
    ... )
    >>> df_disc.head()
        X    Y    Z
    0   mid  mid  mid
    1   mid  mid  low
    2   mid  mid  mid
    3  high  mid  mid
    4   low  mid  low

    Returns
    -------
    pandas.DataFrame: A discretized dataframe.
    """
    df_copy = data.copy()
    if method == "rounding":
        for column in data.columns:
            df_copy[column] = pd.cut(
                df_copy[column],
                bins=cardinality[column],
                include_lowest=True,
                labels=labels.get(column),
            )
    elif method == "quantile":
        for column in data.columns:
            df_copy[column] = pd.qcut(df_copy[column], q=cardinality[column], labels=labels.get(column))

    return df_copy


def llm_pairwise_orient(
    x,
    y,
    descriptions,
    system_prompt=None,
    llm_model="gemini/gemini-1.5-flash",
    **kwargs,
):
    """
    Asks a Large Language Model (LLM) for the
     orientation of an edge between `x` and `y`.

    Parameters
    ----------
    x: str
        The first variable's name

    y: str
        The second variable's name

    descriptions: dict
        A dict of the form {variable: description}
          containing text description of the variables.

    system_prompt: str
        A system prompt to give the LLM.

    llm_model: str (default: gemini/gemini-pro)
        The LLM model to use. Please refer to litellm
          documentation (https://docs.litellm.ai/docs/providers)
        for available model options. Default is gemini-pro.

    kwargs: kwargs
        Any additional parameters to pass to litellm.completion method.

    Returns
    -------
    tuple:
        Returns a tuple (source, target) representing the edge direction.
    """
    try:
        from litellm import completion
    except ImportError as e:
        raise ImportError(
            f"{e}. litellm is required for using"
            " LLM based pairwise orientation. "
            "Please install using: pip install litellm"
        ) from None

    if system_prompt is None:
        system_prompt = "You are an expert in Causal Inference"

    prompt = f""" {system_prompt}. You are
      given two variables with the following descriptions:
        <A>: {descriptions[x]}
        <B>: {descriptions[y]}

        Which of the following two options is the most likely causal direction between them:
        1. <A> causes <B>
        2. <B> causes <A>

        Return a single number (1 or 2) as your answer. I do not need the reasoning behind it.
        Do not add any formatting in the answer.
        """
    response = completion(model=llm_model, messages=[{"role": "user", "content": prompt}])
    response = response.choices[0].message.content
    response_txt = response.strip().lower().replace("*", "")
    if response_txt in ("a", "1"):
        return (x, y)
    elif response_txt in ("b", "2"):
        return (y, x)
    else:
        raise ValueError("Results from the LLM are unclear. Try calling the function again.")


def manual_pairwise_orient(x, y):
    """
    Generates a prompt for the user to
      input the direction between the variables.

    Parameters
    ----------
    x: str
        The first variable's name

    y: str
        The second variable's name

    Returns
    -------
    tuple:
        Returns a tuple (source, target) representing the edge direction.
    """
    user_input = input(
        f"Select the edge direction between"
        f" {x} and {y}. \n 1. {x} -> {y} \n 2. {x} <- {y} \n"
        "3. No edge \n Please enter 1, 2 or 3: "
    )
    if user_input == "1":
        return (x, y)
    elif user_input == "2":
        return (y, x)
    elif user_input == "3":
        return None


def preprocess_data(df):
    """
    Tries to figure out the data type of each variable `df`.

    Assigns one of (numerical, categorical unordered, categorical ordered) datatypes to each column in `df`. Also
    changes any object datatypes to categorical.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe.

    Returns
    -------
    (pd.DataFrame, dtypes): tuple of transformed dataframe and a dictionary with inferred datatype of each column.
    """
    df = df.copy()
    dtypes = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("int")
            dtypes[col] = "N"
        elif pd.api.types.is_numeric_dtype(df[col]):
            dtypes[col] = "N"
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            dtypes[col] = "C"
            df[col] = df[col].astype("category")
        elif isinstance(df[col].dtype, pd.CategoricalDtype):
            if df[col].dtype.ordered:
                dtypes[col] = "O"
            else:
                dtypes[col] = "C"
        else:
            raise ValueError(
                f"Couldn't infer datatype of column: {col} from data. "
                "Try specifying the appropriate datatype to the column."
            )

    logger.info(
        f" Datatype (N=numerical, C=Categorical Unordered,O=Categorical Ordered)inferred from data: \n {dtypes}"
    )
    return (df, dtypes)


def _heuristic_categorical_detection(df, dtypes):
    """
    Creates a warning if numerical values are detected for a categorical variable.
    """
    # credit: https://stackoverflow.com/a/35827646
    potential_categorical = []
    for var in df.columns:
        if dtypes[var] == "N":
            if 1.0 * df[var].nunique() / df[var].count() < 0.1:
                potential_categorical.append(var)
    if len(potential_categorical) > 0:
        logger.warning(
            f"Variables: {potential_categorical} are likely categorical, but using numerical values. Please set the"
            " dtype as `categorical` in pandas dataframe if that's the case, otherwise ignore this warning."
        )


def get_dataset_type(data: pd.DataFrame) -> str:
    """
    Returns continuous, discrete or mixed depending on the type of variable
    data in the given dataset.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to analyze

    Returns
    -------
    str
        `continuous`, `discrete` or `mixed`.
    """

    df, dtypes = preprocess_data(data)
    dtypes_set = set(dtypes.values())

    if "N" in dtypes_set:
        _heuristic_categorical_detection(df, dtypes)

    if len(dtypes_set) == 1:
        if "N" in dtypes_set:
            return "continuous"
        elif "C" in dtypes_set:
            return "discrete"
    return "mixed"


def to_timeseries_format(df: pd.DataFrame, return_format: str = "pd-multiindex"):
    """
    Converts given wide format dataframe to different time series formats.

    Takes a pandas dataframe with columns taken as ("Variable name", timestep) and rows represented as
    traces ( "wide" format) and converts it to different format as specified in `return_format` argument.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe represented in the wide format (on rows we have samples, on columns, unsorted pairs of
        ("Variable", "timestep")

    return_format : {'pd-multiindex', 'numpy3d', 'pd-list', 'sorted'}
        Controls the return representation. The options are:

        "numpy3d" : returns a numpy 3D tensor, where first dimension represents trace, second dimension
                    represents variable, third dimension represent timestep

        "pd-multiindex" : returns the pandas multiindex DataFrame, with indexes of ("Variable name", "timestep")

        "pd-list" : returns a list of pandas DataFrames. For every sample, a Dataframe is created, where rows
                    contain timestep and columns represent variables

        "sorted" : makes sure that the representation of [sample, ("variable", "timestep")] is sorted, which
                   makes further processing easier

    Returns
    -------
    np.ndarray or pd.DataFrame or list of pd.DataFrame:
        Depends on `return_format` variable. `numpy3d` returns a numpy array (`np.ndarray`), while rest of the
        representations return a pandas DataFrame.

    Examples
    --------

    For input dataframe `df`, represented in the wide format

      (D, 0) (G, 0) (I, 0) (D, 1) (G, 1) (D, 2) (G, 2) (I, 1) (I, 2)
    0      1      1      0      0      0      0      0      1      0
    1      0      2      0      1      1      1      1      1      1

    >>> to_timeseries_format(df, return_format="numpy3d")
    [[[1 0 0]
      [1 0 0]
      [0 1 0]]
     [[0 1 1]
      [2 1 1]
      [0 1 1]]]

    >>> to_timeseries_format(df, return_format="pd-multiindex")
    variable       D  G  I
    instance time
    0        0     1  1  0
             1     0  0  1
             2     0  0  0
    1        0     0  2  0
             1     1  1  1
             2     1  1  1

    >>> to_timeseries_format(df, return_format="pd-list")
    [variable  D  G  I
     time
     0         1  1  0
     1         0  0  1
     2         0  0  0,
     variable  D  G  I
     time
     0         0  2  0
     1         1  1  1
     2         1  1  1]

    >>> to_timeseries_format(df, return_format="sorted")
            (D,0), (D,1), (D,2), (G,0), (G,1), (G,2), (I,0), (I,1), (I,2)
    0         1      0      0      1      0      0      0      1      0
    1         0      1      1      2      1      1      0      1      1

    """
    x = df.copy()

    # normalize the columns to multiindex
    if not isinstance(x.columns, pd.MultiIndex):
        x.columns = pd.MultiIndex.from_tuples(x.columns, names=["variable", "time"])
    else:
        x.columns = x.columns.set_names(["variable", "time"])

    unique_variables = x.columns.get_level_values("variable").unique().tolist()
    timesteps = sorted(x.columns.get_level_values("time").unique().tolist())
    N, D, T = len(x), len(unique_variables), len(timesteps)

    # sort the columns, to make the ordering easier
    x = x.sort_index(axis=1)

    # cast to different representation
    panel = x
    return_format = return_format.lower()

    if return_format == "numpy3d":
        # no guarantee that there will be order, which complicates the 3D tensor creation
        panel = panel.to_numpy()
        panel = panel.reshape(N, D, T)

    elif return_format == "pd-multiindex":
        panel = x.stack("time")
        panel.index.set_names(["instance", "time"], inplace=True)
        panel = panel.sort_index()
        panel.columns = panel.columns.get_level_values("variable")

    elif return_format == "pd-list":
        # return the list of dataframes, one per time series
        panel = x.stack("time")
        panel.index.set_names(["instance", "time"], inplace=True)
        panel = panel.sort_index()
        panel.columns = panel.columns.get_level_values("variable")

        panel = [pd.DataFrame(panel.loc[i]) for i in range(df.shape[0])]

    elif return_format == "sorted":
        panel.sort_index(inplace=True, axis=1)

    else:
        raise ValueError(
            f"Unknown representation: {return_format}. Supported `return_types`"
            "are: numpy3d, pd-multiindex, pd-list, sorted"
        )

    return panel
