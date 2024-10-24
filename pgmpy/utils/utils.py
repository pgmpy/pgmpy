import gzip
import json
import os

import pandas as pd

try:
    from importlib.resources import files
except:
    # For python 3.8 and lower
    from importlib_resources import files

from pgmpy.global_vars import logger


def get_example_model(model):
    """
    Fetches the specified model from bnlearn repository and returns a
    pgmpy.model instance.

    Parameter
    ---------
    model: str
        Any model from bnlearn repository (http://www.bnlearn.com/bnrepository).

        Discrete Bayesian Network Options:
            Small Networks:
                1. asia
                2. cancer
                3. earthquake
                4. sachs
                5. survey
            Medium Networks:
                1. alarm
                2. barley
                3. child
                4. insurance
                5. mildew
                6. water
            Large Networks:
                1. hailfinder
                2. hepar2
                3. win95pts
            Very Large Networks:
                1. andes
                2. diabetes
                3. link
                4. munin1
                5. munin2
                6. munin3
                7. munin4
                8. pathfinder
                9. pigs
                10. munin
        Gaussian Bayesian Network Options:
                1. ecoli70
                2. magic-niab
                3. magic-irri
                4. arth150
        Conditional Linear Gaussian Bayesian Network Options:
                1. sangiovese
                2. mehra

    Example
    -------
    >>> from pgmpy.data import get_example_model
    >>> model = get_example_model(model='asia')
    >>> model

    Returns
    -------
    pgmpy.models instance: An instance of one of the model classes in pgmpy.models
                           depending on the type of dataset.
    """
    from pgmpy.readwrite import BIFReader

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
        "magic-niab": "",
        "magic-irri": "",
        "arth150": "",
        "sangiovese": "",
        "mehra": "",
    }

    if model not in filenames:
        raise ValueError(f"Unknown model name: {model}")

    path = filenames[model]

    # Handle BIF files (existing logic)
    if path.endswith(".bif.gz"):
        from pgmpy.readwrite import BIFReader

        ref = files("pgmpy") / path
        with gzip.open(ref) as f:
            content = f.read()
        reader = BIFReader(string=content.decode("utf-8"))
        return reader.get_model()

    # Handle JSON files for Gaussian or Conditional Linear Gaussian Bayesian Networks
    elif path.endswith(".json"):

        from pgmpy.factors.continuous import LinearGaussianCPD
        from pgmpy.models import LinearGaussianBayesianNetwork

        with open(files("pgmpy") / path, "r") as f:
            data = json.load(f)

        # Extract nodes, arcs, and CPDs from the JSON file
        nodes = data.get("nodes")
        arcs = data.get("arcs")
        cpds_data = data.get("cpds")

        # Create the model
        model = LinearGaussianBayesianNetwork(arcs)
        model.add_nodes_from(nodes)

        # Create CPDs and add them to the model
        cpds = []
        for cpd_data in cpds_data:
            cpd = LinearGaussianCPD(
                variable=nodes,
                evidence_mean=cpd_data["coefficients"],
                evidence_variance=cpd_data["variance"],
                evidence=cpd_data["parents"],
            )
            cpds.append(cpd)

        model.add_cpds(*cpds)
        return model

    else:
        raise NotImplementedError(f"The specified dataset {model} isn't available.")


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
        If rounding, equal width bins are created and data is discretized into these bins. Refer pandas.cut for more details.
        If quantile, creates bins such that each bin has an equal number of datapoints. Refer pandas.qcut for more details.

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.utils import discretize
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal(1000)
    >>> Y = 0.2 * X + rng.standard_normal(1000)
    >>> Z = 0.4 * X + 0.5 * Y + rng.standard_normal(1000)
    >>> df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    >>> df_disc = discretize(df, cardinality={'X': 3, 'Y': 3, 'Z': 3}, labels={'X': ['low', 'mid', 'high'], 'Y': ['low', 'mid', 'high'], 'Z': ['low', 'mid', 'high']})
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
            df_copy[column] = pd.qcut(
                df_copy[column], q=cardinality[column], labels=labels.get(column)
            )

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
    Asks a Large Language Model (LLM) for the orientation of an edge between `x` and `y`.

    Parameters
    ----------
    x: str
        The first variable's name

    y: str
        The second variable's name

    description: dict
        A dict of the form {variable: description} containing text description of the variables.

    system_prompt: str
        A system prompt to give the LLM.

    llm_model: str (default: gemini/gemini-pro)
        The LLM model to use. Please refer to litellm documentation (https://docs.litellm.ai/docs/providers)
        for available model options. Default is gemini-pro.

    kwargs: kwargs
        Any additional parameters to pass to litellm.completion method.
    """
    try:
        from litellm import completion
    except ImportError as e:
        raise ImportError(
            e.message
            + ". litellm is required for using LLM based pairwise orientation. Please install using: pip install litellm"
        )

    if system_prompt is None:
        system_prompt = "You are an expert in Causal Inference"

    prompt = f""" {system_prompt}. You are given two variables with the following descriptions:
        <A>: {descriptions[x]}
        <B>: {descriptions[y]}

        Which of the following two options is the most likely causal direction between them:
        1. <A> causes <B>
        2. <B> causes <A>

        Return a single letter answer between the choices above. I do not need the reasoning behind it. Do not add any formatting in the answer.
        """

    response = completion(
        model=llm_model, messages=[{"role": "user", "content": prompt}]
    )
    response = response.choices[0].message.content
    response_txt = response.strip().lower().replace("*", "")
    if response_txt in ("a", "1"):
        return (x, y)
    elif response_txt in ("b", "2"):
        return (y, x)
    else:
        raise ValueError(
            "Results from the LLM are unclear. Try calling the function again."
        )


def manual_pairwise_orient(x, y):
    """
    Generates a prompt for the user to input the direction between the variables.

    Parameters
    ----------
    x: str
        The first variable's name

    y: str
        The second variable's name
    """
    user_input = input(
        f"Select the edge direction between {x} and {y}. \n 1. {x} -> {y} \n 2. {x} <- {y} \n"
    )
    if user_input == "1":
        return (x, y)
    elif user_input == "2":
        return (y, x)


def preprocess_data(df):
    """
    Tries to figure out the data type of each variable `df`.

    Assigns one of (numerical, categorical unordered, categorical ordered) datatypes
    to each column in `df`. Also changes any object datatypes to categorical.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe.

    Returns
    -------
    (pd.DataFrame, dtypes): Tuple of transformed dataframe and a dictionary with inferred datatype of each column.
    """
    df = df.copy()
    dtypes = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("float")
            dtypes[col] = "N"
        elif pd.api.types.is_numeric_dtype(df[col]):
            dtypes[col] = "N"
        elif pd.api.types.is_object_dtype(df[col]):
            dtypes[col] = "C"
            df[col] = df[col].astype("category")
        elif isinstance(df[col].dtype, pd.CategoricalDtype):
            if df[col].dtype.ordered:
                dtypes[col] = "O"
            else:
                dtypes[col] = "C"
        else:
            raise ValueError(
                f"Couldn't infer datatype of column: {col} from data. Try specifying the appropriate datatype to the column."
            )

    logger.info(
        f" Datatype (N=numerical, C=Categorical Unordered, O=Categorical Ordered) inferred from data: \n {dtypes}"
    )
    return (df, dtypes)
