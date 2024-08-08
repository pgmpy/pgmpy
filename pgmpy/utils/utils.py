import gzip
import os

import google.generativeai as genai
import pandas as pd

try:
    from importlib.resources import files
except:
    # For python 3.8 and lower
    from importlib_resources import files


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
        "ecoli70": "",
        "magic-niab": "",
        "magic-irri": "",
        "arth150": "",
        "sangiovese": "",
        "mehra": "",
    }

    if model not in filenames.keys():
        raise ValueError("dataset should be one of the options")
    if filenames[model] == "":
        raise NotImplementedError("The specified dataset isn't available.")

    path = filenames[model]
    ref = files("pgmpy") / path
    with gzip.open(ref) as f:
        content = f.read()
    reader = BIFReader(string=content.decode("utf-8"), n_jobs=1)
    return reader.get_model()


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
    x, y, descriptions, domain=None, llm_model="gemini-1.5-flash", **kwargs
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

    domain: str
        The domain of the variables. The LLM is prompted to be an expert in the domain.

    llm: str (default: gemini)
        The LLM to use. Currently only Google's gemini is supported.
    """
    if llm_model.startswith("gemini"):
        if "GEMINI_API_KEY" not in os.environ:
            raise ValueError(
                "Please set GEMINI_API_KEY environment variable with the API key to use"
            )

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(model_name=llm_model)

        if domain == None:
            domain = "Causal Inference"

        prompt = f""" You are an expert in {domain}. You are given two variables with the following descriptions:
            <A>: {descriptions[x]}
            <B>: {descriptions[y]}

            Which of the following two options is the most likely causal direction between them:
            1. <A> causes <B>
            2. <B> causes <A>

            Return a single letter answer between the choices above. I do not need the reasoning behind it. Do not add any formatting in the answer.
            """
        response = model.generate_content([prompt])
        response_txt = response.text.strip().lower().replace("*", "")
        if response_txt in ("a", "1"):
            return (x, y)
        elif response_txt in ("b", "2"):
            return (y, x)
        else:
            raise ValueError(
                "Results from the LLM are unclear. Try calling the function again."
            )
