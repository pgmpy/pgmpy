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
        """`get_example_model` is deprecated and will be removed in v1.3.0. Please use `pgmpy.example_models.load_model`
        instead.""",
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


_transformers_pipeline_cache = {}


def _build_pairwise_orient_prompt(x, y, descriptions, system_prompt):
    """Build the prompt sent to the LLM for pairwise edge orientation."""
    if system_prompt is None:
        system_prompt = "You are an expert in Causal Inference"

    return f""" {system_prompt}. You are
      given two variables with the following descriptions:
        <A>: {descriptions[x]}
        <B>: {descriptions[y]}

        Which of the following two options is the most likely causal direction between them:
        1. <A> causes <B>
        2. <B> causes <A>

        Return a single number (1 or 2) as your answer. I do not need the reasoning behind it.
        Do not add any formatting in the answer.
        """


def _parse_pairwise_orient_response(response_text, x, y):
    """Parse the raw LLM response into an edge direction tuple."""
    if not response_text:
        raise ValueError(
            "Results from the LLM are unclear (received empty response). "
            "Try calling the function again."
        )

    response_txt = response_text.strip().lower().replace("*", "")

    # Strict match preserves the exact legacy behavior for short answers.
    if response_txt in ("a", "1"):
        return (x, y)
    if response_txt in ("b", "2"):
        return (y, x)

    # Tolerant fallback: some local models prepend whitespace or punctuation
    # before the answer character (e.g. "1.", " 2 ", "- 1"). Scan past leading
    # whitespace/punctuation and accept the first clear answer character.
    for ch in response_txt:
        if ch.isspace() or ch in ".,:;!?()[]{}\"'`#-":
            continue
        if ch == "1" or ch == "a":
            return (x, y)
        if ch == "2" or ch == "b":
            return (y, x)
        break

    raise ValueError(
        f"Results from the LLM are unclear. Got response: {response_text!r}. "
        "Try calling the function again."
    )


def _call_litellm(prompt, llm_model, **kwargs):
    """Send the prompt through the litellm hosted-model router."""
    try:
        from litellm import completion
    except ImportError as e:
        raise ImportError(
            f"{e}. litellm is required for the 'litellm' backend. "
            "Install with: pip install litellm, or choose a different backend "
            "(e.g. 'transformers', 'ollama', 'openai_compatible', or a callable)."
        ) from None

    response = completion(model=llm_model, messages=[{"role": "user", "content": prompt}], **kwargs)
    return response.choices[0].message.content


def _call_transformers(prompt, llm_model, **kwargs):
    """Run the prompt through a local HuggingFace transformers text-generation pipeline.

    Extra kwargs are forwarded to the pipeline *call* (generation kwargs such as
    ``max_new_tokens``, ``temperature``, ``do_sample``). To customize pipeline
    *construction* (``device``, ``torch_dtype``, ``tokenizer``, ...), pass
    ``pipeline_kwargs={...}`` via ``backend_kwargs``.
    """
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            f"{e}. transformers is required for the 'transformers' backend. "
            "Install with: pip install transformers, or choose a different backend."
        ) from None

    pipeline_kwargs = kwargs.pop("pipeline_kwargs", None)
    # Pipelines are expensive to construct — cache per model when no custom
    # construction kwargs are supplied. If the caller passes pipeline_kwargs we
    # skip the cache because the resulting pipeline may not be equivalent.
    cache_pipeline = not pipeline_kwargs
    if cache_pipeline and llm_model in _transformers_pipeline_cache:
        generator = _transformers_pipeline_cache[llm_model]
    else:
        generator = pipeline("text-generation", model=llm_model, **(pipeline_kwargs or {}))
        if cache_pipeline:
            _transformers_pipeline_cache[llm_model] = generator

    gen_kwargs = {"max_new_tokens": 16, "return_full_text": False, "do_sample": False}
    gen_kwargs.update(kwargs)

    outputs = generator(prompt, **gen_kwargs)
    first = outputs[0] if isinstance(outputs, list) else outputs
    if isinstance(first, dict) and "generated_text" in first:
        return first["generated_text"]
    return str(first)


def _call_ollama(prompt, llm_model, *, host="http://localhost:11434", timeout=120, options=None, **kwargs):
    """Send the prompt to a local Ollama server (https://ollama.com)."""
    try:
        import requests
    except ImportError as e:
        raise ImportError(
            f"{e}. requests is required for the 'ollama' backend. "
            "Install with: pip install requests, or choose a different backend."
        ) from None

    payload = {"model": llm_model, "prompt": prompt, "stream": False}
    if options is not None:
        payload["options"] = options
    payload.update(kwargs)

    url = host.rstrip("/") + "/api/generate"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def _call_openai_compatible(prompt, llm_model, *, base_url, api_key=None, timeout=120, **kwargs):
    """Send the prompt to any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Works with vLLM, llama.cpp-server, LM Studio, text-generation-webui,
    LiteLLM proxy, and other local servers that implement the same schema.
    """
    try:
        import requests
    except ImportError as e:
        raise ImportError(
            f"{e}. requests is required for the 'openai_compatible' backend. "
            "Install with: pip install requests, or choose a different backend."
        ) from None

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(kwargs)

    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


_BUILTIN_BACKENDS = {
    "litellm": _call_litellm,
    "transformers": _call_transformers,
    "ollama": _call_ollama,
    "openai_compatible": _call_openai_compatible,
}


def llm_pairwise_orient(
    x,
    y,
    descriptions,
    system_prompt=None,
    llm_model="gemini/gemini-1.5-flash",
    *,
    backend="litellm",
    backend_kwargs=None,
    **kwargs,
):
    """
    Asks a Large Language Model (LLM) for the
     orientation of an edge between `x` and `y`.

    The function builds a standard prompt and delegates the actual model call
    to one of several pluggable backends, so both hosted providers (via
    litellm) and locally-hosted models (HuggingFace transformers, Ollama,
    OpenAI-compatible servers, or any user-supplied callable) can be used.

    Parameters
    ----------
    x: str
        The first variable's name.

    y: str
        The second variable's name.

    descriptions: dict
        A dict of the form ``{variable: description}`` containing text
        descriptions of the variables.

    system_prompt: str, optional
        A system prompt to give the LLM.

    llm_model: str (default: "gemini/gemini-1.5-flash")
        The model identifier. Interpreted by the chosen backend:
          * ``"litellm"``: any provider/model string supported by litellm
            (https://docs.litellm.ai/docs/providers).
          * ``"transformers"``: a HuggingFace Hub model id or local path.
          * ``"ollama"``: a model tag pulled in Ollama, e.g. ``"llama3"``.
          * ``"openai_compatible"``: the ``model`` field the server expects.
        Ignored when ``backend`` is a callable.

    backend: str or callable (default: "litellm")
        Which backend to use to call the LLM. One of:
          * ``"litellm"``  -- requires ``pip install litellm``. Default,
            preserves the previous behavior.
          * ``"transformers"``  -- requires ``pip install transformers``
            (and a suitable backend like ``torch``). Runs a local
            text-generation pipeline.
          * ``"ollama"``  -- calls a local Ollama server; only requires
            ``requests``. Configure via ``backend_kwargs={"host": "..."}``.
          * ``"openai_compatible"``  -- posts to any OpenAI-compatible
            ``/v1/chat/completions`` endpoint (vLLM, llama.cpp-server,
            LM Studio, text-generation-webui, etc.). Only requires
            ``requests``. ``backend_kwargs`` must include ``base_url``;
            ``api_key`` is optional.
          * a callable  -- called as ``backend(prompt, **backend_kwargs)``
            and must return the raw text response as a string. Use this
            for any setup not covered above (e.g. a warm HuggingFace
            pipeline you want to reuse, a llama-cpp-python binding, or a
            direct Anthropic SDK call).

    backend_kwargs: dict, optional
        Backend-specific options. Forwarded to the backend function.
        Examples:
          * Ollama: ``{"host": "http://localhost:11434", "options": {"temperature": 0.0}}``
          * OpenAI-compatible: ``{"base_url": "http://localhost:8000/v1", "api_key": "sk-..."}``
          * transformers: ``{"max_new_tokens": 32, "pipeline_kwargs": {"device": 0}}``

    kwargs: kwargs
        Additional kwargs forwarded to the backend. For ``backend="litellm"``
        these are passed to ``litellm.completion`` (preserving historical
        behavior). For other backends they are merged with ``backend_kwargs``.

    Returns
    -------
    tuple:
        Returns a tuple ``(source, target)`` representing the edge direction.

    Examples
    --------
    Using the default litellm backend (requires a provider API key):

    >>> from pgmpy.utils import llm_pairwise_orient
    >>> descriptions = {"Age": "Age of a person", "Income": "Yearly income"}
    >>> llm_pairwise_orient("Age", "Income", descriptions)  # doctest: +SKIP
    ('Age', 'Income')

    Using a local Ollama server:

    >>> llm_pairwise_orient(
    ...     "Age", "Income", descriptions,
    ...     llm_model="llama3",
    ...     backend="ollama",
    ...     backend_kwargs={"host": "http://localhost:11434"},
    ... )  # doctest: +SKIP

    Using a local HuggingFace transformers pipeline:

    >>> llm_pairwise_orient(
    ...     "Age", "Income", descriptions,
    ...     llm_model="meta-llama/Llama-3.2-1B-Instruct",
    ...     backend="transformers",
    ... )  # doctest: +SKIP

    Using any OpenAI-compatible local server (vLLM / llama.cpp-server / LM Studio):

    >>> llm_pairwise_orient(
    ...     "Age", "Income", descriptions,
    ...     llm_model="llama-3-8b",
    ...     backend="openai_compatible",
    ...     backend_kwargs={"base_url": "http://localhost:8000/v1"},
    ... )  # doctest: +SKIP

    Using a custom callable (e.g. a pre-loaded pipeline reused across calls):

    >>> from transformers import pipeline
    >>> pipe = pipeline("text-generation", model="...", device=0)  # doctest: +SKIP
    >>> def my_backend(prompt, **_):
    ...     out = pipe(prompt, max_new_tokens=16, return_full_text=False)
    ...     return out[0]["generated_text"]
    >>> llm_pairwise_orient(
    ...     "Age", "Income", descriptions, backend=my_backend,
    ... )  # doctest: +SKIP
    """
    prompt = _build_pairwise_orient_prompt(x, y, descriptions, system_prompt)

    call_kwargs = dict(backend_kwargs or {})
    # Top-level kwargs are merged in for convenience. For the litellm backend
    # this preserves the historical behavior of accepting arbitrary completion
    # kwargs at the top level; for the other backends it just provides a
    # second place to put backend-specific options.
    call_kwargs.update(kwargs)

    if callable(backend):
        response_text = backend(prompt, **call_kwargs)
    elif isinstance(backend, str) and backend in _BUILTIN_BACKENDS:
        response_text = _BUILTIN_BACKENDS[backend](prompt, llm_model, **call_kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Expected one of {sorted(_BUILTIN_BACKENDS)} or a callable."
        )

    return _parse_pairwise_orient_response(response_text, x, y)


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
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     [
    ...         [1, 1, 0, 0, 0, 0, 0, 1, 0],
    ...         [0, 2, 0, 1, 1, 1, 1, 1, 1],
    ...     ],
    ...     columns=[
    ...         ("D", 0), ("G" , 0), ("I" , 0),
    ...         ("D", 1), ("G", 1),
    ...         ("D", 2), ("G", 2),
    ...         ("I", 1), ("I", 2)
    ...     ],
    ... )

    For input dataframe `df`, represented in the wide format

      (D, 0) (G, 0) (I, 0) (D, 1) (G, 1) (D, 2) (G, 2) (I, 1) (I, 2)
    0      1      1      0      0      0      0      0      1      0
    1      0      2      0      1      1      1      1      1      1

    >>> to_timeseries_format(df, return_format="numpy3d")
    array([[[1, 0, 0],
            [1, 0, 0],
            [0, 1, 0]],
            [[0, 1, 1],
            [2, 1, 1],
            [0, 1, 1]]])

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
    variable D     G     I
    time     0 1 2 0 1 2 0 1 2
    0        1 0 0 1 0 0 0 1 0
    1        0 1 1 2 1 1 0 1 1
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
