import gzip
from urllib.request import urlretrieve


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
        "asia": "pgmpy/utils/example_models/asia.bif.gz",
        "cancer": "pgmpy/utils/example_models/cancer.bif.gz",
        "earthquake": "pgmpy/utils/example_models/earthquake.bif.gz",
        "sachs": "pgmpy/utils/example_models/sachs.bif.gz",
        "survey": "pgmpy/utils/example_models/survey.bif.gz",
        "alarm": "pgmpy/utils/example_models/alarm.bif.gz",
        "barley": "pgmpy/utils/example_models/barley.bif.gz",
        "child": "pgmpy/utils/example_models/child.bif.gz",
        "insurance": "pgmpy/utils/example_models/insurance.bif.gz",
        "mildew": "pgmpy/utils/example_models/mildew.bif.gz",
        "water": "pgmpy/utils/example_models/water.bif.gz",
        "hailfinder": "pgmpy/utils/example_models/hailfinder.bif.gz",
        "hepar2": "pgmpy/utils/example_models/hepar2.bif.gz",
        "win95pts": "pgmpy/utils/example_models/win95pts.bif.gz",
        "andes": "pgmpy/utils/example_models/andes.bif.gz",
        "diabetes": "pgmpy/utils/example_models/diabetes.bif.gz",
        "link": "pgmpy/utils/example_models/link.bif.gz",
        "munin1": "pgmpy/utils/example_models/munin1.bif.gz",
        "munin2": "pgmpy/utils/example_models/munin2.bif.gz",
        "munin3": "pgmpy/utils/example_models/munin3.bif.gz",
        "munin4": "pgmpy/utils/example_models/munin4.bif.gz",
        "pathfinder": "pgmpy/utils/example_models/pathfinder.bif.gz",
        "pigs": "pgmpy/utils/example_models/pigs.bif.gz",
        "munin": "pgmpy/utils/example_models/munin.bif.gz",
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
        raise NotImplementedError("The specified dataset isn't supported")

    with gzip.open(filenames[model], "rb") as f:
        content = f.read()
    reader = BIFReader(string=content.decode("utf-8"), n_jobs=1)
    return reader.get_model()
