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

    model_links = {
        "asia": "http://www.bnlearn.com/bnrepository/asia/asia.bif.gz",
        "cancer": "http://www.bnlearn.com/bnrepository/cancer/cancer.bif.gz",
        "earthquake": "http://www.bnlearn.com/bnrepository/earthquake/earthquake.bif.gz",
        "sachs": "http://www.bnlearn.com/bnrepository/sachs/sachs.bif.gz",
        "survey": "http://www.bnlearn.com/bnrepository/survey/survey.bif.gz",
        "alarm": "http://www.bnlearn.com/bnrepository/alarm/alarm.bif.gz",
        "barley": "http://www.bnlearn.com/bnrepository/barley/barley.bif.gz",
        "child": "http://www.bnlearn.com/bnrepository/child/child.bif.gz",
        "insurance": "http://www.bnlearn.com/bnrepository/insurance/insurance.bif.gz",
        "mildew": "http://www.bnlearn.com/bnrepository/mildew/mildew.bif.gz",
        "water": "http://www.bnlearn.com/bnrepository/water/water.bif.gz",
        "hailfinder": "http://www.bnlearn.com/bnrepository/hailfinder/hailfinder.bif.gz",
        "hepar2": "http://www.bnlearn.com/bnrepository/hepar2/hepar2.bif.gz",
        "win95pts": "http://www.bnlearn.com/bnrepository/win95pts/win95pts.bif.gz",
        "andes": "http://www.bnlearn.com/bnrepository/andes/andes.bif.gz",
        "diabetes": "http://www.bnlearn.com/bnrepository/diabetes/diabetes.bif.gz",
        "link": "http://www.bnlearn.com/bnrepository/link/link.bif.gz",
        "munin1": "http://www.bnlearn.com/bnrepository/munin4/munin1.bif.gz",
        "munin2": "http://www.bnlearn.com/bnrepository/munin4/munin2.bif.gz",
        "munin3": "http://www.bnlearn.com/bnrepository/munin4/munin3.bif.gz",
        "munin4": "http://www.bnlearn.com/bnrepository/munin4/munin4.bif.gz",
        "pathfinder": "http://www.bnlearn.com/bnrepository/pathfinder/pathfinder.bif.gz",
        "pigs": "http://www.bnlearn.com/bnrepository/pigs/pigs.bif.gz",
        "munin": "http://www.bnlearn.com/bnrepository/munin/munin.bif.gz",
        "ecoli70": "",
        "magic-niab": "",
        "magic-irri": "",
        "arth150": "",
        "sangiovese": "",
        "mehra": "",
    }

    if model not in model_links.keys():
        raise ValueError("dataset should be one of the options")
    if model_links[model] == "":
        raise NotImplementedError("The specified dataset isn't supported")

    filename, _ = urlretrieve(model_links[model])
    with gzip.open(filename, "rb") as f:
        content = f.read()
    reader = BIFReader(string=content.decode("utf-8"), n_jobs=1)
    return reader.get_model()
