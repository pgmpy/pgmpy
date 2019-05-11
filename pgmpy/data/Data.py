import gzip
from urllib.request import urlretrieve

import pandas as pd
import numpy as np
from scipy import stats

from pgmpy.readwrite import BIFReader


class Data(object):
    """
    Base Data class.
    """

    def __init__(self, data, variables=None):
        """
        Data class for representing and doing stastical tests on data.

        Parameters
        ----------
        data: 2-D array-like or pandas.DataFrame
            The dataset for which the Data class should be initialized. If `df` is:
            2-D array-like: `variables` needs to be specified.
            pandas.DataFrame: If `variables` is specified, pgmpy changes the column
                names of `df` to `variables`.

        variables: list, array-like (1-D)
            List of variable names. The variable names are applied to `data` in order.

        Examples
        --------
        >>> from pgmpy.data import Data
        >>> df = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                                     columns=['A', 'B', 'C', 'D', 'E'])
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
            self.variables = data.columns
        else:
            data = np.array(data)
            if data.ndim != 2:
                raise ValueError(
                    "data must be a 2-D array or a pandas.DataFrame instance"
                )
            self.data = pd.DataFrame(data, columns=variables)
            self.variables = variables

    def test_independence(self, var1, var2, conditioned_vars=[], test="chi-square"):
        """
        Test the conditon (var1 _|_ var2 | conditioned_vars) in the data.

        Parameters
        ----------
        var1: str or int
            The first variable when testing for the condition: var1 _|_ var2 | conditioned_vars

        var2: str or int
            The second variable when testing for the condition: var1 _|_ var2 | conditioned_vars

        test: str
            The type of test. Options are:
                1. 'chi-square': Chi-Squared test

        conditioned_vars: list
            List of conditioned variables in the condition: var1 _|_ var2 | conditioned_vars.

        Examples
        --------
        >>> from pgmpy.data import Data
        >>> df = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                   columns=['A', 'B', 'C', 'D', 'E'])
        >>> df.test_independence(var1='A', var2='B', test='chi-square')
        >>> df.test_independence(var1='A', var2='B', conditioned_vars=['C', 'D'], test='chi-square')
        """
        if test == "chi-square":
            if not conditioned_vars:
                observed = pd.crosstab(self.data[var1], self.data[var2])
                chi_stat, p_value, dof, _ = stats.chi2_contingency(observed)

            else:
                observed_combinations = (
                    self.data.groupby(conditioned_vars).size().reset_index()
                )
                chi_stat = 0
                dof = 0
                for combination in range(len(observed_combinations)):
                    df_conditioned = self.data.copy()
                    for condition_var in conditioned_vars:
                        df_conditioned = df_conditioned.loc[
                            df_conditioned.loc[:, condition_var]
                            == observed_combinations.loc[combination, condition_var]
                        ]
                    observed = pd.crosstab(df_conditioned[var1], df_conditioned[var2])
                    chi, _, freedom, _ = stats.chi2_contingency(observed)
                    chi_stat += chi
                    dof += freedom
                p_value = 1.0 - stats.chi2.cdf(x=chi_stat, df=dof)
            return chi_stat, p_value, dof

    def cov_matrix(self):
        """
        Returns the covariance matrix of the given dataset

        Returns
        -------
        pd.DataFrame: Dataframe with the covarince values.
        """
        return self.data.cov()


def get_dataset(dataset):
    """
    Fetches the `dataset` and returns a pgmpy.model instance.

    Parameter
    ---------
    dataset: str
        Any dataset from bnlearn repository (http://www.bnlearn.com/bnrepository).

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
    >>> from pgmpy.data import get_dataset
    >>> model = get_dataset(dataset='asia')
    >>> model

    Returns
    -------
    pgmpy.models instance: An instance of one of the model classes in pgmpy.models
                           depending on the type of dataset.
    """
    dataset_links = {
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

    if dataset not in dataset_links.keys():
        raise ValueError("dataset should be one of the options")
    if dataset_links[dataset] == "":
        raise NotImplementedError("The specified dataset isn't supported")

    filename, _ = urlretrieve(dataset_links[dataset])
    with gzip.open(filename, "rb") as f:
        content = f.read()
    reader = BIFReader(content)
    return reader.get_model()
