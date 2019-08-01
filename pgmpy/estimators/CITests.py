import numpy as np
import pandas as pd

from scipy import stats


def pearsonr(X, Y, Z, data):
    """
    Computes Pearson correlation coefficient and p-value for testing non-correlation. Should be used
    only on continuous data. In case when :math:`Z != \null` uses linear regression and computes pearson
    coefficient on residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X _|_ Y | Z

    Y: str
        The second variable for testing the independence condition X _|_ Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X _|_ Y | Z

    data: pandas.DataFrame
        The dataset in which to test the indepenedence condition.

    Returns
    -------
    Pearson's correlation coefficient: float
    p-value: float
    """
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(
            "Variable Z. Expected type: iterable. Got type: {t}".format(t=type(Z))
        )
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "Variable data. Expected type: pandas.DataFrame. Got type: {t}".format(
                t=type(data)
            )
        )

    # Step 2: If Z is empty compute a non-conditional test.
    if len(Z) == 0:
        return stats.pearsonr(data.loc[:, X], data.loc[:, Y])

    # Step 3: If Z is non-empty, use linear regression to compute residuals and test independence on it.
    else:
        X_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, X], rcond=None)[0]
        Y_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, Y], rcond=None)[0]

        residual_X = data.loc[:, X] - data.loc[:, Z].dot(X_coef)
        residual_Y = data.loc[:, X] - data.loc[:, Z].dot(Y_coef)

        return stats.pearsonr(residual_X, residual_Y)


def test_conditional_independence(X, Y, Z, data):
    return pearsonr(X=X, Y=Y, Z=Z, data=data)
