import statsmodels.api as sm
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor
from statsmodels.sandbox.regression.gmm import IV2SLS

from pgmpy.base import DAG, PDAG
from pgmpy.estimators import PC
from pgmpy.inference import CausalInference
from pgmpy.models import LinearGaussianBayesianNetwork as LGBN


class CausalGraphWithVariableRoles:
    """
    Class for specifying a Causal Graph.

    If `graph` is not provided, a causal discovery algorithm is
    used to learn the DAG.

    Parameters
    ----------
    graph: pgmpy.models.DAG or Causal Discovery Algorithm instance.
        The DAG representing the causal graph over all relevant variables.

    exposure: str
        The exposure or treatment variable.

    outcome: str
        The outcome or target variable.

    **kwargs: dict
        Additional parameters for the estimators.

    Examples
    --------
    >>> # Model specification with a specified DAG.
    >>> model = CausalGraphWithVariableRoles(
    ...     graph=DAG.from_dagitty("dag{ I -> A <- U -> B A -> B }"),
    ...     exposure="A",
    ...     outcome="B",
    ... )

    >>> # Model specification with a causal discovery algorithm.
    >>> model = CausalGraphWithVariableRoles(
    ...     graph=PC(df_train), exposure="A", outcome="B"
    ... )
    """

    def __init__(self, graph, exposure, outcome, **kwargs):
        if isinstance(graph, DAG):
            self.graph = graph
        else:
            self.graph = graph.estimate(**kwargs)
            if isinstance(self.graph, PDAG):
                self.graph = self.graph.to_dag()

        self.exposure = exposure
        self.outcome = outcome

        self.infer = CausalInference(self.graph)

    def get_adjustment_set(self):
        """
        Returns the adjustment set for estimating the causal effect of exposure on outcome.

        Returns
        -------
        set: adjustment set
            The set of variables to control for to estimate the effect of `exposure` on `outcome`.
        """
        return self.infer.get_minimal_adjustment_set(self.exposure, self.outcome)

    def get_frontdoor_set(self):
        """
        Returns the frontdoor set for estimating the causal effect of exposure on outcome.

        Returns
        -------
        set: frontdoor set
            The set of variables that can act as frontdoor adjustment for estimating the effect of exposure on outcome.
        """
        return self.infer.get_all_frontdoor_adjustment_sets(
            self.exposure, self.outcome
        )[0]

    def get_ivs(self):
        """
        Returns the Instrumental variables for estimating the causal effect of exposure on outcome.
        """
        return self.infer.get_ivs(self.exposure, self.outcome)


class RCT:
    """
    Class for defining a Randomized Controlled Trial (RCT).

    The class internally utilizes it to construct a causal DAG, and uses that
    for causal prediction.

    Parameters
    ----------
    treatment: str
        The treatment variable

    covariates: set
        The set of covariates for the RCT.

    randomization_var: set
        The randomization variables for the RCT. The randomization variables
        shouldn't have any effect on the outcome variable. If it has, please
        add it to covariates.

    Examples
    --------
    >>> model = RCT(treatment="A", covariates={"U"}, randomization_var="I")
    """

    def __init__(self, treatment, covariates, randomization_var):
        self.exposure = treatment
        self.covariates = covariates
        self.randomization_var = randomization_var

    def get_adjustment_set(self):
        return self.covariates

    def get_frontdoor_set(self):
        # There can't be any frontdoor set as there are no mediator variables.
        return None

    def get_ivs(self):
        # If imperfect randomization.
        return self.randomization_var


class AdjustmentRegressor:
    """
    Causal Predictions using adjustment variables.

    Parameters
    ----------
    model: CausalGraphWithVariableRoles or RCT
        The model to use for the causal predictions.

    estimator: str (ols or dml)
        The statistical estimator to use for predictions.

    Examples
    --------
    >>> # AdjustmentRegressor  on an RCT model with a linear regression estimator.
    >>> model = RCT(treatment="A", covariates={"U"}, randomization_var="I")

    >>> causal_regressor = AdjustmentRegressor(model, estimator="ols")
    >>> causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    >>> pred1_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    >>> # AdjustmentRegressor  on an RCT model with a Double ML estimator.
    >>> causal_regressor = AdjustmentRegressor(model, estimator="dml")
    >>> causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    >>> pred1_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    >>> # AdjustmentRegressor on a CausalGraphWithVariableRoles with a linear regression estimator.
    >>> model = CausalGraphWithVariableRoles(
    ...     graph=DAG.from_dagitty("dag{ I -> A <- U -> B A -> B }"),
    ...     exposure="A",
    ...     outcome="B",
    ... )
    >>> causal_regressor = AdjustmentRegressor(model, estimator="ols")
    >>> causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    >>> pred2_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    """

    def __init__(self, model, estimator):
        self.adjustment_set = list(model.get_adjustment_set())
        self.estimator = estimator

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: pd.DataFrame
            A dataframe with data on all the variables in the model except the outcome variable.

        y: pd.Series
            A series with data on the outcome variable.
        """
        if self.estimator == "ols":
            self.est = sm.OLS(y, X.loc[:, [model.exposure] + self.adjustment_set]).fit()
        elif self.estimator == "dml":
            df = X.copy()
            df["outcome"] = y
            dml_data = DoubleMLData(
                df,
                y_col="outcome",
                d_cols=model.exposure,
                x_cols=list(set(df.columns) - {"outcome", model.exposure}),
            )
            ml_g = RandomForestRegressor()
            ml_m = RandomForestRegressor()

            self.est = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=5)
            self.est.fit()

    def predict(self, X):
        """
        Parameters
        ----------
        X: pd.DataFrame
            A dataframe with data on all the variables in the model except the outcome variable.
        """
        return self.est.predict(X.loc[:, [model.exposure] + self.adjustment_set])


class IVRegressor:
    """
    Causal Predictions using Instrumental variables.

    Parameters
    ----------
    model: CausalGraphWithVariableRoles or RCT
        The model to use for the causal predictions.

    estimator: IV2SLS
        The statistical estimator to use for predictions.

    Examples
    --------
    >>> # IV prediction using an RCT model.
    >>> model = RCT(treatment="A", covariates={"U"}, randomization_var="I")
    >>> causal_regressor = IVRegressor(model, estimator=IV2SLS)
    >>> causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    >>> pred1_iv = causal_regressor.predict(df_test.drop(["B"], axis=1))

    >>> # IV prediction using a CausalGraphWithVariableRoles model.
    >>> model = CausalGraphWithVariableRoles(
    ...     graph=DAG.from_dagitty("dag{ I -> A <- U -> B A -> B }"),
    ...     exposure="A",
    ...     outcome="B",
    ... )
    >>> causal_regressor = IVRegressor(model, estimator=IV2SLS)
    >>> causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    >>> pred2_iv = causal_regressor.predict(df_test.drop(["B"], axis=1))
    """

    def __init__(self, model, estimator):
        self.ivs = list(model.get_ivs())
        if len(self.ivs) == 0:
            raise ValueError("No IVs, use another estimator")
        self.estimator = estimator

    def fit(self, X, y):
        self.est = self.estimator(
            endog=y,
            exog=X.loc[:, [model.exposure]],
            instrument=X.loc[:, [model.exposure] + self.ivs],
        ).fit()

    def predict(self, X):
        return self.est.predict(X.loc[:, [model.exposure]])


if __name__ == "__main__":
    # The assumed causal model is: I -> A <- U -> B; A -> B.
    dag = LGBN.from_dagitty("dag{ I -> A <- U -> B A-> B }")
    dag.add_cpds(*dag.get_random_cpds(seed=42))
    sim_df = dag.simulate(int(1e3), seed=42)

    df_train = sim_df.iloc[:800, :]
    df_test = sim_df.iloc[800:, :]

    # Case 1: Model defined using RCT class.
    model = RCT(treatment="A", covariates={"U"}, randomization_var="I")

    ### Prediction using adjustment variables
    causal_regressor = AdjustmentRegressor(model, estimator="ols")
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred1_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    causal_regressor = AdjustmentRegressor(model, estimator="dml")
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred1_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    ### Prediction using IVs
    causal_regressor = IVRegressor(model, estimator=IV2SLS)
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred1_iv = causal_regressor.predict(df_test.drop(["B"], axis=1))

    # Case 2: Model defined using Casual Graph.
    model = CausalGraphWithVariableRoles(
        graph=DAG.from_dagitty("dag{ I -> A <- U -> B A -> B }"),
        exposure="A",
        outcome="B",
    )

    ### Prediction using adjustment variables
    causal_regressor = AdjustmentRegressor(model, estimator="ols")
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred2_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    ### Prediction using IVs
    causal_regressor = IVRegressor(model, estimator=IV2SLS)
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred2_iv = causal_regressor.predict(df_test.drop(["B"], axis=1))

    # Case 3: Model defined using a causal discovery method.
    model = CausalGraphWithVariableRoles(graph=PC(df_train), exposure="A", outcome="B")

    ### Prediction using adjustment variables
    causal_regressor = AdjustmentRegressor(model, estimator="ols")
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred3_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    ### Prediction using IVs
    causal_regressor = IVRegressor(model, estimator=IV2SLS)
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred3_iv = causal_regressor.predict(df_test.drop(["B"], axis=1))
