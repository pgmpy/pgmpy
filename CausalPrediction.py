import statsmodels.api as sm

# from doubleml import DoubleMLData, DoubleMLPLR
from statsmodels.sandbox.regression.gmm import IV2SLS

from pgmpy.base import DAG, PDAG
from pgmpy.estimators import PC
from pgmpy.inference import CausalInference
from pgmpy.models import LinearGaussianBayesianNetwork as LGBN


class CausalGraphWithVariableRoles:
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
        return self.infer.get_minimal_adjustment_set(self.exposure, self.outcome)

    def get_frontdoor_set(self):
        return self.infer.get_all_frontdoor_adjustment_sets(
            self.exposure, self.outcome
        )[0]

    def get_ivs(self):
        return self.infer.get_ivs(self.exposure, self.outcome)


class RCT:
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
    def __init__(self, model, estimator):
        self.adjustment_set = list(model.get_adjustment_set())
        self.estimator = estimator

    def fit(self, X, y):
        self.est = self.estimator(
            y, X.loc[:, [model.exposure] + self.adjustment_set]
        ).fit()

    def predict(self, X):
        return self.est.predict(X.loc[:, [model.exposure] + self.adjustment_set])


class IVRegressor:
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

    # Case 1
    model = RCT(treatment="A", covariates={"U"}, randomization_var="I")

    ### Using adjustment variables
    causal_regressor = AdjustmentRegressor(model, estimator=sm.OLS)
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred1_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    ### Using IVs
    causal_regressor = IVRegressor(model, estimator=IV2SLS)
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred1_iv = causal_regressor.predict(df_test.drop(["B"], axis=1))

    # Case 2
    model = CausalGraphWithVariableRoles(
        graph=DAG.from_dagitty("dag{ I -> A <- U -> B A -> B }"),
        exposure="A",
        outcome="B",
    )

    ### Using adjustment variables
    causal_regressor = AdjustmentRegressor(model, estimator=sm.OLS)
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred2_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    ### Using IVs
    causal_regressor = IVRegressor(model, estimator=IV2SLS)
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred2_iv = causal_regressor.predict(df_test.drop(["B"], axis=1))

    # Case 3
    model = CausalGraphWithVariableRoles(graph=PC(df_train), exposure="A", outcome="B")

    ### Using adjustment variables
    causal_regressor = AdjustmentRegressor(model, estimator=sm.OLS)
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred3_adj = causal_regressor.predict(df_test.drop(["B"], axis=1))

    ### Using IVs
    causal_regressor = IVRegressor(model, estimator=IV2SLS)
    causal_regressor.fit(X=df_train.drop(["B"], axis=1), y=df_train["B"])
    pred3_iv = causal_regressor.predict(df_test.drop(["B"], axis=1))
