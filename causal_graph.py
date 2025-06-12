from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.models import BayesianNetwork

class CausalGraphLearner:
    def __init__(self):
        self.model = None

    def learn_structure(self, data):
        hc = HillClimbSearch(data)
        best_model = hc.estimate(scoring_method=BicScore(data))
        self.model = BayesianNetwork(best_model.edges())
        return self.model

    def fit_parameters(self, data):
        self.model.fit(data, estimator=BayesianEstimator)

    def get_model(self):
        return self.model