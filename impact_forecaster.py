from .causal_graph import CausalGraphLearner
from .intervention_engine import simulate_intervention
from .counterfactual import estimate_counterfactual

class CausalImpactForecaster:
    def __init__(self):
        self.graph_learner = CausalGraphLearner()
        self.model = None

    def fit(self, X, y):
        data = X.copy()
        data['target'] = y
        self.model = self.graph_learner.learn_structure(data)
        self.graph_learner.fit_parameters(data)

    def simulate_intervention(self, intervention):
        return simulate_intervention(self.graph_learner.get_model(), intervention)

    def counterfactual(self, original_data, intervention_data):
        return estimate_counterfactual(original_data, intervention_data, 'target')
