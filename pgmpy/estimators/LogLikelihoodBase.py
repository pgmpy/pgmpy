import pandas as pd
from abc import ABC, abstractmethod
from pgmpy.models import (
    BayesianNetwork,
    DiscreteBayesianNetwork
)


class LogLikelihoodBase(ABC):
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "data must be a pandas.DataFrame instance"
            )
            
        self.data = data
        self.state_names = self._get_state_names()
        self.dtypes = self._get_dtypes()
        
    def _get_state_names(self):
        state_names = {}
        for column in self.data.columns:
            if self.data[column].dtype.name == 'category':
                state_names[column] = (
                    self.data[column].cat.categories.tolist()
                )
            else:
                state_names[column] = sorted(
                    self.data[column].unique().tolist()
                )
        return state_names
        
    def _get_dtypes(self):
        dtypes = {}
        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                dtypes[column] = 'N'
            else:
                dtypes[column] = 'C'
        return dtypes
        
    @abstractmethod
    def local_score(self, variable, parents):
        pass
        
    def score(self, model):
        if not isinstance(model, (BayesianNetwork, DiscreteBayesianNetwork)):
            raise ValueError(
                "model must be a BayesianNetwork or "
                "DiscreteBayesianNetwork instance"
            )
        if self.data.empty:
            raise ValueError(
                "Input data is empty."
            )
        missing_vars = set(model.nodes()) - set(self.data.columns)
        if missing_vars:
            raise ValueError(
                f"Missing variables in data: {missing_vars}"
            )
        
        if hasattr(model, 'get_cpds'):
            for node in model.nodes():
                cpd = model.get_cpds(node)
                if cpd is None:
                    raise ValueError(
                        f"Model is missing CPD for node: {node}"
                    )
        score = 0
        for node in model.nodes():
            parents = list(model.predecessors(node))
            
            if hasattr(self, 'use_cpd') and getattr(self, 'use_cpd', False):
                score += self.local_score(node, parents, model=model)
            else:
                score += self.local_score(node, parents)
        return score 