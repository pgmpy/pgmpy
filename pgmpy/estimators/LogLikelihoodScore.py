import numpy as np
from pgmpy.estimators.LogLikelihoodBase import LogLikelihoodBase


class LogLikelihoodScore(LogLikelihoodBase):
    def __init__(self, data, use_cpd=False):
        super(LogLikelihoodScore, self).__init__(data)
        self.use_cpd = use_cpd
        
    def local_score(self, variable, parents, model=None):
        data = self.data[[variable] + parents].copy()
        
        data = data.dropna()
        if len(data) == 0:
            return 0.0
            
        if self.use_cpd:
            if model is None:
                raise ValueError("model must be provided when use_cpd=True")
            cpd = model.get_cpds(variable)
            if cpd is None:
                raise ValueError(f"No CPD found for variable {variable}")
            return self._cpd_score(cpd, data, variable, parents)
        else:
            if self.dtypes[variable] == 'N':
                return self._numeric_score(data, variable, parents)
            else:
                return self._categorical_score(data, variable, parents)
            
    def _cpd_score(self, cpd, data, variable, parents):
        values = cpd.values
        log_prob = np.log(values)
        return np.sum(log_prob)
            
    def _numeric_score(self, data, variable, parents):
        if not parents:
            mean = data[variable].mean()
            var = data[variable].var()
            if var == 0:
                var = 1e-10
            score = -0.5 * len(data) * (
                np.log(2 * np.pi * var) + 
                ((data[variable] - mean) ** 2 / var).sum() / len(data)
            )
        else:
            score = 0
            for parent_state, group in data.groupby(parents):
                if len(group) > 1:
                    mean = group[variable].mean()
                    var = group[variable].var()
                    if var == 0:
                        var = 1e-10
                    score -= 0.5 * len(group) * (
                        np.log(2 * np.pi * var) + 
                        ((group[variable] - mean) ** 2 / var).sum() / len(group)
                    )
        return score
        
    def _categorical_score(self, data, variable, parents):
        if not parents:
            counts = data[variable].value_counts()
            probs = counts / len(data)
            score = (counts * np.log(probs)).sum()
        else:
            score = 0
            for parent_state, group in data.groupby(parents):
                if len(group) > 0:
                    counts = group[variable].value_counts()
                    probs = counts / len(group)
                    score += (counts * np.log(probs)).sum()
        return score 