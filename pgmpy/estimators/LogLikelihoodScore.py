import numpy as np
import pandas as pd
from pgmpy.estimators.LogLikelihoodBase import LogLikelihoodBase
from pgmpy.factors.discrete import TabularCPD


class LogLikelihoodScore(LogLikelihoodBase):
    """
    Class for computing log-likelihood scores in Bayesian networks.
    
    This class provides a unified implementation for computing log-likelihood
    scores across different types of Bayesian networks and data types.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset against which to score the model.
    """
    
    def __init__(self, data, use_cpd=False):
        super(LogLikelihoodScore, self).__init__(data)
        self.use_cpd = use_cpd
        
    def local_score(self, variable, parents, model=None):
        """
        Compute the local score for a variable given its parents.
        
        Parameters
        ----------
        variable : str
            The variable for which to compute the score.
        parents : list
            List of parent variables.
        model : pgmpy.models.BayesianNetwork, optional
            The model containing CPDs. Required if use_cpd=True.
            
        Returns
        -------
        float
            The local score for the variable.
            
        Raises
        ------
        ValueError
            If use_cpd=True but model is not provided or if the model
            doesn't have a CPD for the variable.
        """
        # Get the data for the variable and its parents
        data = self.data[[variable] + parents].copy()
        
        # Handle missing values
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
            # Compute the score based on data type
            if self.dtypes[variable] == 'N':
                return self._numeric_score(data, variable, parents)
            else:
                return self._categorical_score(data, variable, parents)
            
    def _cpd_score(self, cpd, data, variable, parents):
        
        values = cpd.values
        
        log_prob = np.log(values)
        
        return np.sum(log_prob)
            
    def _numeric_score(self, data, variable, parents):
        """
        Compute score for numeric variables.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data containing the variable and its parents.
        variable : str
            The variable for which to compute the score.
        parents : list
            List of parent variables.
            
        Returns
        -------
        float
            The local score for the numeric variable.
        """
        if not parents:
            # If no parents, compute unconditional score
            mean = data[variable].mean()
            var = data[variable].var()
            if var == 0:
                var = 1e-10  # Avoid division by zero
            score = -0.5 * len(data) * (
                np.log(2 * np.pi * var) + 
                ((data[variable] - mean) ** 2 / var).sum() / len(data)
            )
        else:
            # If parents exist, compute conditional score
            score = 0
            for parent_state, group in data.groupby(parents):
                if len(group) > 1:  # Need at least 2 points for variance
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
        """
        Compute score for categorical variables.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data containing the variable and its parents.
        variable : str
            The variable for which to compute the score.
        parents : list
            List of parent variables.
            
        Returns
        -------
        float
            The local score for the categorical variable.
        """
        if not parents:
            # If no parents, compute unconditional score
            counts = data[variable].value_counts()
            probs = counts / len(data)
            score = (counts * np.log(probs)).sum()
        else:
            # If parents exist, compute conditional score
            score = 0
            for parent_state, group in data.groupby(parents):
                if len(group) > 0:
                    counts = group[variable].value_counts()
                    probs = counts / len(group)
                    score += (counts * np.log(probs)).sum()
        return score 