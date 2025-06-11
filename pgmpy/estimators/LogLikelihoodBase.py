import pandas as pd
from abc import ABC, abstractmethod
from pgmpy.models import (
    BayesianNetwork,
    DiscreteBayesianNetwork
)


class LogLikelihoodBase(ABC):
    """
    Base class for computing log-likelihood scores in Bayesian networks.
    
    This class provides a unified interface for computing log-likelihood scores
    across different types of Bayesian networks and data types. It serves as the
    foundation for specialized implementations that handle specific data types
    or model structures.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset against which to score the model.
        
    Attributes
    ----------
    data : pandas.DataFrame
        The dataset used for scoring.
    state_names : dict
        Dictionary mapping each variable to its possible states.
    dtypes : dict
        Dictionary mapping each variable to its data type ('N' for numeric,
        'C' for categorical).
    """
    
    def __init__(self, data):
        """
        Initialize the LogLikelihoodBase class.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The dataset against which to score the model.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "data must be a pandas.DataFrame instance"
            )
            
        self.data = data
        self.state_names = self._get_state_names()
        self.dtypes = self._get_dtypes()
        
    def _get_state_names(self):
        """
        Get the possible states for each variable in the dataset.
        
        Returns
        -------
        dict
            Dictionary mapping each variable to its possible states.
        """
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
        """
        Get the data type for each variable in the dataset.
        
        Returns
        -------
        dict
            Dictionary mapping each variable to its data type ('N' for numeric,
            'C' for categorical).
        """
        dtypes = {}
        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                dtypes[column] = 'N'
            else:
                dtypes[column] = 'C'
        return dtypes
        
    @abstractmethod
    def local_score(self, variable, parents):
        """
        Compute the local score for a variable given its parents.
        
        Parameters
        ----------
        variable : str
            The variable for which to compute the score.
        parents : list
            List of parent variables.
            
        Returns
        -------
        float
            The local score for the variable.
        """
        pass
        
    def score(self, model):
        """
        Compute the overall score for a Bayesian network model.
        
        Parameters
        ----------
        model : pgmpy.models.BayesianNetwork or
               pgmpy.models.DiscreteBayesianNetwork
            The Bayesian network model to score.
            
        Returns
        -------
        float
            The overall score for the model.
            
        Raises
        ------
        ValueError
            If the model is invalid or if there are missing variables in the data.
        """
        if not isinstance(model, (BayesianNetwork, DiscreteBayesianNetwork)):
            raise ValueError(
                "model must be a BayesianNetwork or "
                "DiscreteBayesianNetwork instance"
            )
        if self.data.empty:
            raise ValueError(
                "Input data is empty."
            )
        # Check if all variables in the model are present in the data
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
        # Compute the score for each variable
        score = 0
        for node in model.nodes():
            parents = list(model.predecessors(node))
            
            if hasattr(self, 'use_cpd') and getattr(self, 'use_cpd', False):
                score += self.local_score(node, parents, model=model)
            else:
                score += self.local_score(node, parents)
        return score 