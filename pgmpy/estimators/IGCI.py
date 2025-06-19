"""
Information Geometric Causal Inference (IGCI) for causal direction determination.

IGCI is a bivariate causal discovery method that determines causal direction based on
the independence between cause mechanism and cause distribution. It works with continuous
variables and assumes a deterministic relationship between cause and effect.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Union, Optional, List, Dict

from pgmpy.estimators.base import BaseEstimator


class IGCI(BaseEstimator):
    """
    Information Geometric Causal Inference (IGCI) for determining causal direction.

    IGCI is based on the principle that if X causes Y, then the distribution of the cause 
    P(X) and the causal mechanism mapping X to Y are independent. This leads to certain 
    properties of the joint distribution that can be exploited to determine the causal 
    direction.

    Parameters
    ----------
    data: pandas DataFrame
        DataFrame containing the variables. Should have at least two columns.
    
    assume_normalized: bool, default=False
        If True, assumes the data is already normalized to [0, 1]. Otherwise, 
        data will be normalized during computation.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import IGCI
    >>> # Generate some cause-effect data
    >>> x = np.random.uniform(0, 5, 1000)
    >>> y = x**2 + 0.1 * np.random.normal(0, 1, 1000)
    >>> data = pd.DataFrame({'X': x, 'Y': y})
    >>> igci = IGCI(data)
    >>> direction = igci.estimate_direction('X', 'Y')
    >>> print(f"Causal direction: {'X->Y' if direction == 1 else 'Y->X'}")
    
    References
    ----------
    [1] P. Daniusis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel, K. Zhang, 
        B. Schölkopf: "Inferring deterministic causal relations" (2010)
        https://arxiv.org/abs/1203.3475
    """

    def __init__(self, data: pd.DataFrame, assume_normalized: bool = False):
        """
        Initialize the IGCI estimator.
        """
        super(IGCI, self).__init__(data=data)
        self.assume_normalized = assume_normalized

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range.

        Parameters
        ----------
        data: numpy.ndarray
            Data to normalize.

        Returns
        -------
        numpy.ndarray
            Normalized data in range [0, 1].
        """
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Avoid division by zero if all values are the same
        if max_val == min_val:
            return np.zeros_like(data)
        
        return (data - min_val) / (max_val - min_val)

    def _compute_entropy_based_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute IGCI score using the entropy-based estimator.

        Parameters
        ----------
        x: numpy.ndarray
            First variable (potential cause).
        y: numpy.ndarray
            Second variable (potential effect).

        Returns
        -------
        float
            IGCI score. Positive value suggests X->Y, negative suggests Y->X.
        """
        if not self.assume_normalized:
            x = self._normalize_data(x)
            y = self._normalize_data(y)

        x = np.clip(x, 1e-10, 1 - 1e-10)
        y = np.clip(y, 1e-10, 1 - 1e-10)
        
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        
        dx = np.diff(x_sorted)
        dy = np.diff(y_sorted)
        
        log_dx = np.log(dx)
        log_dy = np.log(dy)
        
        x_to_y = -np.mean(log_dy)
        y_to_x = -np.mean(log_dx)
        
        return x_to_y - y_to_x

    def _compute_slope_based_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute IGCI score using the slope-based estimator.

        Parameters
        ----------
        x: numpy.ndarray
            First variable (potential cause).
        y: numpy.ndarray
            Second variable (potential effect).

        Returns
        -------
        float
            IGCI score. Positive value suggests X->Y, negative suggests Y->X.
        """
        if not self.assume_normalized:
            x = self._normalize_data(x)
            y = self._normalize_data(y)

        x_indices = np.argsort(x)
        y_indices = np.argsort(y)
        
        x_sorted = x[x_indices]
        y_sorted = y[x_indices] 
        
        x_by_y = x[y_indices]  
        y_by_y = y[y_indices]
        
        dx = np.diff(x_sorted)
        dy = np.diff(y_sorted)
        dx_by_y = np.diff(x_by_y)
        dy_by_y = np.diff(y_by_y)
        
        valid_x_to_y = (dx > 0)
        valid_y_to_x = (dy_by_y > 0)
        
        slopes_x_to_y = np.abs(dy[valid_x_to_y] / dx[valid_x_to_y])
        slopes_y_to_x = np.abs(dx_by_y[valid_y_to_x] / dy_by_y[valid_y_to_x])
        
        x_to_y = np.mean(np.log(slopes_x_to_y)) if len(slopes_x_to_y) > 0 else 0
        y_to_x = np.mean(np.log(slopes_y_to_x)) if len(slopes_y_to_x) > 0 else 0
        
        return y_to_x - x_to_y

    def estimate_direction(
        self, 
        var1: str, 
        var2: str, 
        method: str = 'entropy'
    ) -> int:
        """
        Estimate causal direction between two variables.

        Parameters
        ----------
        var1: str
            First variable name.
        var2: str
            Second variable name.
        method: str, default='entropy'
            Method to use for IGCI. Options are 'entropy' or 'slope'.

        Returns
        -------
        int
            1 if var1 -> var2, -1 if var2 -> var1, 0 if undetermined.

        Raises
        ------
        ValueError
            If method is not 'entropy' or 'slope'.
        """
        if method not in ['entropy', 'slope']:
            raise ValueError("Method must be either 'entropy' or 'slope'")
            
        x = self.data[var1].values
        y = self.data[var2].values
        
        valid_indices = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_indices]
        y = y[valid_indices]
        
        if len(x) < 10:
            return 0  
            
        if method == 'entropy':
            score = self._compute_entropy_based_score(x, y)
        else: 
            score = self._compute_slope_based_score(x, y)
            
        if score > 0:
            return 1  
        elif score < 0:
            return -1  
        else:
            return 0  
            
    def estimate(
        self, 
        variables: Optional[List[Tuple[str, str]]] = None, 
        method: str = 'entropy'
    ) -> Dict[Tuple[str, str], int]:
        """
        Estimate causal directions for multiple variable pairs.

        Parameters
        ----------
        variables: List of variable pairs, optional
            List of (var1, var2) pairs to analyze. If None, will analyze all
            possible pairs in the data.
        method: str, default='entropy'
            Method to use for IGCI. Options are 'entropy' or 'slope'.

        Returns
        -------
        Dict[Tuple[str, str], int]
            Dictionary mapping variable pairs to causal directions:
            1 for first->second, -1 for second->first, 0 for undetermined.
        """
        if variables is None:
            columns = list(self.data.columns)
            variables = [(columns[i], columns[j]) 
                        for i in range(len(columns)) 
                        for j in range(i+1, len(columns))]
        
        results = {}
        for var1, var2 in variables:
            direction = self.estimate_direction(var1, var2, method=method)
            results[(var1, var2)] = direction
            
        return results