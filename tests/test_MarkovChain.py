# tests/test_MarkovChain.py
import pytest
from pgmpy.models import MarkovChain

def test_add_transition_model_invalid_variable():
    mc = MarkovChain()
    mc.add_variables_from(['A', 'B'])
    
    import numpy as np
    tm = np.array([[0.5, 0.5], [0.2, 0.8]])
    
    # Should raise ValueError for variable not in mc.variables
    with pytest.raises(ValueError):
        mc.add_transition_model('C', tm)