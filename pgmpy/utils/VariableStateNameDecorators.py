"""
The file contains decorators to manage the user defined
variable state names. It maps the internal representaion
of the varibale states to the user defined state names and
vice versa.
"""

class stateNameInit():
    """
    The class behaves as a decorator for __init__ methods.
    It adds a dictionary as an attribute to the various classes
    where mapping is required for the variable state names.
    The dictionary has the following format - 
    {'x1': ['on', 'off']}
    where, 'x1' is a variable and 'on', 'off' are its correspinding
    states.

    Example
    -------
    >>> import numpy as np
    >>> from pgmpy.factors import Factor
    >>> sn = {'speed': ['low', 'medium', 'high'], 'switch': ['on', 'off'], 'time': ['day', 'night']}
    >>> phi = Factor(['speed', 'switch', 'time'], [3, 2, 2], np.ones(12), state_names=sn)
    >>> print(phi.state_names)
    {'speed': ['low', 'medium', 'high'], 'switch': ['on', 'off'], 'time': ['day', 'night']}
    """
    def __call__(self, f):
        def wrapper(*args, **kwargs): 
            # Case, when no state names dict is provided.
            if not 'state_names' in kwargs:
                # args[0] represents the self parameter of the __init__ method
                args[0].state_names = None
            else:
                args[0].state_names = kwargs['state_names']
                del kwargs['state_names']
            f(*args, **kwargs)
        return wrapper
