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
    """
    pass
