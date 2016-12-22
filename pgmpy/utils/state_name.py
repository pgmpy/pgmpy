from wrapt import decorator
"""
The file contains decorators to manage the user defined
variable state names. It maps the internal representaion
of the varibale states to the user defined state names and
vice versa.
"""


class StateNameInit():
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
    >>> from pgmpy.factors.discrete import DiscreteFactor
    >>> sn = {'speed': ['low', 'medium', 'high'],
    ...       'switch': ['on', 'off'],
    ...       'time': ['day', 'night']}
    >>> phi = DiscreteFactor(['speed', 'switch', 'time'],
    ...                      [3, 2, 2], np.ones(12), state_names=sn)
    >>> print(phi.state_names)
    {'speed': ['low', 'medium', 'high'], 'switch': ['on', 'off'], 'time': ['day', 'night']}
    """
    @decorator
    def __call__(self, f, instance, args, kwargs):
        # Case, when no state names dict is provided.
        if 'state_names' not in kwargs:
            # instance represents the self parameter of the __init__ method
            instance.state_names = None
        else:
            instance.state_names = kwargs['state_names']
            del kwargs['state_names']
        f(*args, **kwargs)


class StateNameDecorator():
    """
    This class behaves as a decorator for the various methods
    that can use variable state names either in input parameters
    or in return values.

    Parameters
    ----------
    argument: string or None
        The parameter that needs to be wrapped. None,
        if no parameter is to be wrapped.
    return_val: boolean
        True if the return value needs to be wrapped else
        False.
    """
    def __init__(self, argument, return_val):
        self.arg = argument
        self.return_val = return_val
        self.state_names = None
        self.arg_formats = [self.is_list_of_states,
                            self.is_list_of_list_of_states,
                            self.is_dict_of_states]

    def is_list_of_states(self, arg):
        """
        A list of states example -
        [('x1', 'easy'), ('x2', 'hard')]

        Returns
        -------
        True, if arg is a list of states else False.

        """
        return isinstance(arg, list) and all(isinstance(i, tuple) for i in arg)

    def is_list_of_list_of_states(self, arg):
        """
        A list of list of states example -
        [[('x1', 'easy'), ('x2', 'hard')], [('x1', 'hard'), ('x2', 'medium')]]

        Returns
        -------
        True, if arg is a list of list of states else False.

        """
        if arg is None:
            return False
        return all([isinstance(arg, list),
                   all(isinstance(i, list) for i in arg),
                   all((isinstance(i, tuple) for i in lst) for lst in arg)])

    def is_dict_of_states(self, arg):
        """
        A dict of states example -
        {'x1': 'easy', 'x2':'hard', 'x3': 'medium'}

        Returns
        -------
        True, if arg is dict of states else False.

        """
        return isinstance(arg, dict)

    def get_mapped_value(self, arg_val):
        for arg_format in self.arg_formats:
            if arg_format(arg_val):
                return self.map_states(arg_val, arg_format)
        return None

    def map_states(self, arg_val, arg_format):
        if arg_format == self.is_list_of_states:
            if not isinstance(arg_val[0][1], str):
                # If the input parameter is consistent with the internal
                # state names architecture
                if not self.return_val:
                    return arg_val
                else:
                    return [(var, self.state_names[var][state])
                            for var, state in arg_val]
            else:
                return [(var, self.state_names[var].index(state))
                        for var, state in arg_val]

        if arg_format == self.is_list_of_list_of_states:
            if not isinstance(arg_val[0][0][1], str):
                # If the input parameter is consistent with the internal
                # state names architecture
                if not self.return_val:
                    return arg_val
                else:
                    mapped_arg_val = []
                    for elem in arg_val:
                        mapped_elem = [(var, self.state_names[var][state])
                                       for var, state in elem]
                        mapped_arg_val.append(mapped_elem)
            else:
                mapped_arg_val = []
                for elem in arg_val:
                    mapped_elem = [(var, self.state_names[var].index(state))
                                   for var, state in elem]
                    mapped_arg_val.append(mapped_elem)
            return mapped_arg_val

        if arg_format == self.is_dict_of_states:
            if not any([isinstance(i, str) for i in arg_val.values()]):
                # If the input parameter is consistent with the internal
                # state names architecture
                if not self.return_val:
                    return arg_val
                else:
                    for var in arg_val:
                        arg_val[var] = self.state_names[var][arg_val[var]]
            else:
                for var in arg_val:
                    arg_val[var] = self.state_names[var].index(arg_val[var])
            return arg_val

    @decorator
    def __call__(self, f, instance, args, kwargs):
        # instance represents the self parameter of the __init__ method
        method_self = instance

        if not method_self.state_names:
            return f(*args, **kwargs)
        else:
            self.state_names = method_self.state_names

        if self.arg and not self.return_val:
            # If input parameters are in kwargs format
            if self.arg in kwargs:
                arg_val = kwargs[self.arg]
                kwargs[self.arg] = self.get_mapped_value(arg_val)
                return f(*args, **kwargs)
            # If input parameters are in args format
            else:
                for arg_val in args[:]:
                    mapped_arg_val = self.get_mapped_value(arg_val)
                    if mapped_arg_val:
                        mapped_args = list(args)
                        mapped_args[args.index(arg_val)] = mapped_arg_val
                return f(*mapped_args, **kwargs)

        elif not self.arg and self.return_val:
            return_val = f(*args, **kwargs)

            mapped_return_val = self.get_mapped_value(return_val)
            # If the function returns only one output
            if mapped_return_val:
                return mapped_return_val

            # If the function returns more than one output.
            for ret_val in list(return_val):
                mapped_ret_val = self.get_mapped_value(ret_val)
                if ret_val:
                    return_val[return_val.index(ret_val)] = mapped_ret_val
            return return_val
