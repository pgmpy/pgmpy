#!/usr/bin/env python3
from warnings import warn

import numpy as np
from pandas import DataFrame

from pgmpy.factors import State
from pgmpy.utils import sample_discrete


class MarkovChain:
    """
    Class to represent a Markov Chain with multiple kernels for factored state space,
    along with methods to simulate a run.

    Public Methods:
    ---------------
    set_start_state(state)
    add_variable(variable, cardinality)
    add_variables_from(vars_list, cards_list)
    add_transition_model(variable, transition_dict)
    sample(start_state, size)

    Examples:
    ---------
    Create an empty Markov Chain:
    >>> from pgmpy.models import MarkovChain as MC
    >>> model = MC()

    And then add variables to it
    >>> model.add_variables_from(['intel', 'diff'], [2, 3])

    Or directly create a Markov Chain from a list of variables and their cardinalities
    >>> model = MC(['intel', 'diff'], [2, 3])

    Add transition models
    >>> intel_tm = {0: {0: 0.25, 1: 0.75}, 1: {0: 0.5, 1: 0.5}}
    >>> model.add_transition_model('intel', intel_tm)
    >>> diff_tm = {0: {0: 0.1, 1: 0.5, 2: 0.4}, 1: {0: 0.2, 1: 0.2, 2: 0.6 }, 2: {0: 0.7, 1: 0.15, 2: 0.15}}
    >>> model.add_transition_model('diff', diff_tm)

    Set a start state
    >>> from pgmpy.factors import State
    >>> model.set_start_state([State('intel', 0), State('diff', 2)])

    Sample from it
    >>> model.sample(size=5)
       intel  diff
    0      0     2
    1      1     0
    2      0     1
    3      1     0
    4      0     2
    """
    def __init__(self, variables=None, card=None, start_state=None):
        """
        Parameters:
        -----------
        variables: array-like iterable object
            A list of variables of the model.

        card: array-like iterable object
            A list of cardinalities of the variables.

        start_state: array-like iterable object
            List of tuples representing the starting states of the variables.
        """
        if variables is None:
            variables = []
        if card is None:
            card = []
        if not hasattr(variables, '__iter__') or isinstance(variables, str):
            raise ValueError('variables must be a non-string iterable.')
        if not hasattr(card, '__iter__') or isinstance(card, str):
            raise ValueError('card must be a non-string iterable.')
        self.variables = variables
        self.cardinalities = {v: c for v, c in zip(variables, card)}
        self.transition_models = {var: {} for var in variables}
        if start_state is None or self._check_state(start_state):
            self.state = start_state

    def set_start_state(self, start_state):
        """
        Set the start state of the Markov Chain. If the start_state is given as a array-like iterable, its contents
        are reordered in the internal representation.

        Parameters:
        -----------
        start_state: dict or array-like iterable object
            Dict (or list) of tuples representing the starting states of the variables.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChain as MC
        >>> from pgmpy.factors import State
        >>> model = MC(['a', 'b'], [2, 2])
        >>> model.set_start_state([State('a', 0), State('b', 1)])
        """
        if start_state is not None:
            if not hasattr(start_state, '__iter__') or isinstance(start_state, str):
                raise ValueError('start_state must be a non-string iterable.')
            # Must be an array-like iterable. Reorder according to self.variables.
            state_dict = {var: st for var, st in start_state}
            start_state = [State(var, state_dict[var]) for var in self.variables]
        if start_state is None or self._check_state(start_state):
            self.state = start_state

    def _check_state(self, state):
        """
        Checks if a list representing the state of the variables is valid.
        """
        if not hasattr(state, '__iter__') or isinstance(state, str):
            raise ValueError('Start state must be a non-string iterable object.')
        state_vars = {s.var for s in state}
        model_vars = set(self.transition_models.keys())
        if not state_vars == model_vars:
            raise ValueError('Start state must represent a complete assignment to all variables.'
                             'Expected variables in state: {svar}, Got: {mvar}.'.format(svar=state_vars,
                                                                                        mvar=model_vars))
        for var, val in state:
            if val >= self.cardinalities[var]:
                raise ValueError('Assignment {val} to {var} invalid.'.format(val=val, var=var))
        return True

    def add_variable(self, variable, card=0):
        """
        Add a variable to the model.

        Parameters:
        -----------
        variable: any hashable python object

        card: int
            Representing the cardinality of the variable to be added.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChain as MC
        >>> model = MC()
        >>> model.add_variable('x', 4)
        """
        if variable not in self.variables:
            self.variables.append(variable)
        else:
            warn('Variable {var} already exists.'.format(var=variable))
        self.cardinalities[variable] = card
        self.transition_models[variable] = {}

    def add_variables_from(self, variables, cards):
        """
        Add several variables to the model at once.

        Parameters:
        -----------
        variables: array-like iterable object
            List of variables to be added.

        cards: array-like iterable object
            List of cardinalities of the variables to be added.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChain as MC
        >>> model = MC()
        >>> model.add_variables_from(['x', 'y'], [3, 4])
        """
        for var, card in zip(variables, cards):
            self.add_variable(var, card)

    def add_transition_model(self, variable, transition_model):
        """
        Adds a transition model for a particular variable.

        Parameters:
        -----------
        variable: any hashable python object
            must be an existing variable of the model.
        transition_model: dict
            representing valid transition probabilities defined for every possible state of the variable.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChain as MC
        >>> model = MC()
        >>> model.add_variable('grade', 3)
        >>> grade_tm = {0: {0: 0.1, 1: 0.5, 2: 0.4}, 1: {0: 0.2, 1: 0.2, 2: 0.6 }, 2: {0: 0.7, 1: 0.15, 2: 0.15}}
        >>> model.add_transition_model('grade', grade_tm)
        """
        # check if the transition model is valid
        if not isinstance(transition_model, dict):
            raise ValueError('Transition model must be a dict.')

        exp_states = set(range(self.cardinalities[variable]))
        tm_states = set(transition_model.keys())
        if not exp_states == tm_states:
            raise ValueError('Transitions must be defined for all states of variable {v}. '
                             'Expected states: {es}, Got: {ts}.'.format(v=variable, es=exp_states, ts=tm_states))

        for _, transition in transition_model.items():
            if not isinstance(transition, dict):
                raise ValueError('Each transition must be a dict.')
            prob_sum = 0

            for _, prob in transition.items():
                if prob < 0 or prob > 1:
                    raise ValueError('Transitions must represent valid probability weights.')
                prob_sum += prob

            if not np.allclose(prob_sum, 1):
                raise ValueError('Transition probabilities must sum to 1.')

        self.transition_models[variable] = transition_model

    def sample(self, start_state=None, size=1):
        """
        Sample from the Markov Chain.

        Parameters:
        -----------
        start_state: dict or array-like iterable
            Representing the starting states of the variables. If None is passed, a random start_state is chosen.
        size: int
            Number of samples to be generated.

        Return Type:
        ------------
        pandas.DataFrame

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChain as MC
        >>> from pgmpy.factors import State
        >>> model = MC(['intel', 'diff'], [2, 3])
        >>> model.set_start_state([State('intel', 0), State('diff', 2)])
        >>> intel_tm = {0: {0: 0.25, 1: 0.75}, 1: {0: 0.5, 1: 0.5}}
        >>> model.add_transition_model('intel', intel_tm)
        >>> diff_tm = {0: {0: 0.1, 1: 0.5, 2: 0.4}, 1: {0: 0.2, 1: 0.2, 2: 0.6 }, 2: {0: 0.7, 1: 0.15, 2: 0.15}}
        >>> model.add_transition_model('diff', diff_tm)
        >>> model.sample(size=5)
           intel  diff
        0      0     2
        1      1     0
        2      0     1
        3      1     0
        4      0     2
        """
        if start_state is None:
            if self.state is None:
                self.state = self.random_state()
            # else use previously-set state
        else:
            self.set_start_state(start_state)

        sampled = DataFrame(index=range(size), columns=self.variables)
        sampled.loc[0] = [st for var, st in self.state]

        for i in range(size - 1):
            for j, (var, st) in enumerate(self.state):
                next_st = sample_discrete(list(self.transition_models[var][st].keys()),
                                          list(self.transition_models[var][st].values()))[0]
                self.state[j] = State(var, next_st)
            sampled.loc[i + 1] = [st for var, st in self.state]

        return sampled

    def prob_from_sample(self, state, sample=None, window_size=None):
        """
        Given an instantiation (partial or complete) of the variables of the model,
        compute the probability of observing it over multiple windows in a given sample.

        If 'sample' is not passed as an argument, generate the statistic by sampling from the
        Markov Chain, starting with a random initial state.

        Examples:
        ---------
        >>> from pgmpy.models.MarkovChain import MarkovChain as MC
        >>> model = MC(['intel', 'diff'], [3, 2])
        >>> intel_tm = {0: {0: 0.2, 1: 0.4, 2:0.4}, 1: {0: 0, 1: 0.5, 2: 0.5}, 2: {2: 1}}
        >>> model.add_transition_model('intel', intel_tm)
        >>> diff_tm = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.25, 1:0.75}}
        >>> model.add_transition_model('diff', diff_tm)
        >>> model.prob_from_sample([State('diff', 0)])
        array([ 0.27,  0.4 ,  0.18,  0.23, ..., 0.29])
        """
        if sample is None:
            # generate sample of size 10000
            sample = self.sample(self.random_state(), size=10000)
        if window_size is None:
            window_size = len(sample) // 100  # default window size is 100
        windows = len(sample) // window_size
        probabilities = np.zeros(windows)

        for i in range(windows):
            for j in range(window_size):
                ind = i * window_size + j
                state_eq = [sample.loc[ind, v] == s for v, s in state]
                if all(state_eq):
                    probabilities[i] += 1

        return probabilities / window_size

    def generate_sample(self, start_state=None, size=1):
        """
        Generator version of self.sample

        Return Type:
        ------------
        List of State namedtuples, representing the assignment to all variables of the model.

        Examples:
        ---------
        >>> from pgmpy.models.MarkovChain import MarkovChain
        >>> from pgmpy.factors import State
        >>> model = MarkovChain()
        >>> model.add_variables_from(['intel', 'diff'], [3, 2])
        >>> intel_tm = {0: {0: 0.2, 1: 0.4, 2:0.4}, 1: {0: 0, 1: 0.5, 2: 0.5}, 2: {0: 0.3, 1: 0.3, 2: 0.4}}
        >>> model.add_transition_model('intel', intel_tm)
        >>> diff_tm = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.25, 1:0.75}}
        >>> model.add_transition_model('diff', diff_tm)
        >>> gen = model.generate_sample([State('intel', 0), State('diff', 0)], 2)
        >>> [sample for sample in gen]
        [[State(var='intel', state=2), State(var='diff', state=1)],
         [State(var='intel', state=2), State(var='diff', state=0)]]
        """
        if start_state is None:
            if self.state is None:
                self.state = self.random_state()
            # else use previously-set state
        else:
            self.set_start_state(start_state)
        # sampled.loc[0] = [self.state[var] for var in self.variables]

        for i in range(size):
            for j, (var, st) in enumerate(self.state):
                next_st = sample_discrete(list(self.transition_models[var][st].keys()),
                                          list(self.transition_models[var][st].values()))[0]
                self.state[j] = State(var, next_st)
            yield self.state[:]

    def random_state(self):
        """
        Generates a random state of the Markov Chain.

        Return Type:
        ------------
        List of namedtuples, representing a random assignment to all variables of the model.

        Examples:
        ---------
        >>> from pgmpy.models import MarkovChain as MC
        >>> model = MC(['intel', 'diff'], [2, 3])
        >>> model.random_state()
        [State('diff', 2), State('intel', 1)]
        """
        return [State(var, np.random.randint(self.cardinalities[var])) for var in self.variables]
