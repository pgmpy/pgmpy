#!/usr/bin/env python

try:
    from lxml import etree
except ImportError:
    try:
        import xml.etree.cElementTree as etree
    except ImportError:
        try:
            import xml.etree.ElementTree as etree
        except ImportError:
            print("Failed to import ElementTree from any known place")
import numpy as np

__all__ = ['PomdpXReader']


class PomdpXReader(object):
    """
    Class for reading PomdpX file format from files or strings
    """

    def __init__(self, path=None, string=None):
        """
        Initialize an instance of PomdpX reader class

        Parameters
        ----------
        path : file or string
            File containing PomdpX information.
        string : string
            String containing PomdpX information.

        Example
        -------
        >>> reader = PomdpXReader('TestPomdpX.xml')
        """
        if path:
            self.network = etree.ElementTree(file=path).getroot()
        elif string:
            self.network = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")
        self.description = self.network.find('Description').text
        self.discount = self.network.find('Discount').text
        self.variables = None
        self.edge_list = None
        self.variable_states = None
        self.variable_parents = None
        self.variable_CPD = None
        self.variable_property = None

    def get_variables(self):
        """
        Returns list of variables of the network

        Example
        -------
        >>> reader = PomdpXReader("pomdpx.xml")
        >>> reader.get_variables()
        { 'StateVar': [ { 'vnamePrev' : 'rover_0'
                      'vnameCurr': 'rover_1'
                      'fullyObs': true
                      'NumValues': 3
                      },
                      { 'vnamePrev': 'rock_0'
                        'vnameCurr': 'rock_1'
                        'ValueEnum' : ['good', 'bad']
                      } ]
          'ObsVar': [ {'vname': 'obs_sensor'
                        'ValueEnum':['ogood', 'obad']
                      },]
          'ActionVar' : [ {'vname' : 'action_rover'
                           'ValueEnum' : 'amw ame ac as'} ]
          'RewardVar' : [ {'vname' : 'reward_rover'} ]
        }


        """
        self.variables = {}
        for variable in self.network.findall('Variable'):
            variables = {}
            variables['StateVar'] = []
            variables['ObsVar'] = []
            variables['ActionVar'] = []
            variables['RewardVar'] = []
            for var in variable.findall('StateVar'):
                state_variables = {}
                state_variables['vnamePrev'] = var.get('vnamePrev')
                state_variables['vnameCurr'] = var.get('vnameCurr')
                if var.get('fullyObs'):
                    state_variables['fullyObs'] = True
                else:
                    state_variables['fullyObs'] = False
                if var.find('NumValues') is not None:
                    state_variables['NumValues'] = var.find('NumValues').text
                if var.find('ValueEnum') is not None:
                    state_variables['ValueEnum'] = \
                        var.find('ValueEnum').text.split(' ')
                variables['StateVar'].append(state_variables)

            for var in variable.findall('ObsVar'):
                obs_variables = {}
                obs_variables['vname'] = var.get('vname')
                obs_variables['ValueEnum'] = \
                    var.find('ValueEnum').text.split()
                variables['ObsVar'].append(obs_variables)

            for var in variable.findall('ActionVar'):
                action_variables = {}
                action_variables['vname'] = var.get('vname')
                action_variables['ValueEnum'] = \
                    var.find('ValueEnum').text.split()
                variables['ActionVar'].append(action_variables)

            for var in variable.findall('RewardVar'):
                reward_variables = {}
                reward_variables['vname'] = var.get('vname')
                variables['RewardVar'].append(reward_variables)

            self.variables.update(variables)

        return self.variables

    def get_initial_beliefs(self):
        initial_state_belief = []
        for variable in self.network.findall('InitialStateBelief'):
            for var in variable.findall('CondProb'):
                cond_prob = {}
                cond_prob['Var'] = var.find('Var').text
                cond_prob['Parent'] = var.find('Parent').text.split()
                if var.find('Parameter').get('type') is None:
                    cond_prob['Type'] = 'TBL'
                else:
                    cond_prob['Type'] = var.find('Parameter').get('type')
                cond_prob['Parameter'] = self.get_parameter(var)
                initial_state_belief.append(cond_prob)

        return initial_state_belief

    def get_state_transition_function(self):
        state_transition_function = []
        for variable in self.network.findall('StateTransitionFunction'):
            for var in variable.findall('CondProb'):
                cond_prob = {}
                cond_prob['Var'] = var.find('Var').text
                cond_prob['Parent'] = var.find('Parent').text.split()
                if var.find('Parameter').get('type') is None:
                    cond_prob['Type'] = 'TBL'
                else:
                    cond_prob['Type'] = var.find('Parameter').get('type')
                cond_prob['Parameter'] = self.get_parameter(var)
                state_transition_function.append(cond_prob)

        return state_transition_function

    def get_obs_function(self):
        obs_function = []
        for variable in self.network.findall('ObsFunction'):
            for var in variable.findall('CondProb'):
                cond_prob = {}
                cond_prob['Var'] = var.find('Var').text
                cond_prob['Parent'] = var.find('Parent').text.split()
                if var.find('Parameter').get('type') is None:
                    cond_prob['Type'] = 'TBL'
                else:
                    cond_prob['Type'] = var.find('Parameter').get('type')
                cond_prob['Parameter'] = self.get_parameter(var)
                obs_function.append(cond_prob)

        return obs_function

    def get_reward_function(self):
        reward_function = []
        for variable in self.network.findall('RewardFunction'):
            for var in variable.findall('Func'):
                func = {}
                func['Var'] = var.find('Var').text
                func['Parent'] = var.find('Parent').text.split()
                if var.find('Parameter').get('type') is None:
                    func['Type'] = 'TBL'
                else:
                    func['Type'] = var.find('Parameter').get('type')
                func['Parameter'] = self.get_parameter(var)
                reward_function.append(func)

        return reward_function

    def get_parameter(self, var):
        parameter = []
        for parameter_tag in var.findall('Parameter'):
            if parameter_tag.get('type') is None or 'TBL':
                parameter = self.get_parameter_tbl(parameter_tag)
            else:
                pass
                parameter = self.get_parameter_dd(parameter_tag)
        return parameter

    def get_parameter_tbl(self, parameter):
        par = []
        for entry in parameter.findall('Entry'):
            instance = {}
            instance['Instance'] = entry.find('Instance').text.split()
            if entry.find('ProbTable') is None:
                instance['ValueTable'] = entry.find('ValueTable').text.split()
            else:
                instance['ProbTable'] = entry.find('ProbTable').text.split()
            par.append(instance)
        return par

    def get_parameter_dd(self, parameter):
        #TODO support for decision diagram class
        return parameter