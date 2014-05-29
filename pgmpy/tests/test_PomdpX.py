import unittest
from pgmpy.readwrite import PomdpXReader
import os


class TestPomdpXReaderString(unittest.TestCase):
    def setUp(self):
        file = str(os.getcwd()) + '/testpomdpx.xml'
        self.reader = PomdpXReader(path=file)

    def test_get_variables(self):
        var_expected = {'StateVar': [
                        {'vnamePrev': 'rover_0',
                        'vnameCurr': 'rover_1',
                        'NumValues': '3',
                        'fullyObs': True},
                        {'vnamePrev': 'rock_0',
                        'vnameCurr': 'rock_1',
                        'fullyObs': False,
                        'ValueEnum': ['good', 'bad']}],
                        'ObsVar': [{'vname': 'obs_sensor',
                                    'ValueEnum': ['ogood', 'obad']}],
                        'RewardVar': [{'vname': 'reward_rover'}],
                        'ActionVar': [{'vname': 'action_rover',
                                       'ValueEnum': ['amw', 'ame', 'ac', 'as']}]
                        }
        self.maxDiff = None
        self.assertEqual(self.reader.get_variables(), var_expected)

    def test_get_initial_belief_system(self):
        belief_expected = [{'Var': 'rover_0',
                            'Parent': ['null'],
                            'Type': 'TBL',
                            'Parameter': [{'Instance': ['-'],
                                           'ProbTable': ['0.0', '1.0', '0.0']}]
                            },
                           {'Var': 'rock_0',
                            'Parent': ['null'],
                            'Type': 'TBL',
                            'Parameter': [{'Instance': ['-'],
                                           'ProbTable': ['uniform']}]
                            }]
        self.maxDiff = None
        self.assertEqual(self.reader.get_initial_beliefs(), belief_expected)

    def test_get_state_transition_function(self):
        state_transition_function_expected = [{'Var': 'rover_1',
                                               'Parent': ['action_rover', 'rover_0'],
                                               'Type': 'TBL',
                                               'Parameter': [{'Instance': ['amw', 's0', 's2'],
                                                              'ProbTable': ['1.0']},
                                                             {'Instance': ['amw', 's1', 's0'],
                                                              'ProbTable': ['1.0']},
                                                             {'Instance': ['ame', 's0', 's1'],
                                                              'ProbTable': ['1.0']},
                                                             {'Instance': ['ame', 's1', 's2'],
                                                              'ProbTable': ['1.0']},
                                                             {'Instance': ['ac', 's0', 's0'],
                                                              'ProbTable': ['1.0']},
                                                             {'Instance': ['ac', 's1', 's1'],
                                                              'ProbTable': ['1.0']},
                                                             {'Instance': ['as', 's0', 's0'],
                                                              'ProbTable': ['1.0']},
                                                             {'Instance': ['as', 's1', 's2'],
                                                              'ProbTable': ['1.0']},
                                                             {'Instance': ['*', 's2', 's2'],
                                                              'ProbTable': ['1.0']}]},
                                              {'Var': 'rock_1',
                                               'Parent': ['action_rover', 'rover_0', 'rock_0'],
                                               'Type': 'TBL',
                                               'Parameter': [{'Instance': ['amw', '*', '-', '-'],
                                                              'ProbTable': ['1.0', '0.0', '0.0', '1.0']},
                                                             {'Instance': ['ame', '*', '-', '-'],
                                                              'ProbTable': ['identity']},
                                                             {'Instance': ['ac', '*', '-', '-'],
                                                              'ProbTable': ['identity']},
                                                             {'Instance': ['as', '*', '-', '-'],
                                                              'ProbTable': ['identity']},
                                                             {'Instance': ['as', 's0', '*', '-'],
                                                              'ProbTable': ['0.0', '1.0']},
                                                             ]}]
        self.maxDiff = None
        self.assertEqual(self.reader.get_state_transition_function(), state_transition_function_expected)

    def test_obs_function(self):
        obs_function_expected = [{'Var': 'obs_sensor',
                                  'Parent': ['action_rover', 'rover_1', 'rock_1'],
                                  'Type': 'TBL',
                                  'Parameter': [{'Instance': ['amw', '*', '*', '-'],
                                                'ProbTable': ['1.0', '0.0']},
                                                {'Instance': ['ame', '*', '*', '-'],
                                                'ProbTable': ['1.0', '0.0']},
                                                {'Instance': ['as', '*', '*', '-'],
                                                'ProbTable': ['1.0', '0.0']},
                                                {'Instance': ['ac', 's0', '-', '-'],
                                                'ProbTable': ['1.0', '0.0', '0.0', '1.0']},
                                                {'Instance': ['ac', 's1', '-', '-'],
                                                'ProbTable': ['0.8', '0.2', '0.2', '0.8']},
                                                {'Instance': ['ac', 's2', '*', '-'],
                                                'ProbTable': ['1.0', '0.0']}]},
                                ]
        self.maxDiff = None
        self.assertEqual(self.reader.get_obs_function(), obs_function_expected)

    def test_reward_function(self):
        reward_function_expected = [{'Var': 'reward_rover',
                                     'Parent': ['action_rover', 'rover_0', 'rock_0'],
                                     'Type': 'TBL',
                                     'Parameter': [{'Instance': ['ame', 's1','*'],
                                                    'ValueTable': ['10']},
                                                   {'Instance': ['amw', 's0','*'],
                                                    'ValueTable': ['-100']},
                                                   {'Instance': ['as', 's1','*'],
                                                    'ValueTable': ['-100']},
                                                   {'Instance': ['as', 's0','good'],
                                                    'ValueTable': ['10']},
                                                   {'Instance': ['as', 's0','bad'],
                                                    'ValueTable': ['-10']}]}]
        self.maxDiff = None
        self.assertEqual(self.reader.get_reward_function(), reward_function_expected)