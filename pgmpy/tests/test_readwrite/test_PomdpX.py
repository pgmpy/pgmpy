#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import warnings

from pgmpy.readwrite import PomdpXReader, PomdpXWriter
from pgmpy.extern import six

try:
    from lxml import etree
except ImportError:
    try:
        # Python 2.5
        import xml.etree.cElementTree as etree
        six.print_("running with cElementTree on Python 2.5+")
    except ImportError:
        try:
            import xml.etree.ElementTree as etree
            print("running with ElementTree on Python 2.5+")
        except ImportError:
            warnings.warn("Failed to import ElementTree from any known place")


class TestPomdpXReaderString(unittest.TestCase):
    def setUp(self):
        string = """<pomdpx version="1.0" id="rockSample"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="pomdpx.xsd">
         <Description>RockSample problem for map size 1 x 3.
           Rock is at 0, Rover’s initial position is at 1.
           Exit is at 2.
         </Description>
         <Discount>0.95</Discount>
         <Variable>
              <StateVar vnamePrev="rover_0" vnameCurr="rover_1"
                fullyObs="true">
                  <NumValues>3</NumValues>
              </StateVar>
              <StateVar vnamePrev="rock_0" vnameCurr="rock_1">
                  <ValueEnum>good bad</ValueEnum>
              </StateVar>
              <ObsVar vname="obs_sensor">
                  <ValueEnum>ogood obad</ValueEnum>
              </ObsVar>
              <ActionVar vname="action_rover">
                  <ValueEnum>amw ame ac as</ValueEnum>
              </ActionVar>
              <RewardVar vname="reward_rover" />
         </Variable>
         <InitialStateBelief>
              <CondProb>
                  <Var>rover_0</Var>
                  <Parent>null</Parent>
                  <Parameter type="TBL">
                        <Entry>
                            <Instance> - </Instance>
                            <ProbTable>0.0 1.0 0.0</ProbTable>
                        </Entry>
              </Parameter>
         </CondProb>
         <CondProb>
              <Var>rock_0</Var>
              <Parent>null</Parent>
              <Parameter type="TBL">
                  <Entry>
                      <Instance>-</Instance>
                      <ProbTable>uniform</ProbTable>
                  </Entry>
              </Parameter>
         </CondProb>
      </InitialStateBelief>
      <StateTransitionFunction>
          <CondProb>
              <Var>rover_1</Var>
              <Parent>action_rover rover_0</Parent>
              <Parameter type="TBL">
                  <Entry>
                      <Instance>amw s0 s2</Instance>
                      <ProbTable>1.0</ProbTable>
                  </Entry>
                  <Entry>
                      <Instance>amw s1 s0</Instance>
                      <ProbTable>1.0</ProbTable>
                  </Entry>
                  <Entry>
                      <Instance>ame s0 s1</Instance>
                      <ProbTable>1.0</ProbTable>
                  </Entry>
                  <Entry>
                      <Instance>ame s1 s2</Instance>
                      <ProbTable>1.0</ProbTable>
                  </Entry>
                  <Entry>
                      <Instance>ac s0 s0</Instance>
                      <ProbTable>1.0</ProbTable>
                  </Entry>
                  <Entry>
                      <Instance>ac s1 s1</Instance>
                      <ProbTable>1.0</ProbTable>
                  </Entry>
                  <Entry>
                      <Instance>as s0 s0</Instance>
                      <ProbTable>1.0</ProbTable>
                  </Entry>
                  <Entry>
                      <Instance>as s1 s2</Instance>
                      <ProbTable>1.0</ProbTable>
                  </Entry>
                  <Entry>
                      <Instance>* s2 s2</Instance>
                      <ProbTable>1.0</ProbTable>
                  </Entry>
           </Parameter>
       </CondProb>
       <CondProb>
           <Var>rock_1</Var>
           <Parent>action_rover rover_0 rock_0</Parent>
           <Parameter>
               <Entry>
                   <Instance>amw * - - </Instance>
                   <ProbTable>1.0 0.0 0.0 1.0</ProbTable>
               </Entry>
               <Entry>
                   <Instance>ame * - - </Instance>
                   <ProbTable>identity</ProbTable>
               </Entry>
               <Entry>
                   <Instance>ac * - - </Instance>
                   <ProbTable>identity</ProbTable>
               </Entry>
               <Entry>
                   <Instance>as * - - </Instance>
                   <ProbTable>identity</ProbTable>
               </Entry>
               <Entry>
                   <Instance>as s0 * - </Instance>
                   <ProbTable>0.0 1.0</ProbTable>
               </Entry>
           </Parameter>
       </CondProb>
   </StateTransitionFunction>
   <ObsFunction>
       <CondProb>
           <Var>obs_sensor</Var>
           <Parent>action_rover rover_1 rock_1</Parent>
           <Parameter type="TBL">
               <Entry>
                   <Instance>amw * * - </Instance>
                   <ProbTable>1.0 0.0</ProbTable>
               </Entry>
               <Entry>
                   <Instance>ame * * - </Instance>
                   <ProbTable>1.0 0.0</ProbTable>
               </Entry>
               <Entry>
                   <Instance>as * * - </Instance>
                   <ProbTable>1.0 0.0</ProbTable>
               </Entry>
               <Entry>
                   <Instance>ac s0 - - </Instance>
                   <ProbTable>1.0 0.0 0.0 1.0</ProbTable>
               </Entry>
               <Entry>
                   <Instance>ac s1 - - </Instance>
                   <ProbTable>0.8 0.2 0.2 0.8</ProbTable>
               </Entry>
                <Entry>
                     <Instance>ac s2 * - </Instance>
                     <ProbTable>1.0 0.0</ProbTable>
                </Entry>
            </Parameter>
        </CondProb>
    </ObsFunction>
    <RewardFunction>
        <Func>
            <Var>reward_rover</Var>
            <Parent>action_rover rover_0 rock_0</Parent>
            <Parameter type="TBL">
                <Entry>
                     <Instance>ame s1 *</Instance>
                     <ValueTable>10</ValueTable>
                </Entry>
                <Entry>
                     <Instance>amw s0 *</Instance>
                     <ValueTable>-100</ValueTable>
                </Entry>
                <Entry>
                     <Instance>as s1 *</Instance>
                     <ValueTable>-100</ValueTable>
                </Entry>
                <Entry>
                     <Instance>as s0 good</Instance>
                     <ValueTable>10</ValueTable>
                </Entry>
                <Entry>
                     <Instance>as s0 bad</Instance>
                     <ValueTable>-10</ValueTable>
                </Entry>
            </Parameter>
        </Func>
    </RewardFunction>
 </pomdpx>
 """
        self.reader_string = PomdpXReader(string=string)
        self.reader_file = PomdpXReader(path=six.StringIO(string))

    def test_get_variables(self):
        var_expected = {'StateVar': [
                        {'vnamePrev': 'rover_0',
                         'vnameCurr': 'rover_1',
                         'ValueEnum': ['s0', 's1', 's2'],
                         'fullyObs': True},
                        {'vnamePrev': 'rock_0',
                         'vnameCurr': 'rock_1',
                         'fullyObs': False,
                         'ValueEnum': ['good', 'bad']}],
                        'ObsVar': [{'vname': 'obs_sensor',
                                    'ValueEnum': ['ogood', 'obad']}],
                        'RewardVar': [{'vname': 'reward_rover'}],
                        'ActionVar': [{'vname': 'action_rover',
                                       'ValueEnum': ['amw', 'ame',
                                                     'ac', 'as']}]
                        }
        self.maxDiff = None
        self.assertEqual(self.reader_string.get_variables(), var_expected)
        self.assertEqual(self.reader_file.get_variables(), var_expected)

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
        self.assertEqual(self.reader_string.get_initial_beliefs(), belief_expected)
        self.assertEqual(self.reader_file.get_initial_beliefs(), belief_expected)

    def test_get_state_transition_function(self):
        state_transition_function_expected = \
            [{'Var': 'rover_1',
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
        self.assertEqual(self.reader_string.get_state_transition_function(),
                         state_transition_function_expected)
        self.assertEqual(self.reader_file.get_state_transition_function(),
                         state_transition_function_expected)

    def test_obs_function(self):
        obs_function_expected = \
            [{'Var': 'obs_sensor',
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
                             'ProbTable': ['1.0', '0.0']}]}]
        self.maxDiff = None
        self.assertEqual(self.reader_string.get_obs_function(), obs_function_expected)
        self.assertEqual(self.reader_file.get_obs_function(), obs_function_expected)

    def test_reward_function(self):
        reward_function_expected = \
            [{'Var': 'reward_rover',
              'Parent': ['action_rover', 'rover_0', 'rock_0'],
              'Type': 'TBL',
              'Parameter': [{'Instance': ['ame', 's1', '*'],
                             'ValueTable': ['10']},
                            {'Instance': ['amw', 's0', '*'],
                             'ValueTable': ['-100']},
                            {'Instance': ['as', 's1', '*'],
                             'ValueTable': ['-100']},
                            {'Instance': ['as', 's0', 'good'],
                             'ValueTable': ['10']},
                            {'Instance': ['as', 's0', 'bad'],
                             'ValueTable': ['-10']}]}]
        self.maxDiff = None
        self.assertEqual(self.reader_string.get_reward_function(),
                         reward_function_expected)
        self.assertEqual(self.reader_file.get_reward_function(),
                         reward_function_expected)

    def test_get_parameter_dd(self):
        string = """
        <pomdpx version="1.0" id="rockSample"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="pomdpx.xsd">
         <Description>RockSample problem for map size 1 x 3.
           Rock is at 0, Rover’s initial position is at 1.
           Exit is at 2.
         </Description>
         <Discount>0.95</Discount>
        <InitialStateBelief>
        <CondProb>
         <Var>rover_0</Var>
  <Parent>null</Parent>
   <Parameter type = "DD">
      <DAG>
         <Node var = "rover_0">
             <Edge val="s0"><Terminal>0.0</Terminal></Edge>
              <Edge val="s1">
                 <Node var = "rock_0">
                     <Edge val = "good">
                       <Terminal>0.5</Terminal>
                  </Edge>
                   <Edge val = "bad">
                      <Terminal>0.5</Terminal>
                   </Edge>
                </Node>
           </Edge>
           <Edge val="s2"><Terminal>0.0</Terminal></Edge>
         </Node>
      </DAG>
  </Parameter>
  </CondProb>
  </InitialStateBelief>
  </pomdpx>
 """
        self.reader_string = PomdpXReader(string=string)
        self.reader_file = PomdpXReader(path=six.StringIO(string))
        expected_dd_parameter = [{
            'Var': 'rover_0',
            'Parent': ['null'],
            'Type': 'DD',
            'Parameter': {'rover_0': {'s0': '0.0',
                                      's1': {'rock_0': {'good': '0.5',
                                                        'bad': '0.5'}},
                                      's2': '0.0'}}}]
        self.maxDiff = None
        self.assertEqual(expected_dd_parameter,
                         self.reader_string.get_initial_beliefs())
        self.assertEqual(expected_dd_parameter,
                         self.reader_file.get_initial_beliefs())

    def test_initial_belief_dd(self):
        string = """
    <pomdpx version="1.0" id="rockSample"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="pomdpx.xsd">
         <Description>RockSample problem for map size 1 x 3.
           Rock is at 0, Rover’s initial position is at 1.
           Exit is at 2.
         </Description>
         <Discount>0.95</Discount>
        <InitialStateBelief>
        <CondProb>
            <Var>rover_0</Var>
            <Parent>null</Parent>
            <Parameter type="DD">
                <DAG>
                    <Node var="rover_0">
                        <Edge val="s0">
                            <Terminal>0.0</Terminal>
                        </Edge>
                        <Edge val="s1">
                            <SubDAG type="uniform" var="rock_0"/>
                        </Edge>
                        <Edge val="s2">
                            <Terminal>0.0</Terminal>
                        </Edge>
                    </Node>
                </DAG>
            </Parameter>
        </CondProb>
    </InitialStateBelief>
    </pomdpx>
    """
        self.reader_string = PomdpXReader(string=string)
        self.reader_file = PomdpXReader(path=six.StringIO(string))
        expected_belief_dd = [{
            'Var': 'rover_0',
            'Parent': ['null'],
            'Type': 'DD',
            'Parameter': {'rover_0': {'s0': '0.0',
                                      's1': {'type': 'uniform',
                                             'var': 'rock_0'},
                                      's2': '0.0'}}}]
        self.maxDiff = None
        self.assertEqual(self.reader_string.get_initial_beliefs(),
                         expected_belief_dd)
        self.assertEqual(self.reader_file.get_initial_beliefs(),
                         expected_belief_dd)

    def test_reward_function(self):
        string = """
        <pomdpx version="1.0" id="rockSample"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="pomdpx.xsd">
         <Description>RockSample problem for map size 1 x 3.
           Rock is at 0, Rover’s initial position is at 1.
           Exit is at 2.
         </Description>
         <Discount>0.95</Discount>
        <RewardFunction>
        <Func>
            <Var>reward_rover</Var>
            <Parent>action_rover rover_0 rock_0</Parent>
            <Parameter type="DD">
                <DAG>
                    <Node var="action_rover">
                        <Edge val="amw">
                            <Node var="rover_0">
                                <Edge val="s0">
                                    <Terminal>-100.0</Terminal>
                                </Edge>
                                <Edge val="s1">
                                    <Terminal>0.0</Terminal>
                                </Edge>
                                <Edge val="s2">
                                    <Terminal>0.0</Terminal>
                                </Edge>
                            </Node>
                        </Edge>
                        <Edge val="ame">
                            <Node var="rover_0">
                                <Edge val="s0">
                                    <Terminal>0.0</Terminal>
                                </Edge>
                                <Edge val="s1">
                                    <Terminal>10.0</Terminal>
                                </Edge>
                                <Edge val="s2">
                                    <Terminal>0.0</Terminal>
                                </Edge>
                            </Node>
                        </Edge>
                        <Edge val="ac">
                            <Terminal>0.0</Terminal>
                        </Edge>
                        <Edge val="as">
                            <Node var="rover_0">
                                <Edge val="s0">
                                    <Node var="rock_0">
                                        <Edge val="good">
                                            <Terminal>10</Terminal>
                                        </Edge>
                                        <Edge val="bad">
                                            <Terminal>-10</Terminal>
                                        </Edge>
                                    </Node>
                                </Edge>
                                <Edge val="s1">
                                    <Terminal>-100</Terminal>
                                </Edge>
                                <Edge val="s2">
                                    <Terminal>-100</Terminal>
                                </Edge>
                            </Node>
                        </Edge>
                    </Node>
                </DAG>
            </Parameter>
        </Func>
    </RewardFunction>
    </pomdpx>
        """
        self.reader_string = PomdpXReader(string=string)
        self.reader_file = PomdpXReader(path=six.StringIO(string))
        expected_reward_function_dd =\
            [{'Var': 'reward_rover',
              'Parent': ['action_rover', 'rover_0', 'rock_0'],
              'Type': 'DD',
              'Parameter': {'action_rover': {'amw': {'rover_0': {'s0': '-100.0',
                                                                 's1': '0.0',
                                                                 's2': '0.0'}},
                                             'ame': {'rover_0': {'s0': '0.0',
                                                                 's1': '10.0',
                                                                 's2': '0.0'}},
                                             'ac': '0.0',
                                             'as': {'rover_0': {'s0': {'rock_0': {'good': '10',
                                                                                  'bad': '-10'}},
                                                                's1': '-100',
                                                                's2': '-100'}}}}}]
        self.maxDiff = None
        self.assertEqual(self.reader_string.get_reward_function(),
                         expected_reward_function_dd)
        self.assertEqual(self.reader_file.get_reward_function(),
                         expected_reward_function_dd)

    def test_state_transition_function(self):
        string = """
         <pomdpx version="1.0" id="rockSample"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="pomdpx.xsd">
         <Description>RockSample problem for map size 1 x 3.
           Rock is at 0, Rover’s initial position is at 1.
           Exit is at 2.
         </Description>
         <Discount>0.95</Discount>
        <StateTransitionFunction>
        <CondProb>
            <Var>rover_1</Var>
            <Parent>action_rover rover_0</Parent>
            <Parameter type="DD">
                <DAG>
                    <Node var="action_rover">
                        <Edge val="amw">
                            <Node var="rover_0">
                                <Edge val="s0">
                                    <SubDAG type="deterministic" var="rover_1" val="s2"/>
                                </Edge>
                                <Edge val="s1">
                                    <SubDAG type="deterministic" var="rover_1" val="s0"/>
                                </Edge>
                                <Edge val="s2">
                                    <SubDAG type="deterministic" var="rover_1" val="s2"/>
                                </Edge>
                            </Node>
                        </Edge>
                        <Edge val="ame">
                            <Node var="rover_0">
                                <Edge val="s0">
                                    <SubDAG type="deterministic" var="rover_1" val="s1"/>
                                </Edge>
                                <Edge val="s1">
                                    <SubDAG type="deterministic" var="rover_1" val="s2"/>
                                </Edge>
                                <Edge val="s2">
                                    <SubDAG type="deterministic" var="rover_1" val="s2"/>
                                </Edge>
                            </Node>
                        </Edge>
                        <Edge val="ac">
                            <SubDAG type="persistent" var="rover_1"/>
                        </Edge>
                        <Edge val="as">
                            <Node var="rover_0">
                                <Edge val="s0">
                                    <SubDAG type="deterministic" var="rover_1" val="s0"/>
                                </Edge>
                                <Edge val="s1">
                                    <SubDAG type="deterministic" var="rover_1" val="s2"/>
                                </Edge>
                                <Edge val="s2">
                                    <SubDAG type="deterministic" var="rover_1" val="s2"/>
                                </Edge>
                            </Node>
                        </Edge>
                    </Node>
                </DAG>
            </Parameter>
        </CondProb>
        <CondProb>
            <Var>rock_1</Var>
            <Parent>action_rover rover_0 rock_0</Parent>
            <Parameter type="DD">
                <DAG>
                    <Node var="action_rover">
                        <Edge val="amw">
                            <SubDAG type="persistent" var="rock_1"/>
                        </Edge>
                        <Edge val="ame">
                            <SubDAG type="persistent" var="rock_1"/>
                        </Edge>
                        <Edge val="ac">
                            <SubDAG type="persistent" var="rock_1"/>
                        </Edge>
                        <Edge val="as">
                            <Node var="rover_0">
                                <Edge val="s0">
                                    <SubDAG type="deterministic" var="rock_1" val="bad"/>
                                </Edge>
                                <Edge val="s1">
                                    <SubDAG type="persistent" var="rock_1"/>
                                </Edge>
                                <Edge val="s2">
                                    <SubDAG type="persistent" var="rock_1"/>
                                </Edge>
                            </Node>
                        </Edge>
                    </Node>
                </DAG>
            </Parameter>
        </CondProb>
    </StateTransitionFunction>
</pomdpx>
        """
        self.reader_string = PomdpXReader(string=string)
        self.reader_file = PomdpXReader(path=six.StringIO(string))
        expected_state_transition_function = \
            [{'Var': 'rover_1',
              'Parent': ['action_rover', 'rover_0'],
              'Type': 'DD',
              'Parameter': {'action_rover': {'amw': {'rover_0': {'s0': {'type': 'deterministic',
                                                                        'var': 'rover_1',
                                                                        'val': 's2'},
                                                                 's1': {'type': 'deterministic',
                                                                        'var': 'rover_1',
                                                                        'val': 's0'},
                                                                 's2': {'type': 'deterministic',
                                                                        'var': 'rover_1',
                                                                        'val': 's2'}}},
                                             'ame': {'rover_0': {'s0': {'type': 'deterministic',
                                                                        'var': 'rover_1',
                                                                        'val': 's1'},
                                                                 's1': {'type': 'deterministic',
                                                                        'var': 'rover_1',
                                                                        'val': 's2'},
                                                                 's2': {'type': 'deterministic',
                                                                        'var': 'rover_1',
                                                                        'val': 's2'},
                                                                 }},
                                             'ac': {'type': 'persistent',
                                                    'var': 'rover_1'},
                                             'as': {'rover_0': {'s0': {'type': 'deterministic',
                                                                       'var': 'rover_1',
                                                                       'val': 's0'},
                                                                's1': {'type': 'deterministic',
                                                                       'var': 'rover_1',
                                                                       'val': 's2'},
                                                                's2': {'type': 'deterministic',
                                                                       'var': 'rover_1',
                                                                       'val': 's2'}}}}}},
             {'Var': 'rock_1',
              'Parent': ['action_rover', 'rover_0', 'rock_0'],
              'Type': 'DD',
              'Parameter': {'action_rover': {'amw': {'type': 'persistent',
                                                     'var': 'rock_1'},
                                             'ame': {'type': 'persistent',
                                                     'var': 'rock_1'},
                                             'ac': {'type': 'persistent',
                                                    'var': 'rock_1'},
                                             'as': {'rover_0': {'s0': {'type': 'deterministic',
                                                                       'var': 'rock_1',
                                                                       'val': 'bad'},
                                                                's1': {'type': 'persistent',
                                                                       'var': 'rock_1'},
                                                                's2': {'type': 'persistent',
                                                                       'var': 'rock_1'}}}}}}]
        self.maxDiff = None
        self.assertEqual(self.reader_string.get_state_transition_function(),
                         expected_state_transition_function)
        self.assertEqual(self.reader_file.get_state_transition_function(),
                         expected_state_transition_function)

    def test_obs_function_dd(self):
        string = """
        <pomdpx version="1.0" id="rockSample"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="pomdpx.xsd">
         <Description>RockSample problem for map size 1 x 3.
           Rock is at 0, Rover’s initial position is at 1.
           Exit is at 2.
         </Description>
         <Discount>0.95</Discount>
        <ObsFunction>
        <CondProb>
            <Var>obs_sensor</Var>
            <Parent>action_rover rover_1 rock_1</Parent>
            <Parameter type="DD">
                <DAG>
                    <Node var="action_rover">
                        <Edge val="amw">
                            <SubDAG type="deterministic" var="obs_sensor" val="ogood"/>
                        </Edge>
                        <Edge val="ame">
                            <SubDAG type="deterministic" var="obs_sensor" val="ogood"/>
                        </Edge>
                        <Edge val="ac">
                            <Node var="rover_1">
                                <Edge val="s0">
                                    <Node var="rock_1">
                                        <Edge val="good">
                                            <SubDAG type="deterministic" var="obs_sensor" val="ogood"/>
                                        </Edge>
                                        <Edge val="bad">
                                            <SubDAG type="deterministic" var="obs_sensor" val="obad"/>
                                        </Edge>
                                    </Node>
                                </Edge>
                                <Edge val="s1">
                                    <SubDAG type="template" idref="obs_rock"/>
                                </Edge>
                                <Edge val="s2">
                                    <SubDAG type="template" idref="obs_rock"/>
                                </Edge>
                            </Node>
                        </Edge>
                        <Edge val="as">
                            <SubDAG type="deterministic" var="obs_sensor" val="ogood"/>
                        </Edge>
                    </Node>
                </DAG>
                <SubDAGTemplate id="obs_rock">
                    <Node var="rock_1">
                        <Edge val="good">
                            <Node var="obs_sensor">
                                <Edge val="ogood">
                                    <Terminal>0.8</Terminal>
                                </Edge>
                                <Edge val="obad">
                                    <Terminal>0.2</Terminal>
                                </Edge>
                            </Node>
                        </Edge>
                        <Edge val="bad">
                            <Node var="obs_sensor">
                                <Edge val="ogood">
                                    <Terminal>0.2</Terminal>
                                </Edge>
                                <Edge val="obad">
                                    <Terminal>0.8</Terminal>
                                </Edge>
                            </Node>
                        </Edge>
                    </Node>
                </SubDAGTemplate>
            </Parameter>
        </CondProb>
    </ObsFunction>
    </pomdpx>
        """
        self.reader_string = PomdpXReader(string=string)
        self.reader_file = PomdpXReader(path=six.StringIO(string))
        expected_obs_function = \
            [{'Var': 'obs_sensor',
              'Parent': ['action_rover', 'rover_1', 'rock_1'],
              'Type': 'DD',
              'Parameter': {'action_rover': {'amw': {'type': 'deterministic',
                                                     'var': 'obs_sensor',
                                                     'val': 'ogood'},
                                             'ame': {'type': 'deterministic',
                                                     'var': 'obs_sensor',
                                                     'val': 'ogood'},
                                             'ac': {'rover_1': {'s0': {'rock_1': {'good': {'type': 'deterministic',
                                                                                           'var': 'obs_sensor',
                                                                                           'val': 'ogood'},
                                                                                  'bad': {'type': 'deterministic',
                                                                                          'var': 'obs_sensor',
                                                                                          'val': 'obad'}}},
                                                                's1': {'type': 'template',
                                                                       'idref': 'obs_rock'},
                                                                's2': {'type': 'template',
                                                                       'idref': 'obs_rock'}}},
                                             'as': {'type': 'deterministic',
                                                    'var': 'obs_sensor',
                                                    'val': 'ogood'}},
                            'SubDAGTemplate': {'rock_1': {'good': {'obs_sensor': {'ogood': '0.8',
                                                                                  'obad': '0.2'}},
                                                          'bad': {'obs_sensor': {'ogood': '0.2',
                                                                                 'obad': '0.8'}}}},
                            'id': 'obs_rock'}}]
        self.maxDiff = None
        self.assertEqual(self.reader_string.get_obs_function(), expected_obs_function)
        self.assertEqual(self.reader_file.get_obs_function(), expected_obs_function)

    def tearDown(self):
        del self.reader_file
        del self.reader_string


class TestPomdpXWriter(unittest.TestCase):
    def setUp(self):
        self.model_data = {'discription': '',
                           'discount': '0.95',
                           'variables': {
                               'StateVar': [{'vnamePrev': 'rover_0',
                                             'vnameCurr': 'rover_1',
                                             'ValueEnum': ['s0', 's1', 's2'],
                                             'fullyObs': True},
                                            {'vnamePrev': 'rock_0',
                                             'vnameCurr': 'rock_1',
                                             'fullyObs': False,
                                             'ValueEnum': ['good', 'bad']}],
                               'ObsVar': [{'vname': 'obs_sensor',
                                           'ValueEnum': ['ogood', 'obad']}],
                               'RewardVar': [{'vname': 'reward_rover'}],
                               'ActionVar': [{'vname': 'action_rover',
                                              'ValueEnum': ['amw', 'ame',
                                                            'ac', 'as']}]},
                           'initial_state_belief': [{'Var': 'rover_0',
                                                     'Parent': ['null'],
                                                     'Type': 'TBL',
                                                     'Parameter': [{'Instance': ['-'],
                                                                    'ProbTable': ['0.0', '1.0', '0.0']}]},
                                                    {'Var': 'rock_0',
                                                     'Parent': ['null'],
                                                     'Type': 'TBL',
                                                     'Parameter': [{'Instance': ['-'],
                                                                    'ProbTable': ['uniform']}]}],
                           'state_transition_function': [{'Var': 'rover_1',
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
                                                                        ]}],
                           'obs_function': [{'Var': 'obs_sensor',
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
                                                            'ProbTable': ['1.0', '0.0']}]}],
                           'reward_function': [{'Var': 'reward_rover',
                                                'Parent': ['action_rover', 'rover_0', 'rock_0'],
                                                'Type': 'TBL',
                                                'Parameter': [{'Instance': ['ame', 's1', '*'],
                                                               'ValueTable': ['10']},
                                                              {'Instance': ['amw', 's0', '*'],
                                                               'ValueTable': ['-100']},
                                                              {'Instance': ['as', 's1', '*'],
                                                               'ValueTable': ['-100']},
                                                              {'Instance': ['as', 's0', 'good'],
                                                               'ValueTable': ['10']},
                                                              {'Instance': ['as', 's0', 'bad'],
                                                               'ValueTable': ['-10']}]}]}

        self.writer = PomdpXWriter(model_data=self.model_data)

    def test_variables(self):
        expected_variables = etree.XML("""
<Variable>
  <StateVar fullyObs="true" vnameCurr="rover_1" vnamePrev="rover_0">
    <NumValues>3</NumValues>
  </StateVar>
  <StateVar fullyObs="false" vnameCurr="rock_1" vnamePrev="rock_0">
    <ValueEnum>good bad</ValueEnum>
  </StateVar>
  <ObsVar vname="obs_sensor">
    <ValueEnum>ogood obad</ValueEnum>
  </ObsVar>
  <ActionVar vname="action_rover">
    <ValueEnum>amw ame ac as</ValueEnum>
  </ActionVar>
  <RewardVar vname="reward_rover" />
</Variable>""")
        self.maxDiff = None
        self.assertEqual(self.writer.get_variables(),
                         etree.tostring(expected_variables))

    def test_add_initial_belief(self):
        expected_belief_xml = etree.XML("""
<InitialStateBelief>
  <CondProb>
    <Var>rover_0</Var>
    <Parent>null</Parent>
    <Parameter type="TBL">
      <Entry>
        <Instance> - </Instance>
        <ProbTable>0.0 1.0 0.0</ProbTable>
      </Entry>
    </Parameter>
  </CondProb>
  <CondProb>
    <Var>rock_0</Var>
    <Parent>null</Parent>
    <Parameter type="TBL">
      <Entry>
        <Instance> - </Instance>
        <ProbTable>uniform</ProbTable>
      </Entry>
    </Parameter>
  </CondProb>
</InitialStateBelief>""")
        self.maxDiff = None
        self.assertEqual(str(self.writer.add_initial_belief()),
                         str(etree.tostring(expected_belief_xml)))

    def test_add_transition_function(self):
        expected_transition_xml = etree.XML("""
<StateTransitionFunction>
  <CondProb>
    <Var>rover_1</Var>
    <Parent>action_rover rover_0</Parent>
    <Parameter type="TBL">
      <Entry>
        <Instance>amw s0 s2</Instance>
        <ProbTable>1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>amw s1 s0</Instance>
        <ProbTable>1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>ame s0 s1</Instance>
        <ProbTable>1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>ame s1 s2</Instance>
        <ProbTable>1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>ac s0 s0</Instance>
        <ProbTable>1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>ac s1 s1</Instance>
        <ProbTable>1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>as s0 s0</Instance>
        <ProbTable>1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>as s1 s2</Instance>
        <ProbTable>1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>* s2 s2</Instance>
        <ProbTable>1.0</ProbTable>
      </Entry>
    </Parameter>
  </CondProb>
  <CondProb>
    <Var>rock_1</Var>
    <Parent>action_rover rover_0 rock_0</Parent>
    <Parameter type="TBL">
      <Entry>
        <Instance>amw * - - </Instance>
        <ProbTable>1.0 0.0 0.0 1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>ame * - - </Instance>
        <ProbTable>identity</ProbTable>
      </Entry>
      <Entry>
        <Instance>ac * - - </Instance>
        <ProbTable>identity</ProbTable>
      </Entry>
      <Entry>
        <Instance>as * - - </Instance>
        <ProbTable>identity</ProbTable>
      </Entry>
      <Entry>
        <Instance>as s0 * - </Instance>
        <ProbTable>0.0 1.0</ProbTable>
      </Entry>
    </Parameter>
  </CondProb>
</StateTransitionFunction>""")
        self.maxDiff = None
        self.assertEqual(self.writer.add_state_transition_function(),
                         etree.tostring(expected_transition_xml))

    def test_add_obs_function(self):
        expected_obs_xml = etree.XML("""
<ObsFunction>
  <CondProb>
    <Var>obs_sensor</Var>
    <Parent>action_rover rover_1 rock_1</Parent>
    <Parameter type="TBL">
      <Entry>
        <Instance>amw * * - </Instance>
        <ProbTable>1.0 0.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>ame * * - </Instance>
        <ProbTable>1.0 0.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>as * * - </Instance>
        <ProbTable>1.0 0.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>ac s0 - - </Instance>
        <ProbTable>1.0 0.0 0.0 1.0</ProbTable>
      </Entry>
      <Entry>
        <Instance>ac s1 - - </Instance>
        <ProbTable>0.8 0.2 0.2 0.8</ProbTable>
      </Entry>
      <Entry>
        <Instance>ac s2 * - </Instance>
        <ProbTable>1.0 0.0</ProbTable>
      </Entry>
    </Parameter>
  </CondProb>
</ObsFunction>""")
        self.maxDiff = None
        self.assertEqual(self.writer.add_obs_function(),
                         etree.tostring(expected_obs_xml))

    def test_add_reward_function(self):
        expected_reward_xml = etree.XML("""
<RewardFunction>
  <Func>
    <Var>reward_rover</Var>
    <Parent>action_rover rover_0 rock_0</Parent>
    <Parameter type="TBL">
      <Entry>
        <Instance>ame s1 * </Instance>
        <ValueTable>10</ValueTable>
      </Entry>
      <Entry>
        <Instance>amw s0 * </Instance>
        <ValueTable>-100</ValueTable>
      </Entry>
      <Entry>
        <Instance>as s1 * </Instance>
        <ValueTable>-100</ValueTable>
      </Entry>
      <Entry>
        <Instance>as s0 good</Instance>
        <ValueTable>10</ValueTable>
      </Entry>
      <Entry>
        <Instance>as s0 bad</Instance>
        <ValueTable>-10</ValueTable>
      </Entry>
    </Parameter>
  </Func>
</RewardFunction>""")
        self.maxDiff = None
        self.assertEqual(self.writer.add_reward_function(),
                         etree.tostring(expected_reward_xml))

    def test_initial_state_belief_dd(self):
        self.model_data = {'initial_state_belief': [{
            'Var': 'rover_0',
            'Parent': ['null'],
            'Type': 'DD',
            'Parameter': {'rover_0': {'s0': '0.0',
                                      's1': {'type': 'uniform',
                                             'var': 'rock_0'},
                                      's2': '0.0'}}}]}
        self.writer = PomdpXWriter(model_data=self.model_data)
        expected_xml = etree.XML("""
<InitialStateBelief>
  <CondProb>
    <Var>rover_0</Var>
    <Parent>null</Parent>
    <Parameter type="DD">
      <DAG>
        <Node var="rover_0">
          <Edge val="s0">
            <Terminal>0.0</Terminal>
          </Edge>
          <Edge val="s1">
            <SubDAG type="uniform" var="rock_0"/>
          </Edge>
          <Edge val="s2">
            <Terminal>0.0</Terminal>
          </Edge>
        </Node>
      </DAG>
    </Parameter>
  </CondProb>
</InitialStateBelief>""")
        self.maxDiff = None
        self.assertEqual(self.writer.add_initial_belief(),
                         etree.tostring(expected_xml))

    def test_state_transition_function_dd(self):
        self.model_data = {
            'state_transition_function': [{
                'Var': 'rover_1',
                'Parent': ['action_rover', 'rover_0'],
                'Type': 'DD',
                'Parameter': {'action_rover': {
                    'amw': {'rover_0': {'s0': {
                        'type': 'deterministic',
                        'var': 'rover_1',
                        'val': 's2'},
                        's1': {'type': 'deterministic',
                               'var': 'rover_1',
                               'val': 's0'},
                        's2': {'type': 'deterministic',
                               'var': 'rover_1',
                               'val': 's2'}}},
                    'ame': {'rover_0': {'s0': {'type': 'deterministic',
                                               'var': 'rover_1',
                                               'val': 's1'},
                                        's1': {'type': 'deterministic',
                                               'var': 'rover_1',
                                               'val': 's2'},
                                        's2': {'type': 'deterministic',
                                               'var': 'rover_1',
                                               'val': 's2'},
                                        }},
                    'ac': {'type': 'persistent',
                           'var': 'rover_1'},
                    'as': {'rover_0': {'s0': {'type': 'deterministic',
                                              'var': 'rover_1',
                                              'val': 's0'},
                                       's1': {'type': 'deterministic',
                                              'var': 'rover_1',
                                              'val': 's2'},
                                       's2': {'type': 'deterministic',
                                              'var': 'rover_1',
                                              'val': 's2'}}}}}},
                {'Var': 'rock_1',
                 'Parent': ['action_rover', 'rover_0', 'rock_0'],
                 'Type': 'DD',
                 'Parameter': {'action_rover': {
                     'amw': {'type': 'persistent',
                             'var': 'rock_1'},
                     'ame': {'type': 'persistent',
                             'var': 'rock_1'},
                     'ac': {'type': 'persistent',
                            'var': 'rock_1'},
                     'as': {'rover_0': {'s0': {'type': 'deterministic',
                                               'var': 'rock_1',
                                               'val': 'bad'},
                                        's1': {'type': 'persistent',
                                               'var': 'rock_1'},
                                        's2': {'type': 'persistent',
                                               'var': 'rock_1'}}}}}}]}

        self.writer = PomdpXWriter(model_data=self.model_data)
        expected_xml = etree.XML("""
<StateTransitionFunction>
  <CondProb>
    <Var>rover_1</Var>
    <Parent>action_rover rover_0</Parent>
    <Parameter type="DD">
      <DAG>
        <Node var="action_rover">
          <Edge val="ac">
            <SubDAG type="persistent" var="rover_1"/>
          </Edge>
          <Edge val="ame">
            <Node var="rover_0">
              <Edge val="s0">
                <SubDAG type="deterministic" val="s1" var="rover_1"/>
              </Edge>
              <Edge val="s1">
                <SubDAG type="deterministic" val="s2" var="rover_1"/>
              </Edge>
              <Edge val="s2">
                <SubDAG type="deterministic" val="s2" var="rover_1"/>
              </Edge>
            </Node>
          </Edge>
          <Edge val="amw">
            <Node var="rover_0">
              <Edge val="s0">
                <SubDAG type="deterministic" val="s2" var="rover_1"/>
              </Edge>
              <Edge val="s1">
                <SubDAG type="deterministic" val="s0" var="rover_1"/>
              </Edge>
              <Edge val="s2">
                <SubDAG type="deterministic" val="s2" var="rover_1"/>
              </Edge>
            </Node>
          </Edge>
          <Edge val="as">
            <Node var="rover_0">
              <Edge val="s0">
                <SubDAG type="deterministic" val="s0" var="rover_1"/>
              </Edge>
              <Edge val="s1">
                <SubDAG type="deterministic" val="s2" var="rover_1"/>
              </Edge>
              <Edge val="s2">
                <SubDAG type="deterministic" val="s2" var="rover_1"/>
              </Edge>
            </Node>
          </Edge>
        </Node>
      </DAG>
    </Parameter>
  </CondProb>
  <CondProb>
    <Var>rock_1</Var>
    <Parent>action_rover rover_0 rock_0</Parent>
    <Parameter type="DD">
      <DAG>
        <Node var="action_rover">
          <Edge val="ac">
            <SubDAG type="persistent" var="rock_1"/>
          </Edge>
          <Edge val="ame">
            <SubDAG type="persistent" var="rock_1"/>
          </Edge>
          <Edge val="amw">
            <SubDAG type="persistent" var="rock_1"/>
          </Edge>
          <Edge val="as">
            <Node var="rover_0">
              <Edge val="s0">
                <SubDAG type="deterministic" val="bad" var="rock_1"/>
              </Edge>
              <Edge val="s1">
                <SubDAG type="persistent" var="rock_1"/>
              </Edge>
              <Edge val="s2">
                <SubDAG type="persistent" var="rock_1"/>
              </Edge>
            </Node>
          </Edge>
        </Node>
      </DAG>
    </Parameter>
  </CondProb>
</StateTransitionFunction>""")
        self.maxDiff = None
        self.assertEqual(str(self.writer.add_state_transition_function()),
                         str(etree.tostring(expected_xml)))

    def test_obs_function_dd(self):
        self.model_data = {
            'obs_function': [{
                'Var': 'obs_sensor',
                'Parent': ['action_rover', 'rover_1', 'rock_1'],
                'Type': 'DD',
                'Parameter': {'action_rover': {
                    'amw': {'type': 'deterministic',
                            'var': 'obs_sensor',
                            'val': 'ogood'},
                    'ame': {'type': 'deterministic',
                            'var': 'obs_sensor',
                            'val': 'ogood'},
                    'ac': {'rover_1': {'s0': {'rock_1': {'good': {
                        'type': 'deterministic',
                        'var': 'obs_sensor',
                        'val': 'ogood'},
                        'bad': {'type': 'deterministic',
                                'var': 'obs_sensor',
                                'val': 'obad'}}},
                        's1': {'type': 'template',
                               'idref': 'obs_rock'},
                        's2': {'type': 'template',
                               'idref': 'obs_rock'}}},
                    'as': {'type': 'deterministic',
                           'var': 'obs_sensor',
                           'val': 'ogood'}},
                    'SubDAGTemplate': {'rock_1': {'good': {'obs_sensor': {
                        'ogood': '0.8',
                        'obad': '0.2'}},
                        'bad': {'obs_sensor': {
                            'ogood': '0.2',
                            'obad': '0.8'}}}},
                    'id': 'obs_rock'}}]}

        self.writer = PomdpXWriter(model_data=self.model_data)
        expected_xml = etree.XML("""
<ObsFunction>
  <CondProb>
    <Var>obs_sensor</Var>
    <Parent>action_rover rover_1 rock_1</Parent>
    <Parameter type="DD">
      <DAG>
        <Node var="action_rover">
          <Edge val="ac">
            <Node var="rover_1">
              <Edge val="s0">
                <Node var="rock_1">
                  <Edge val="bad">
                    <SubDAG type="deterministic" val="obad" var="obs_sensor" />
                  </Edge>
                  <Edge val="good">
                    <SubDAG type="deterministic" val="ogood" var="obs_sensor"/>
                  </Edge>
                </Node>
              </Edge>
              <Edge val="s1">
                <SubDAG idref="obs_rock" type="template"/>
              </Edge>
              <Edge val="s2">
                <SubDAG idref="obs_rock" type="template"/>
              </Edge>
            </Node>
          </Edge>
          <Edge val="ame">
            <SubDAG type="deterministic" val="ogood" var="obs_sensor"/>
          </Edge>
          <Edge val="amw">
            <SubDAG type="deterministic" val="ogood" var="obs_sensor" />
          </Edge>
          <Edge val="as">
            <SubDAG type="deterministic" val="ogood" var="obs_sensor"/>
          </Edge>
        </Node>
      </DAG>
      <SubDAGTemplate id="obs_rock">
        <Node var="rock_1">
          <Edge val="bad">
            <Node var="obs_sensor">
              <Edge val="obad">
                <Terminal>0.8</Terminal>
              </Edge>
              <Edge val="ogood">
                <Terminal>0.2</Terminal>
              </Edge>
            </Node>
          </Edge>
          <Edge val="good">
            <Node var="obs_sensor">
              <Edge val="obad">
                <Terminal>0.2</Terminal>
              </Edge>
              <Edge val="ogood">
                <Terminal>0.8</Terminal>
              </Edge>
            </Node>
          </Edge>
        </Node>
      </SubDAGTemplate>
    </Parameter>
  </CondProb>
</ObsFunction>""")
        self.maxDiff = None
        self.assertEqual(str(self.writer.add_obs_function()),
                         str(etree.tostring(expected_xml)))

    def test_reward_function_dd(self):
        self.model_data = {
            'reward_function': [{
                'Var': 'reward_rover',
                'Parent': ['action_rover', 'rover_0', 'rock_0'],
                'Type': 'DD',
                'Parameter': {
                    'action_rover': {
                        'amw': {'rover_0': {'s0': '-100.0',
                                            's1': '0.0',
                                            's2': '0.0'}},
                        'ame': {'rover_0': {'s0': '0.0',
                                            's1': '10.0',
                                            's2': '0.0'}},
                        'ac': '0.0',
                        'as': {'rover_0': {'s0': {'rock_0': {'good': '10',
                                                             'bad': '-10'}},
                                           's1': '-100',
                                           's2': '-100'}}}}}]}

        self.writer = PomdpXWriter(model_data=self.model_data)
        expected_xml = etree.XML("""
<RewardFunction>
  <Func>
    <Var>reward_rover</Var>
    <Parent>action_rover rover_0 rock_0</Parent>
    <Parameter type="DD">
      <DAG>
        <Node var="action_rover">
          <Edge val="ac">
            <Terminal>0.0</Terminal>
          </Edge>
          <Edge val="ame">
            <Node var="rover_0">
              <Edge val="s0">
                <Terminal>0.0</Terminal>
              </Edge>
              <Edge val="s1">
                <Terminal>10.0</Terminal>
              </Edge>
              <Edge val="s2">
                <Terminal>0.0</Terminal>
              </Edge>
            </Node>
          </Edge>
          <Edge val="amw">
            <Node var="rover_0">
              <Edge val="s0">
                <Terminal>-100.0</Terminal>
              </Edge>
              <Edge val="s1">
                <Terminal>0.0</Terminal>
              </Edge>
              <Edge val="s2">
                <Terminal>0.0</Terminal>
              </Edge>
            </Node>
          </Edge>
          <Edge val="as">
            <Node var="rover_0">
              <Edge val="s0">
                <Node var="rock_0">
                  <Edge val="bad">
                    <Terminal>-10</Terminal>
                  </Edge>
                  <Edge val="good">
                    <Terminal>10</Terminal>
                  </Edge>
                </Node>
              </Edge>
              <Edge val="s1">
                <Terminal>-100</Terminal>
              </Edge>
              <Edge val="s2">
                <Terminal>-100</Terminal>
              </Edge>
            </Node>
          </Edge>
        </Node>
      </DAG>
    </Parameter>
  </Func>
</RewardFunction>""")
        self.maxDiff = None
        self.assertEqual(self.writer.add_reward_function(),
                         etree.tostring(expected_xml))
