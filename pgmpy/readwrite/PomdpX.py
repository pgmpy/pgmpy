#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from collections import defaultdict

import warnings


try:
    from lxml import etree
except ImportError:
    try:
        import xml.etree.ElementTree as etree
    except ImportError:
        # import xml.etree.cElementTree as etree
        # print("running with cElementTree on Python 2.5+")
        # Commented out because behaviour is different from expected

        warnings.warn("Failed to import ElementTree from any known place")


class PomdpXReader(object):
    """
    Class for reading PomdpX file format from files or strings
    """

    def __init__(self, path=None, string=None):
        """
        Initialize an instance of PomdpX reader class

        Parameters
        ----------
        path : file or str
            Path of the file containing PomdpX information.

        string : str
            String containing PomdpX information.

        Example
        -------
        reader = PomdpXReader('TestPomdpX.xml')

        Reference
        ---------
        http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation
        """
        if path:
            self.network = etree.ElementTree(file=path).getroot()
        elif string:
            self.network = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")

    def get_description(self):
        """
        Return the problem description

        Example
        >>> reader = PomdpXReader('Test_Pomdpx.xml')
        >>> reader.get_description()
        'RockSample problem for map size 1 x 3.
        Rock is at 0, Rover’s initial position is at 1.
        Exit is at 2.'
        --------
        >>> reader = PomdpXReader('Test_PomdpX.xml')
        >>> reader.get_description()
        'RockSample problem for map size 1 x 3.
         Rock is at 0, Rover’s initial position is at 1.
         Exit is at 2.'
        """
        return self.network.find("Description").text

    def get_discount(self):
        """
        Returns the discount factor for the problem

        Example
        --------
        >>> reader = PomdpXReader('Test_PomdpX.xml')
        >>> reader.get_discount()
        0.95
        """
        return float(self.network.find("Discount").text)

    def get_variables(self):
        """
        Returns list of variables of the network

        Example
        -------
        >>> reader = PomdpXReader("pomdpx.xml")
        >>> reader.get_variables()
        {'StateVar': [
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
        """
        self.variables = defaultdict(list)
        for variable in self.network.findall("Variable"):
            _variables = defaultdict(list)
            for var in variable.findall("StateVar"):
                state_variables = defaultdict(list)
                state_variables["vnamePrev"] = var.get("vnamePrev")
                state_variables["vnameCurr"] = var.get("vnameCurr")
                if var.get("fullyObs"):
                    state_variables["fullyObs"] = True
                else:
                    state_variables["fullyObs"] = False
                state_variables["ValueEnum"] = []
                if var.find("NumValues") is not None:
                    for i in range(0, int(var.find("NumValues").text)):
                        state_variables["ValueEnum"].append("s" + str(i))
                if var.find("ValueEnum") is not None:
                    state_variables["ValueEnum"] = var.find("ValueEnum").text.split()
                _variables["StateVar"].append(state_variables)

            for var in variable.findall("ObsVar"):
                obs_variables = defaultdict(list)
                obs_variables["vname"] = var.get("vname")
                obs_variables["ValueEnum"] = var.find("ValueEnum").text.split()
                _variables["ObsVar"].append(obs_variables)

            for var in variable.findall("ActionVar"):
                action_variables = defaultdict(list)
                action_variables["vname"] = var.get("vname")
                action_variables["ValueEnum"] = var.find("ValueEnum").text.split()
                _variables["ActionVar"].append(action_variables)

            for var in variable.findall("RewardVar"):
                reward_variables = defaultdict(list)
                reward_variables["vname"] = var.get("vname")
                _variables["RewardVar"].append(reward_variables)

            self.variables.update(_variables)

        return self.variables

    def get_initial_beliefs(self):
        """
        Returns the state, action and observation variables as a dictionary
        in the case of table type parameter and a nested structure in case of
        decision diagram parameter

        Examples
        --------
        >>> reader = PomdpXReader('Test_PomdpX.xml')
        >>> reader.get_initial_beliefs()
        [{'Var': 'rover_0',
          'Parent': ['null'],
          'Type': 'TBL',
          'Parameter': [{'Instance': ['-'],
          'ProbTable': ['0.0', '1.0', '0.0']}]
         },
         {'Var': '',
          '...': ...,'
          '...': '...',
          }]
        """
        initial_state_belief = []
        for variable in self.network.findall("InitialStateBelief"):
            for var in variable.findall("CondProb"):
                cond_prob = defaultdict(list)
                cond_prob["Var"] = var.find("Var").text
                cond_prob["Parent"] = var.find("Parent").text.split()
                if not var.find("Parameter").get("type"):
                    cond_prob["Type"] = "TBL"
                else:
                    cond_prob["Type"] = var.find("Parameter").get("type")
                cond_prob["Parameter"] = self.get_parameter(var)
                initial_state_belief.append(cond_prob)

        return initial_state_belief

    def get_state_transition_function(self):
        """
        Returns the transition of the state variables as nested dict in the
        case of table type parameter and a nested structure in case of
        decision diagram parameter

        Example
        --------
        >>> reader = PomdpXReader('Test_PomdpX.xml')
        >>> reader.get_state_transition_function()
        [{'Var': 'rover_1',
          'Parent': ['action_rover', 'rover_0'],
          'Type': 'TBL',
          'Parameter': [{'Instance': ['amw', 's0', 's2'],
                         'ProbTable': ['1.0']},
                         {'Instance': ['amw', 's1', 's0'],
                         'ProbTable': ['1.0']},
                         ...
                        ]
        }]
        """
        state_transition_function = []
        for variable in self.network.findall("StateTransitionFunction"):
            for var in variable.findall("CondProb"):
                cond_prob = defaultdict(list)
                cond_prob["Var"] = var.find("Var").text
                cond_prob["Parent"] = var.find("Parent").text.split()
                if not var.find("Parameter").get("type"):
                    cond_prob["Type"] = "TBL"
                else:
                    cond_prob["Type"] = var.find("Parameter").get("type")
                cond_prob["Parameter"] = self.get_parameter(var)
                state_transition_function.append(cond_prob)

        return state_transition_function

    def get_obs_function(self):
        """
        Returns the observation function as nested dict in the case of table-
        type parameter and a nested structure in case of
        decision diagram parameter

        Example
        --------
        >>> reader = PomdpXReader('Test_PomdpX.xml')
        >>> reader.get_obs_function()
        [{'Var': 'obs_sensor',
              'Parent': ['action_rover', 'rover_1', 'rock_1'],
              'Type': 'TBL',
              'Parameter': [{'Instance': ['amw', '*', '*', '-'],
                             'ProbTable': ['1.0', '0.0']},
                         ...
                        ]
        }]
        """
        obs_function = []
        for variable in self.network.findall("ObsFunction"):
            for var in variable.findall("CondProb"):
                cond_prob = defaultdict(list)
                cond_prob["Var"] = var.find("Var").text
                cond_prob["Parent"] = var.find("Parent").text.split()
                if not var.find("Parameter").get("type"):
                    cond_prob["Type"] = "TBL"
                else:
                    cond_prob["Type"] = var.find("Parameter").get("type")
                cond_prob["Parameter"] = self.get_parameter(var)
                obs_function.append(cond_prob)

        return obs_function

    def get_reward_function(self):
        """
        Returns the reward function as nested dict in the case of table-
        type parameter and a nested structure in case of
        decision diagram parameter

        Example
        --------
        >>> reader = PomdpXReader('Test_PomdpX.xml')
        >>> reader.get_reward_function()
        [{'Var': 'reward_rover',
              'Parent': ['action_rover', 'rover_0', 'rock_0'],
              'Type': 'TBL',
              'Parameter': [{'Instance': ['ame', 's1', '*'],
                             'ValueTable': ['10']},
                         ...
                        ]
        }]
        """
        reward_function = []
        for variable in self.network.findall("RewardFunction"):
            for var in variable.findall("Func"):
                func = defaultdict(list)
                func["Var"] = var.find("Var").text
                func["Parent"] = var.find("Parent").text.split()
                if not var.find("Parameter").get("type"):
                    func["Type"] = "TBL"
                else:
                    func["Type"] = var.find("Parameter").get("type")
                func["Parameter"] = self.get_parameter(var)
                reward_function.append(func)

        return reward_function

    def get_parameter(self, var):
        """
        This method supports the functional tags by providing the actual
        values in the function as list of dict in case of table type parameter or as
        nested dict in case of decision diagram
        """
        parameter = []

        for parameter_tag in var.findall("Parameter"):
            parameter_type = "TBL"
            if parameter_tag.get("type") is not None:
                parameter_type = parameter_tag.get("type")
            if parameter_type == "TBL":
                parameter = self.get_parameter_tbl(parameter_tag)
            elif parameter_type == "DD":
                parameter = defaultdict(list)
                parameter = self.get_parameter_dd(parameter_tag)

        return parameter

    def get_parameter_tbl(self, parameter):
        """
        This method returns parameters as list of dict in case of table type
        parameter
        """
        par = []
        for entry in parameter.findall("Entry"):
            instance = defaultdict(list)
            instance["Instance"] = entry.find("Instance").text.split()
            if entry.find("ProbTable") is None:
                instance["ValueTable"] = entry.find("ValueTable").text.split()
            else:
                instance["ProbTable"] = entry.find("ProbTable").text.split()
            par.append(instance)
        return par

    def get_parameter_dd(self, parameter):
        """
        This method returns parameters as nested dicts in case of decision
        diagram parameter.
        """
        dag = defaultdict(list)
        dag_elem = parameter.find("DAG")
        node = dag_elem.find("Node")
        root = node.get("var")

        def get_param(node):
            edges = defaultdict(list)
            for edge in node.findall("Edge"):
                if edge.find("Terminal") is not None:
                    edges[edge.get("val")] = edge.find("Terminal").text
                elif edge.find("Node") is not None:
                    node_cpd = defaultdict(list)
                    node_cpd[edge.find("Node").get("var")] = get_param(
                        edge.find("Node")
                    )
                    edges[edge.get("val")] = node_cpd
                elif edge.find("SubDAG") is not None:
                    subdag_attribute = defaultdict(list)
                    subdag_attribute["type"] = edge.find("SubDAG").get("type")
                    if subdag_attribute["type"] == "template":
                        subdag_attribute["idref"] = edge.find("SubDAG").get("idref")
                    if edge.find("SubDAG").get("var"):
                        subdag_attribute["var"] = edge.find("SubDAG").get("var")
                    if edge.find("SubDAG").get("val"):
                        subdag_attribute["val"] = edge.find("SubDAG").get("val")
                    edges[edge.get("val")] = subdag_attribute
            return edges

        if parameter.find("SubDAGTemplate") is not None:
            SubDAGTemplate = parameter.find("SubDAGTemplate")
            subdag_root = SubDAGTemplate.find("Node")
            subdag_node = subdag_root.get("var")
            subdag_dict = defaultdict(list)
            subdag_dict[subdag_node] = get_param(subdag_root)
            dag["SubDAGTemplate"] = subdag_dict
            dag["id"] = SubDAGTemplate.get("id")
        dag[root] = get_param(node)
        return dag


class PomdpXWriter(object):
    """
    Class for writing models in PomdpX
    """

    def __init__(self, model_data, encoding="utf-8", prettyprint=True):
        """
        Initialise a PomdpXWriter Object

        Parameters
        ---------------
        model: A Bayesian of Markov Model
            The model to write
        encoding: String(optional)
            Encoding for text data
        prettyprint: Bool(optional)
            Indentation in output XML if true
        """
        self.model = model_data

        self.encoding = encoding
        self.prettyprint = prettyprint

        self.xml = etree.Element("pomdpx", attrib={"version": "1.0"})
        self.description = etree.SubElement(self.xml, "Description")
        self.discount = etree.SubElement(self.xml, "Discount")
        self.variable = etree.SubElement(self.xml, "Variable")
        self.initial_belief = etree.SubElement(self.xml, "InitialStateBelief")
        self.transition_function = etree.SubElement(self.xml, "StateTransitionFunction")
        self.observation_function = etree.SubElement(self.xml, "ObsFunction")
        self.reward_function = etree.SubElement(self.xml, "RewardFunction")

    def __str__(self, xml):
        """
        Return the XML as string.
        """
        if self.prettyprint:
            self.indent(xml)
        return etree.tostring(xml, encoding=self.encoding)

    def indent(self, elem, level=0):
        """
        Inplace prettyprint formatter.
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _add_value_enum(self, var, tag):
        """
        supports adding variables to the xml

        Parameters
        ---------------
        var: The SubElement variable
        tag: The SubElement tag to which enum value is to be added

        Return
        ---------------
        None
        """
        if var["ValueEnum"][0] == "s0":
            numvalues_tag = etree.SubElement(tag, "NumValues")
            numvalues_tag.text = str(int(var["ValueEnum"][-1][-1]) + 1)
        else:
            valueenum_tag = etree.SubElement(tag, "ValueEnum")
            valueenum_tag.text = ""
            for value in var["ValueEnum"]:
                valueenum_tag.text += value + " "
            valueenum_tag.text = valueenum_tag.text[:-1]

    def get_variables(self):
        """
        Add variables to PomdpX

        Return
        ---------------
        xml containing variables tag
        """
        state_variables = self.model["variables"]["StateVar"]
        for var in state_variables:
            state_var_tag = etree.SubElement(
                self.variable,
                "StateVar",
                attrib={
                    "vnamePrev": var["vnamePrev"],
                    "vnameCurr": var["vnameCurr"],
                    "fullyObs": "true" if var["fullyObs"] else "false",
                },
            )
            self._add_value_enum(var, state_var_tag)

        obs_variables = self.model["variables"]["ObsVar"]
        for var in obs_variables:
            obs_var_tag = etree.SubElement(
                self.variable, "ObsVar", attrib={"vname": var["vname"]}
            )
            self._add_value_enum(var, obs_var_tag)

        action_variables = self.model["variables"]["ActionVar"]
        for var in action_variables:
            action_var_tag = etree.SubElement(
                self.variable, "ActionVar", attrib={"vname": var["vname"]}
            )
            self._add_value_enum(var, action_var_tag)

        reward_var = self.model["variables"]["RewardVar"]
        for var in reward_var:
            etree.SubElement(self.variable, "RewardVar", attrib={"vname": var["vname"]})

        return self.__str__(self.variable)[:-1]

    def add_parameter_dd(self, dag_tag, node_dict):
        """
        helper function for adding parameters in condition

        Parameters
        ---------------
        dag_tag: etree SubElement
                 the DAG tag is contained in this subelement
        node_dict: dictionary
                   the decision diagram dictionary

        Return
        ---------------
        None
        """
        if isinstance(node_dict, defaultdict) or isinstance(node_dict, dict):
            node_tag = etree.SubElement(
                dag_tag, "Node", attrib={"var": next(iter(node_dict.keys()))}
            )
            edge_dict = next(iter(node_dict.values()))
            for edge in sorted(edge_dict.keys(), key=tuple):
                edge_tag = etree.SubElement(node_tag, "Edge", attrib={"val": edge})
                value = edge_dict.get(edge)
                if isinstance(value, str):
                    terminal_tag = etree.SubElement(edge_tag, "Terminal")
                    terminal_tag.text = value
                elif "type" in value:
                    if "val" in value:
                        etree.SubElement(
                            edge_tag,
                            "SubDAG",
                            attrib={
                                "type": value["type"],
                                "var": value["var"],
                                "val": value["val"],
                            },
                        )
                    elif "idref" in value:
                        etree.SubElement(
                            edge_tag,
                            "SubDAG",
                            attrib={"type": value["type"], "idref": value["idref"]},
                        )
                    else:
                        etree.SubElement(
                            edge_tag,
                            "SubDAG",
                            attrib={"type": value["type"], "var": value["var"]},
                        )
                else:
                    self.add_parameter_dd(edge_tag, value)

    def add_conditions(self, condition, condprob):
        """
        helper function for adding probability conditions for model\

        Parameters
        ---------------

        condition:  dictionary
                    contains and element of conditions list
        condprob:   etree SubElement
                    the tag to which condition is added

        Return
        ---------------
        None
        """
        var_tag = etree.SubElement(condprob, "Var")
        var_tag.text = condition["Var"]
        parent_tag = etree.SubElement(condprob, "Parent")
        parent_tag.text = ""
        for parent in condition["Parent"]:
            parent_tag.text += parent + " "
        parent_tag.text = parent_tag.text[:-1]
        parameter_tag = etree.SubElement(
            condprob,
            "Parameter",
            attrib={
                "type": condition["Type"] if condition["Type"] is not None else "TBL"
            },
        )
        if condition["Type"] == "DD":
            dag_tag = etree.SubElement(parameter_tag, "DAG")
            parameter_dict = condition["Parameter"]
            if "SubDAGTemplate" in parameter_dict:
                subdag_tag = etree.SubElement(
                    parameter_tag, "SubDAGTemplate", attrib={"id": parameter_dict["id"]}
                )
                self.add_parameter_dd(subdag_tag, parameter_dict["SubDAGTemplate"])
                del parameter_dict["SubDAGTemplate"]
                del parameter_dict["id"]
                self.add_parameter_dd(dag_tag, parameter_dict)
            else:
                self.add_parameter_dd(dag_tag, parameter_dict)
        else:
            for parameter in condition["Parameter"]:
                entry = etree.SubElement(parameter_tag, "Entry")
                instance = etree.SubElement(entry, "Instance")
                instance.text = ""
                for instance_var in parameter["Instance"]:
                    instance.text += instance_var + " "
                length_instance = len(parameter["Instance"])
                if len(parameter["Instance"][length_instance - 1]) > 1:
                    instance.text = instance.text[:-1]
                if len(parameter["Instance"]) == 1:
                    instance.text = " " + instance.text
                if condprob.tag == "Func":
                    table = "ValueTable"
                else:
                    table = "ProbTable"
                prob_table = parameter[table]
                prob_table_tag = etree.SubElement(entry, table)
                prob_table_tag.text = ""
                for probability in prob_table:
                    prob_table_tag.text += probability + " "
                prob_table_tag.text = prob_table_tag.text[:-1]

    def add_initial_belief(self):
        """
        add initial belief tag to pomdpx model

        Return
        ---------------
        string containing the xml for initial belief tag
        """
        initial_belief = self.model["initial_state_belief"]
        for condition in initial_belief:
            condprob = etree.SubElement(self.initial_belief, "CondProb")
            self.add_conditions(condition, condprob)
        return self.__str__(self.initial_belief)[:-1]

    def add_state_transition_function(self):
        """
        add state transition function tag to pomdpx model

        Return
        ---------------
        string containing the xml for state transition tag
        """
        state_transition_function = self.model["state_transition_function"]
        for condition in state_transition_function:
            condprob = etree.SubElement(self.transition_function, "CondProb")
            self.add_conditions(condition, condprob)
        return self.__str__(self.transition_function)[:-1]

    def add_obs_function(self):
        """
        add observation function tag to pomdpx model

        Return
        ---------------
        string containing the xml for observation function tag
        """
        obs_function = self.model["obs_function"]
        for condition in obs_function:
            condprob = etree.SubElement(self.observation_function, "CondProb")
            self.add_conditions(condition, condprob)
        return self.__str__(self.observation_function)[:-1]

    def add_reward_function(self):
        """
        add reward function tag to pomdpx model

        Return
        ---------------
        string containing the xml for reward function tag
        """
        reward_function = self.model["reward_function"]
        for condition in reward_function:
            condprob = etree.SubElement(self.reward_function, "Func")
            self.add_conditions(condition, condprob)
        return self.__str__(self.reward_function)[:-1]
