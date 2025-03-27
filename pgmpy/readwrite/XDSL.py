import xml.dom.minidom as md
import xml.etree.ElementTree as etree
from io import BytesIO
from itertools import chain

import networkx as nx
import numpy as np

from pgmpy.factors.discrete import State, TabularCPD
from pgmpy.models import BayesianNetwork


class XDSLReader(object):
    def __init__(self, path=None, string=None):
        if path:
            self.network = etree.ElementTree(file=path).getroot()
        elif string:
            self.network = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")
        self.network_name = self.network.attrib["id"]
        self.variables = self.get_variables()
        self.variable_parents = self.get_parents()
        self.edge_list = self.get_edges()
        self.variable_states = self.get_states()
        self.variable_CPD = self.get_values()

    def get_parents(self):
        variable_parents = {}
        nodes = self.network.find("nodes").findall("cpt")
        for node in nodes:
            parents = node.find("parents")
            if parents is not None:
                variable_parents[node.attrib["id"]] = parents.text.split(" ")
            else:
                variable_parents[node.attrib["id"]] = []

        return variable_parents

    def get_variables(self):
        nodes = self.network.find("nodes")
        variables = [variable.attrib["id"] for variable in nodes.findall("cpt")]
        # variables.extend(
        #    [variable.attrib["id"] for variable in nodes.findall("deterministic")]
        # )
        return variables

    def get_edges(self):
        edge_list = [
            [value, key]
            for key in self.variable_parents
            for value in self.variable_parents[key]
        ]
        return edge_list

    def get_states(self):
        nodes = self.network.find("nodes").findall("cpt")
        variable_states = {}
        for cpt in nodes:
            variable_states[cpt.attrib["id"]] = [
                state.attrib["id"] for state in cpt.findall("state")
            ]
        return variable_states

    def get_values(self):
        variable_CPD = {}
        nodes = self.network.find("nodes").findall("cpt")
        for cpt in nodes:

            combined_prob = cpt.find("probabilities")
            num_states = len([state for state in cpt.findall("state")])
            cpd_arr = [[] for k in range(num_states)]
            prob_values = combined_prob.text.split(" ")

            for j in range(num_states):
                for i in range(j, len(prob_values), num_states):
                    cpd_arr[j].append(prob_values[i])

            variable_CPD[cpt.attrib["id"]] = cpd_arr
        return variable_CPD

    def get_model(self, state_name_type=str):
        model = BayesianNetwork()
        model.add_nodes_from(self.variables)
        model.add_edges_from(self.edge_list)
        model.name = self.network_name

        tabular_cpds = []
        for var, values in self.variable_CPD.items():
            evidence_card = [
                len(self.variable_states[evidence_var])
                for evidence_var in self.variable_parents[var]
            ]
            cpd = TabularCPD(
                var,
                len(self.variable_states[var]),
                values,
                evidence=self.variable_parents[var],
                evidence_card=evidence_card,
                state_names={
                    var: list(map(state_name_type, self.variable_states[var]))
                    for var in chain([var], self.variable_parents[var])
                },
            )
            tabular_cpds.append(cpd)

        model.add_cpds(*tabular_cpds)

        return model


class XDSLWriter(object):

    def __init__(
        self,
        model,
        network_id="MyNetwork",
        num_samples="10000",
        disc_samples="10000",
        encoding="utf-8",
        prettyprint=True,
    ):
        if not isinstance(model, BayesianNetwork):
            raise TypeError("model must an instance of BayesianNetwork")
        self.model = model
        self.network_id = network_id
        self.root = etree.Element(
            "smile",
            {
                "version": "1.0",
                "id": network_id,
                "numsamples": num_samples,
                "discsamples": disc_samples,
            },
        )

        self.variables = self.get_variables()
        self.states = self.get_states()
        self._create_extensions()

    def get_variables(self):
        """
        Variables of the model and their corresponding XML elements/representation
        """
        variable_tag = {}
        nodes_elem = etree.SubElement(self.root, "nodes")

        for var in self.model.nodes:
            variable_tag[var] = etree.SubElement(nodes_elem, "cpt", {"id": var})
            # etree.SubElement(variable_tag[var], "NAME").text = var

        return variable_tag

    def get_states(self):
        nodes_elem = etree.SubElement(self.root, "nodes")
        outcome_tag = {}
        cpds = self.model.get_cpds()
        cpd_vars = [cpd.variable for cpd in cpds]
        for var in self.model.nodes:
            idx = cpd_vars.index(var)
            cpd = cpds[idx]

            cpt_elem = self.variables[var]
            # Determine state names: if available in cpd.state_names, use them; otherwise, default to string indices.
            if (
                hasattr(cpd, "state_names")
                and cpd.state_names is not None
                and cpd.variable in cpd.state_names
            ):
                states = cpd.state_names[cpd.variable]
            else:
                states = [str(i) for i in range(cpd.variable_card)]

            # Add a <state> element for each state.
            for st in states:
                etree.SubElement(cpt_elem, "state", {"id": st})

            # Use the network structure to determine the parents in the correct order.
            evidence = cpd.variables
            if len(evidence) > 1:
                parents_str = " ".join(evidence[1:])
                parents_elem = etree.SubElement(cpt_elem, "parents")
                parents_elem.text = parents_str

            # Add the <probabilities> element.
            probs_elem = etree.SubElement(cpt_elem, "probabilities")
            values = np.array(cpd.get_values())
            # Flatten in column-major order so that for each parent configuration the probabilities for all states are listed.
            flat_values = values.flatten(order="F")
            probs_elem.text = " ".join("{:.17f}".format(x) for x in flat_values)

        return outcome_tag

    def _create_extensions(self):

        # Create the <extensions> block with a minimal <genie> element for layout information.
        extensions_elem = etree.SubElement(self.root, "extensions")
        genie_elem = etree.SubElement(
            extensions_elem,
            "genie",
            {
                "version": "1.0",
                "app": "GeNIe 5.0.4830.0 ACADEMIC",
                "name": self.network_id,
            },
        )

        # Provide default layout for each node.
        pos_x, pos_y = 100, 100
        for node in list(nx.topological_sort(self.model)):
            node_elem = etree.SubElement(genie_elem, "node", {"id": node})

            # Set the node name.
            name_elem = etree.SubElement(node_elem, "name")
            name_elem.text = node

            # Appearance details (colors, font).
            etree.SubElement(node_elem, "interior", {"color": "e5f6f7"})
            etree.SubElement(node_elem, "outline", {"color": "000080"})
            etree.SubElement(
                node_elem, "font", {"color": "000000", "name": "Arial", "size": "8"}
            )

            # Set node position (x1, y1, x2, y2).
            pos_elem = etree.SubElement(node_elem, "position")
            pos_elem.text = f"{pos_x} {pos_y} {pos_x+72} {pos_y+48}"

            # Add a default barchart element.
            etree.SubElement(
                node_elem,
                "barchart",
                {"active": "true", "width": "128", "height": "128"},
            )

            # Increment positions for a simple layout.
            pos_x += 100
            pos_y += 50

    def write_xdsl(self, filename=None):
        # Convert the ElementTree to a pretty-printed XML string.
        xml_str = etree.tostring(self.root, encoding="utf-8")
        parsed = md.parseString(xml_str)
        pretty_xml_str = parsed.toprettyxml(indent="    ", encoding="UTF-8")

        # Write the XML string to the specified file.
        if filename is not None:
            with open(filename, "wb") as f:
                f.write(pretty_xml_str)

        # return pretty_xml_str
