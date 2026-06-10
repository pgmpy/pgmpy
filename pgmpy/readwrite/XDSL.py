import random
import warnings
import xml.dom.minidom as md
import xml.etree.ElementTree as etree
from itertools import chain

import networkx as nx

from pgmpy import logger
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import compat_fns


class XDSLReader:
    """
    Initializes the reader object for XDSL file formats[1] created through GeNIe[2].
    Note that XDSLReader only supports cpt blocks from the XDSL file format; elements like
    'deterministic' need to be aapropriately converted into 'cpt' elements before usage.

    Parameters
    ----------
    path : file or str
        Path to the XDSL file.

    string : str
        A string containing the XDSL file content.

    Examples
    --------
    >>> from pgmpy.readwrite import XDSLReader, XDSLWriter
    >>> from pgmpy.example_models import load_model
    >>> asia = load_model("bnlearn/asia")
    >>> XDSLWriter(asia).write("asia_test.xdsl")
    >>> reader = XDSLReader("asia_test.xdsl")
    >>> model = reader.get_model()

    References
    ----------
    - :cite:p:`bayesfusion_xdsl`
    - :cite:p:`bayesfusion_genie`
    """

    def __init__(self, path=None, string=None):
        if path:
            self.network = etree.ElementTree(file=path).getroot()
        elif string:
            self.network = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")
        self.network_name = self.network.attrib["id"]
        self.cpt_elements = self.network.find("nodes").findall("cpt")
        self.variables = self.get_variables()
        self.variable_parents = self.get_parents()
        self.edge_list = self.get_edges()
        self.variable_states = self.get_states()
        self.variable_CPD = self.get_values()

    def get_variables(self):
        """
        Returns list of variables of the network

        Examples
        --------
        >>> from pgmpy.readwrite import XDSLReader, XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> XDSLWriter(load_model("bnlearn/asia")).write("asia_test.xdsl")
        >>> reader = XDSLReader("asia_test.xdsl")
        >>> reader.get_variables()
        ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
        """
        variables = [variable.attrib["id"] for variable in self.cpt_elements]
        for var in variables:
            if isinstance(var, str) and (" " in var):
                raise ValueError(
                    f"XDSLReader does not support models with node names"
                    f" that contain whitespaces. Failed to process node: {var}"
                )

        return variables

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Examples
        --------
        >>> from pgmpy.readwrite import XDSLReader, XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> XDSLWriter(load_model("bnlearn/asia")).write("asia_test.xdsl")
        >>> reader = XDSLReader("asia_test.xdsl")
        >>> reader.get_parents() # doctest: +NORMALIZE_WHITESPACE
        {'asia': [], 'tub': ['asia'], 'smoke': [], 'lung': ['smoke'], 'bronc': ['smoke'],
         'either': ['lung', 'tub'], 'xray': ['either'], 'dysp': ['bronc', 'either']}
        """
        variable_parents = {}
        for node in self.cpt_elements:
            parents = node.find("parents")
            if parents is not None:
                variable_parents[node.attrib["id"]] = parents.text.split(" ")
            else:
                variable_parents[node.attrib["id"]] = []

        return variable_parents

    def get_edges(self):
        """
        Returns the edges of the network

        Examples
        --------
        >>> from pgmpy.readwrite import XDSLReader, XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> XDSLWriter(load_model("bnlearn/asia")).write("asia_test.xdsl")
        >>> reader = XDSLReader("asia_test.xdsl")
        >>> reader.get_edges() # doctest: +NORMALIZE_WHITESPACE
        [['asia', 'tub'], ['smoke', 'lung'], ['smoke', 'bronc'], ['lung', 'either'],
         ['tub', 'either'], ['either', 'xray'], ['bronc', 'dysp'], ['either', 'dysp']]
        """
        edge_list = [[value, key] for key in self.variable_parents for value in self.variable_parents[key]]
        return edge_list

    def get_states(self):
        """
        Returns the states of variables present in the network

        Examples
        --------
        >>> from pgmpy.readwrite import XDSLReader, XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> XDSLWriter(load_model("bnlearn/asia")).write("asia_test.xdsl")
        >>> reader = XDSLReader("asia_test.xdsl")
        >>> reader.get_states() # doctest: +NORMALIZE_WHITESPACE
        {'asia': ['yes', 'no'], 'tub': ['yes', 'no'], 'smoke': ['yes', 'no'], 'lung': ['yes', 'no'],
         'bronc': ['yes', 'no'], 'either': ['yes', 'no'], 'xray': ['yes', 'no'], 'dysp': ['yes', 'no']}
        """
        variable_states = {}
        for cpt in self.cpt_elements:
            variable_states[cpt.attrib["id"]] = [state.attrib["id"] for state in cpt.findall("state")]
        return variable_states

    def get_values(self):
        """
        Returns the CPD of the variables present in the network

        Examples
        --------
        >>> from pgmpy.readwrite import XDSLReader, XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> XDSLWriter(load_model("bnlearn/asia")).write("asia_test.xdsl")
        >>> reader = XDSLReader("asia_test.xdsl")
        >>> reader.get_values() # doctest: +NORMALIZE_WHITESPACE
        {'asia': [[0.01], [0.99]], 'tub': [[0.05, 0.01], [0.95, 0.99]], 'smoke': [[0.5], [0.5]],
         'lung': [[0.1, 0.01], [0.9, 0.99]], 'bronc': [[0.6, 0.3], [0.4, 0.7]],
         'either': [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'xray': [[0.98, 0.05], [0.02, 0.95]],
         'dysp': [[0.9, 0.8, 0.7, 0.1], [0.1, 0.2, 0.3, 0.9]]}
        """
        variable_CPD = {}
        for cpt in self.cpt_elements:
            combined_prob = cpt.find("probabilities")
            num_states = len([state for state in cpt.findall("state")])
            cpd_arr = [[] for k in range(num_states)]
            prob_values = combined_prob.text.split(" ")

            for j in range(num_states):
                for i in range(j, len(prob_values), num_states):
                    cpd_arr[j].append(float(prob_values[i]))

            variable_CPD[cpt.attrib["id"]] = cpd_arr
        return variable_CPD

    def get_model(self, state_name_type=str):
        """
        Returns a Bayesian Network instance from the file/string.

        Parameters
        ----------
        state_name_type: int, str, or bool (default: str)
            The data type to which to convert the state names of the variables.

        Returns
        -------
        DiscreteBayesianNetwork instance: The read model.

        Examples
        --------
        >>> from pgmpy.readwrite import XDSLReader, XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> XDSLWriter(load_model("bnlearn/asia")).write("asia_test.xdsl")
        >>> reader = XDSLReader("asia_test.xdsl")
        >>> model = reader.get_model()
        """
        model = DiscreteBayesianNetwork()
        model.add_nodes_from(self.variables)
        model.add_edges_from(self.edge_list)
        model.name = self.network_name

        tabular_cpds = []
        for var, values in self.variable_CPD.items():
            evidence_card = [len(self.variable_states[evidence_var]) for evidence_var in self.variable_parents[var]]
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


class XDSLWriter:
    """
    Initialise a XDSL writer object to export pgmpy models to XDSL file format[1] used by GeNIe[2].

    Parameters
    ----------
    model: pgmpy.models.DiscreteBayesianNetwork instance.
        The model to write to the file.

    network_id: str (default: "MyNetwork")
        Name/id of the network

    num_samples: int (default: 0)
        Number of samples used for continuous variables

    disc_samples: int (default: 0)
        Number of samples used for discrete variables

    encoding: str (optional, default='utf-8')
        Encoding for text data

    Examples
    ---------
    >>> from pgmpy.readwrite import XDSLWriter
    >>> from pgmpy.example_models import load_model
    >>> asia = load_model("bnlearn/asia")
    >>> writer = XDSLWriter(asia)
    >>> writer.write("asia.xdsl")

    References
    ----------
    - :cite:p:`bayesfusion_xdsl`
    - :cite:p:`bayesfusion_genie`
    """

    def __init__(
        self,
        model,
        network_id="MyNetwork",
        num_samples="0",
        disc_samples="0",
        encoding="utf-8",
    ):
        if not isinstance(model, DiscreteBayesianNetwork):
            raise TypeError("model must an instance of DiscreteBayesianNetwork")
        self.model = model
        self.encoding = encoding
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
        self.cpds = self.get_cpds()
        self._create_extensions()

    def get_variables(self):
        """
        Add variables and their XML elements/representation to XDSL

        Return
        ------
        dict: dict of type {variable: variable tags}

        Examples
        --------
        >>> from pgmpy.readwrite import XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> writer = XDSLWriter(load_model("bnlearn/asia"))
        >>> sorted(writer.get_variables().keys())
        ['asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray']
        """
        variable_tag = {}
        nodes_elem = etree.SubElement(self.root, "nodes")

        for var in self.model.nodes:
            if isinstance(var, str) and " " in var:
                logger.warning(f" Node '{var}' contains whitespaces. This can create issues when loading the model. ")
            variable_tag[var] = etree.SubElement(nodes_elem, "cpt", {"id": var})

        return variable_tag

    def get_cpds(self):
        """
        Add the complete CPT element (with states and probabilities) to XDSL.

        Return
        ---------------
        dict: dict of type {variable: table tag}

        Examples
        -------
        >>> from pgmpy.readwrite import XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> writer = XDSLWriter(load_model("bnlearn/asia"))
        >>> sorted(writer.get_cpds().keys())
        ['asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray']
        """
        outcome_tag = {}
        cpds = self.model.get_cpds()
        cpd_vars = [cpd.variable for cpd in cpds]
        for var in self.model.nodes:
            idx = cpd_vars.index(var)
            cpd = cpds[idx]

            cpt_elem = self.variables[var]
            states = cpd.state_names[cpd.variable]

            # Check for commas in state names and warn if found
            for st in states:
                st_str = str(st)
                if "," in st_str:
                    logger.warning(
                        f"State name '{st_str}' for variable '{var}' contains commas. "
                        "This may cause issues when loading the file. Consider removing any special characters."
                    )
                etree.SubElement(cpt_elem, "state", {"id": st_str})

            evidence = cpd.variables
            if len(evidence) > 1:
                parents_str = " ".join(evidence[1:])
                parents_elem = etree.SubElement(cpt_elem, "parents")
                parents_elem.text = parents_str

            # Add the <probabilities> element.
            probs_elem = etree.SubElement(cpt_elem, "probabilities")
            values = cpd.get_values()

            # Flatten in column-major order so that for each parent
            #  configuration the probabilities for all states are listed.
            flat_values = compat_fns.ravel_f(values)
            probs_elem.text = " ".join(f"{float(x):.16f}" for x in flat_values)

            outcome_tag[var] = cpd

        return outcome_tag

    def _create_extensions(self):
        """
        Create the <extensions> block with a minimal <genie> element for layout information.

        Parameters
        ----------

        """
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

        for node in list(nx.topological_sort(self.model)):
            node_elem = etree.SubElement(genie_elem, "node", {"id": node})

            name_elem = etree.SubElement(node_elem, "name")
            name_elem.text = node

            # Appearance details (colors, font).
            etree.SubElement(node_elem, "interior", {"color": "e5f6f7"})
            etree.SubElement(node_elem, "outline", {"color": "000080"})
            etree.SubElement(node_elem, "font", {"color": "000000", "name": "Arial", "size": "8"})

            # Set node position (x1, y1, x2, y2).
            # Provide random position to each node.
            pos_x, pos_y = random.randint(0, 100), random.randint(0, 100)
            pos_elem = etree.SubElement(node_elem, "position")
            pos_elem.text = f"{pos_x} {pos_y} {pos_x + 72} {pos_y + 48}"

            etree.SubElement(
                node_elem,
                "barchart",
                {"active": "true", "width": "128", "height": "128"},
            )

    def write(self, filename=None):
        """
        Write the xdsl data into the file.

        Parameters
        ----------
        filename: Name (path) of the file.

        Examples
        --------
        >>> from pgmpy.readwrite import XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XDSLWriter(model)
        >>> writer.write("asia.xdsl")
        """
        xml_str = etree.tostring(self.root, encoding=self.encoding)
        parsed = md.parseString(xml_str)
        pretty_xml_str = parsed.toprettyxml(indent="    ", encoding=self.encoding)

        if filename is not None:
            with open(filename, "wb") as f:
                f.write(pretty_xml_str)

    def write_xdsl(self, filename):
        warnings.warn(
            """`XDSLWriter.write_xdsl` is deprecated and will be removed in v1.3.0. Please use `XDSLWriter.write`
            instead.""",
            FutureWarning,
            stacklevel=2,
        )
        self.write(filename)
