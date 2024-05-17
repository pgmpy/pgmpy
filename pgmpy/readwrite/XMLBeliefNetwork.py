import itertools
import xml.etree.ElementTree as etree

import networkx as nx
import numpy as np

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


class XBNReader(object):
    """
    Initializer for XBNReader class.

    Parameters
    ----------
    path: str or file
        Path of the file containing XBN data.

    string: str
        String of XBN data

    Examples
    --------
    >>> reader = XBNReader('test_XBN.xml')

    Reference
    ---------
    [1] Microsoft Research. XML belief network file format.
        http://xml.coverpages.org/xbn-MSdefault19990414.html, 1999.
    """

    def __init__(self, path=None, string=None):
        if path:
            self.network = etree.parse(path).getroot()
        elif string:
            self.network = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")

        self.bnmodel = self.network.find("BNMODEL")
        self.analysisnotebook = self.get_analysisnotebook_values()
        self.model_name = self.get_bnmodel_name()
        self.static_properties = self.get_static_properties()
        self.variables = self.get_variables()
        self.edges = self.get_edges()
        self.variable_CPD = self.get_distributions()

    def get_analysisnotebook_values(self):
        """
        Returns a dictionary of the attributes of ANALYSISNOTEBOOK tag

        Examples
        --------
        >>> reader = XBNReader('xbn_test.xml')
        >>> reader.get_analysisnotebook_values()
        {'NAME': "Notebook.Cancer Example From Neapolitan",
         'ROOT': "Cancer"}
        """
        return {key: value for key, value in self.network.items()}

    def get_bnmodel_name(self):
        """
        Returns the name of the BNMODEL.

        Examples
        --------
        >>> reader = XBNReader('xbn_test.xml')
        >>> reader.get_bnmodel_name()
        'Cancer'
        """
        return self.network.find("BNMODEL").get("NAME")

    def get_static_properties(self):
        """
        Returns a dictionary of STATICPROPERTIES

        Examples
        --------
        >>> reader = XBNReader('xbn_test.xml')
        >>> reader.get_static_properties()
        {'FORMAT': 'MSR DTAS XML', 'VERSION': '0.2', 'CREATOR': 'Microsoft Research DTAS'}
        """
        if self.bnmodel.find("STATICPROPERTIES") is not None:
            return {
                tags.tag: tags.get("VALUE")
                for tags in self.bnmodel.find("STATICPROPERTIES")
            }
        else:
            return {}

    def get_variables(self):
        """
        Returns a list of variables.

        Examples
        --------
        >>> reader = XBNReader('xbn_test.xml')
        >>> reader.get_variables()
        {'a': {'TYPE': 'discrete', 'XPOS': '13495',
               'YPOS': '10465', 'DESCRIPTION': '(a) Metastatic Cancer',
               'STATES': ['Present', 'Absent']}
        'b': {'TYPE': 'discrete', 'XPOS': '11290',
               'YPOS': '11965', 'DESCRIPTION': '(b) Serum Calcium Increase',
               'STATES': ['Present', 'Absent']},
        'c': {....},
        'd': {....},
        'e': {....}
        }
        """
        variables = {}
        for variable in self.bnmodel.find("VARIABLES"):
            variables[variable.get("NAME")] = {
                "TYPE": variable.get("TYPE"),
                "XPOS": variable.get("XPOS"),
                "YPOS": variable.get("YPOS"),
                "DESCRIPTION": variable.find("DESCRIPTION").text,
                "STATES": [state.text for state in variable.findall("STATENAME")],
            }
        return variables

    def get_edges(self):
        """
        Returns a list of tuples. Each tuple contains two elements (parent, child) for each edge.

        Examples
        --------
        >>> reader = XBNReader('xbn_test.xml')
        >>> reader.get_edges()
        [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'), ('c', 'e')]
        """
        return [
            (arc.get("PARENT"), arc.get("CHILD"))
            for arc in self.bnmodel.find("STRUCTURE")
        ]

    def get_distributions(self):
        """
        Returns a dictionary of name and its distribution. Distribution is a ndarray.

        The ndarray is stored in the standard way such that the rightmost variable
        changes most often. Consider a CPD of variable 'd' which has parents 'b' and
        'c' (distribution['CONDSET'] = ['b', 'c'])

                  |  d_0     d_1
        ---------------------------
        b_0, c_0  |  0.8     0.2
        b_0, c_1  |  0.9     0.1
        b_1, c_0  |  0.7     0.3
        b_1, c_1  |  0.05    0.95

        The value of distribution['d']['DPIS'] for the above example will be:
        array([[ 0.8 ,  0.2 ], [ 0.9 ,  0.1 ], [ 0.7 ,  0.3 ], [ 0.05,  0.95]])

        Examples
        --------
        >>> reader = XBNReader('xbn_test.xml')
        >>> reader.get_distributions()
        {'a': {'TYPE': 'discrete', 'DPIS': array([[ 0.2,  0.8]])},
         'e': {'TYPE': 'discrete', 'DPIS': array([[ 0.8,  0.2],
                 [ 0.6,  0.4]]), 'CONDSET': ['c'], 'CARDINALITY': [2]},
         'b': {'TYPE': 'discrete', 'DPIS': array([[ 0.8,  0.2],
                 [ 0.2,  0.8]]), 'CONDSET': ['a'], 'CARDINALITY': [2]},
         'c': {'TYPE': 'discrete', 'DPIS': array([[ 0.2 ,  0.8 ],
                 [ 0.05,  0.95]]), 'CONDSET': ['a'], 'CARDINALITY': [2]},
         'd': {'TYPE': 'discrete', 'DPIS': array([[ 0.8 ,  0.2 ],
                 [ 0.9 ,  0.1 ],
                 [ 0.7 ,  0.3 ],
                 [ 0.05,  0.95]]), 'CONDSET': ['b', 'c']}, 'CARDINALITY': [2, 2]}
        """
        distribution = {}
        for dist in self.bnmodel.find("DISTRIBUTIONS"):
            variable_name = dist.find("PRIVATE").get("NAME")
            distribution[variable_name] = {"TYPE": dist.get("TYPE")}
            if dist.find("CONDSET") is not None:
                distribution[variable_name]["CONDSET"] = [
                    var.get("NAME") for var in dist.find("CONDSET").findall("CONDELEM")
                ]
                distribution[variable_name]["CARDINALITY"] = np.array(
                    [
                        len(
                            set(
                                np.array(
                                    [
                                        list(map(int, dpi.get("INDEXES").split()))
                                        for dpi in dist.find("DPIS")
                                    ]
                                )[:, i]
                            )
                        )
                        for i in range(len(distribution[variable_name]["CONDSET"]))
                    ]
                )
            distribution[variable_name]["DPIS"] = np.array(
                [list(map(float, dpi.text.split())) for dpi in dist.find("DPIS")]
            ).transpose()

        return distribution

    def get_model(self):
        """
        Returns an instance of Bayesian Model.
        """
        model = BayesianNetwork()
        model.add_nodes_from(self.variables)
        model.add_edges_from(self.edges)
        model.name = self.model_name

        tabular_cpds = []
        for var, values in self.variable_CPD.items():
            evidence = values["CONDSET"] if "CONDSET" in values else []
            cpd = values["DPIS"]
            evidence_card = values["CARDINALITY"] if "CARDINALITY" in values else []
            states = self.variables[var]["STATES"]
            cpd = TabularCPD(
                var, len(states), cpd, evidence=evidence, evidence_card=evidence_card
            )
            tabular_cpds.append(cpd)

        model.add_cpds(*tabular_cpds)
        for var, properties in self.variables.items():
            model._node[var] = properties

        return model


class XBNWriter(object):
    """
    Initializer for XBNWriter class

    Parameters
    ----------
    model: BayesianNetwork Instance
        Model to write
    encoding: str(optional)
        Encoding for test data
    prettyprint: Bool(optional)
        Indentation in output XML if true

    Reference
    ---------
    http://xml.coverpages.org/xbn-MSdefault19990414.html

    Examples
    --------
    >>> writer = XBNWriter(model)
    """

    def __init__(self, model, encoding="utf-8", prettyprint=True):
        if not isinstance(model, BayesianNetwork):
            raise TypeError("Model must be an instance of Bayesian Model.")
        self.model = model

        self.encoding = encoding
        self.prettyprint = prettyprint

        self.network = etree.Element("ANALYSISNOTEBOOK")
        self.bnmodel = etree.SubElement(self.network, "BNMODEL")
        if self.model.name:
            etree.SubElement(self.bnmodel, "NAME").text = self.model.name

        self.variables = self.set_variables(self.model.nodes)
        self.structure = self.set_edges(sorted(self.model.edges()))
        self.distribution = self.set_distributions()

    def __str__(self):
        """
        Return the XML as string.
        """
        if self.prettyprint:
            self.indent(self.network)
        return etree.tostring(self.network, encoding=self.encoding)

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

    def set_analysisnotebook(self, **data):
        """
        Set attributes for ANALYSISNOTEBOOK tag

        Parameters
        ----------
        **data: dict
            {name: value} for the attributes to be set.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_analysisnotebook(NAME="Notebook.Cancer Example From Neapolitan",
        ...                             ROOT='Cancer')
        """
        for key, value in data.items():
            self.network.set(str(key), str(value))

    def set_bnmodel_name(self, name):
        """
        Set the name of the BNMODEL.

        Parameters
        ----------
        name: str
            Name of the BNModel.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_bnmodel_name("Cancer")
        """
        self.bnmodel.set("NAME", str(name))

    def set_static_properties(self, **data):
        """
        Set STATICPROPERTIES tag for the network

        Parameters
        ----------
        **data: dict
            {name: value} for name and value of the property.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_static_properties(FORMAT="MSR DTAS XML", VERSION="0.2", CREATOR="Microsoft Research DTAS")
        """
        static_prop = etree.SubElement(self.bnmodel, "STATICPROPERTIES")
        for key, value in data.items():
            etree.SubElement(static_prop, key, attrib={"VALUE": value})

    def set_variables(self, data):
        """
        Set variables for the network.

        Parameters
        ----------
        data: dict
            dict for variable in the form of example as shown.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_variables({'a': {'TYPE': 'discrete', 'XPOS': '13495',
        ...                             'YPOS': '10465', 'DESCRIPTION': '(a) Metastatic Cancer',
        ...                             'STATES': ['Present', 'Absent']},
        ...                       'b': {'TYPE': 'discrete', 'XPOS': '11290',
        ...                             'YPOS': '11965', 'DESCRIPTION': '(b) Serum Calcium Increase',
        ...                             'STATES': ['Present', 'Absent']}})
        """
        variables = etree.SubElement(self.bnmodel, "VARIABLES")
        for var in sorted(data):
            variable = etree.SubElement(
                variables,
                "VAR",
                attrib={
                    "NAME": var,
                    "TYPE": data[var].get("TYPE", ""),
                    "XPOS": data[var].get("XPOS", ""),
                    "YPOS": data[var].get("YPOS", ""),
                },
            )
            etree.SubElement(
                variable,
                "DESCRIPTION",
                attrib={"DESCRIPTION": data[var].get("DESCRIPTION", "")},
            )
            for state in self.model.states[var]:
                etree.SubElement(variable, "STATENAME").text = state

    def set_edges(self, edge_list):
        """
        Set edges/arc in the network.

        Parameters
        ----------
        edge_list: array_like
            list, tuple, dict or set whose each element has two values (parent, child).

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_edges([('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'), ('c', 'e')])
        """
        structure = etree.SubElement(self.bnmodel, "STRUCTURE")
        for edge in edge_list:
            etree.SubElement(
                structure, "ARC", attrib={"PARENT": edge[0], "CHILD": edge[1]}
            )

    def set_distributions(self):
        """
        Set distributions in the network.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer =XBNWriter()
        >>> writer.set_distributions()
        """
        distributions = etree.SubElement(self.bnmodel, "DISTRIBUTIONS")

        cpds = self.model.get_cpds()
        cpds.sort(key=lambda x: x.variable)
        for cpd in cpds:
            cpd_values = cpd.get_values().transpose()
            var = cpd.variable
            dist = etree.SubElement(
                distributions,
                "DIST",
                attrib={"TYPE": self.model.nodes[var].get("TYPE", "")},
            )
            etree.SubElement(dist, "PRIVATE", attrib={"NAME": var})
            dpis = etree.SubElement(dist, "DPIS")
            evidence = cpd.variables[1:]
            evidence_card = cpd.cardinality[1:]
            if evidence:
                condset = etree.SubElement(dist, "CONDSET")
                for condelem in evidence:
                    etree.SubElement(condset, "CONDELEM", attrib={"NAME": condelem})
                indexes_iter = itertools.product(
                    *[range(card) for card in evidence_card]
                )
                for val in range(cpd_values.shape[0]):
                    index_value = " " + " ".join(map(str, next(indexes_iter))) + " "
                    etree.SubElement(
                        dpis, "DPI", attrib={"INDEXES": index_value}
                    ).text = (" " + " ".join(map(str, cpd_values[val])) + " ")
            else:
                etree.SubElement(dpis, "DPI").text = (
                    " " + " ".join(map(str, cpd_values[0])) + " "
                )

    def write_xbn(self, filename):
        """
        Writes the BIF data into a file

        Parameters
        ----------
        filename : Name of the file

        Example
        -------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.readwrite import XBNReader, XBNWriter
        >>> asia = get_example_model('asia')
        >>> writer = XBNWriter(asia)
        >>> writer.write_xbn(filename='asia.xbn')
        """
        writer = self.__str__()
        with open(filename, "wb") as fout:
            fout.write(writer)
