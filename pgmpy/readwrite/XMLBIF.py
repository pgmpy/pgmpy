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

__all__ = ['XMLBIFReader']


class XMLBIFReader:
    """
    Base class for reading network file in XMLBIF format.
    """
    def __init__(self, path=None, string=None):
        """
        Initialisation of XMLBIFReader object.

        Parameters
        ----------
        path : file or str
            File of XMLBIF data
        string : str
            String of XMLBIF data

        Example
        -------
        >>> reader = XMLBIFReader("sample.xml")
        """
        if path:
            self.tree = etree.ElementTree(file=path)
        elif string:
            self.tree = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")
        self.variables = None
        self.edge_list = None
        self.variable_states = None
        self.variable_parents = None
        self.variable_CPD = None

    def get_variables(self):
        """
        Returns list of variables of the network

        Example
        -------
        # xmlbif_test.xml is the file present in
        # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_variables()
        ['light-on', 'bowel-problem', 'dog-out', 'hear-bark', 'family-out']
        """
        root = self.tree.getroot()
        network = root.find('NETWORK')
        self.variables = [variable.find('NAME').text
                          for variable in network.findall('VARIABLE')]
        return self.variables

    def get_edges(self):
        """
        Returns the edges of the network

        Example
        -------
        # xmlbif_test.xml is the file present in
        # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_edges()
        [['family-out', 'light-on'],
         ['family-out', 'dog-out'],
         ['bowel-problem', 'dog-out'],
         ['dog-out', 'hear-bark']]
        """
        if self.variable_parents is None:
            root = self.tree.getroot()
            network = root.find('NETWORK')
            self.variable_parents = {definition.find('FOR').text:
                                         [edge.text for edge in definition.findall('GIVEN')][::-1]
                                     for definition in network.findall('DEFINITION')}
        self.edge_list = [[value, key] for key in self.variable_parents
                          for value in self.variable_parents[key]]
        return self.edge_list

    def get_states(self):
        """
        Returns the states of variables present in the network

        Example
        -------
        # xmlbif_test.xml is the file present in
        # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_states()
        {'bowel-problem': ['true', 'false'],
         'dog-out': ['true', 'false'],
         'family-out': ['true', 'false'],
         'hear-bark': ['true', 'false'],
         'light-on': ['true', 'false']}
        """
        root = self.tree.getroot()
        network = root.find('NETWORK')
        self.variable_states = {variable.find('NAME').text:
                                    [outcome.text for outcome in variable.findall('OUTCOME')]
                                for variable in network.findall('VARIABLE')}
        return self.variable_states

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Example
        -------
        # xmlbif_test.xml is the file present in
        # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_parents()
        {'bowel-problem': [],
         'dog-out': ['family-out', 'bowel-problem'],
         'family-out': [],
         'hear-bark': ['dog-out'],
         'light-on': ['family-out']}
        """
        if self.variable_parents is None:
            root = self.tree.getroot()
            network = root.find('NETWORK')
            self.variable_parents = {definition.find('FOR').text:
                                         [edge.text for edge in definition.findall('GIVEN')][::-1]
                                     for definition in network.findall('DEFINITION')}
        return self.variable_parents

    def get_cpd(self):
        """
        Returns the CPD of the variables present in the network

        Example
        -------
        # xmlbif_test.xml is the file present in
        # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_cpd()
        {'bowel-problem': array([[ 0.01],
                                 [ 0.99]]),
         'dog-out': array([[ 0.99,  0.01,  0.97,  0.03],
                           [ 0.9 ,  0.1 ,  0.3 ,  0.7 ]]),
         'family-out': array([[ 0.15],
                              [ 0.85]]),
         'hear-bark': array([[ 0.7 ,  0.3 ],
                             [ 0.01,  0.99]]),
         'light-on': array([[ 0.6 ,  0.4 ],
                            [ 0.05,  0.95]])}
        """
        root = self.tree.getroot()
        network = root.find('NETWORK')
        self.variable_CPD = {definition.find('FOR').text: list(map(float, table.text.split()))
                             for definition in network.findall('DEFINITION')
                             for table in definition.findall('TABLE')}
        if self.variable_states is None:
            self.variable_states = {variable.find('NAME').text:
                                        [outcome.text for outcome in variable.findall('OUTCOME')]
                                    for variable in network.findall('VARIABLE')}
        for variable in self.variable_CPD:
            arr = np.array(self.variable_CPD[variable])
            arr = arr.reshape((len(self.variable_states[variable]),
                               arr.size//len(self.variable_states[variable])))
            self.variable_CPD[variable] = arr
        return self.variable_CPD
