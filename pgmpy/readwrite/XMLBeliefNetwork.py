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

__all__ = ['XBNReader', 'XBNWriter']


class XBNReader:
    """
    Base class for reading XML Belief Network File Format.
    """
    def __init__(self, path=None, string=None):
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
        reader = XBNReader('test_XBN.xml')

        Reference
        ---------
        http://xml.coverpages.org/xbn-MSdefault19990414.html
        """
        if path:
            self.network = etree.ElementTree(file=path).getroot()
        elif string:
            self.network = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")

        self.bnmodel = self.network.find('BNMODEL')

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
        return self.network.find('BNMODEL').get('NAME')

    def get_static_properties(self):
        """
        Returns a dictionary of STATICPROPERTIES

        Examples
        --------
        >>> reader = XBNReader('xbn_test.xml')
        >>> reader.get_static_properties()
        {'FORMAT': 'MSR DTAS XML', 'VERSION': '0.2', 'CREATOR': 'Microsoft Research DTAS'}
        """
        return {tags.tag: tags.get('VALUE') for tags in self.bnmodel.find('STATICPROPERTIES')}

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
        for variable in self.bnmodel.find('VARIABLES'):
            variables[variable.get('NAME')] = {'TYPE': variable.get('TYPE'),
                                               'XPOS': variable.get('XPOS'),
                                               'YPOS': variable.get('YPOS'),
                                               'DESCRIPTION': variable.find('DESCRIPTION').text,
                                               'STATES': [state.text for state in variable.findall('STATENAME')]}
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
        return [(arc.get('PARENT'), arc.get('CHILD')) for arc in self.bnmodel.find('STRUCTURE')]

    def get_distributions(self):
        """
        Returns a dictionary of name and its distribution. Distribution is a ndarray.

        The ndarray is stored in the standard way such that the rightmost variable changes most often.
        Consider a CPD of variable 'd' which has parents 'b' and 'c' (distribution['CONDSET'] = ['b', 'c'])

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
                 [ 0.6,  0.4]]), 'CONDSET': ['c']},
         'b': {'TYPE': 'discrete', 'DPIS': array([[ 0.8,  0.2],
                 [ 0.2,  0.8]]), 'CONDSET': ['a']},
         'c': {'TYPE': 'discrete', 'DPIS': array([[ 0.2 ,  0.8 ],
                 [ 0.05,  0.95]]), 'CONDSET': ['a']},
         'd': {'TYPE': 'discrete', 'DPIS': array([[ 0.8 ,  0.2 ],
                 [ 0.9 ,  0.1 ],
                 [ 0.7 ,  0.3 ],
                 [ 0.05,  0.95]]), 'CONDSET': ['b', 'c']}}
        """
        #TODO: add parsing of DPI INDEXEX.
        import numpy as np
        distribution = {}
        for dist in self.bnmodel.find('DISTRIBUTIONS'):
            variable_name = dist.find('PRIVATE').get('NAME')
            distribution[variable_name] = {'TYPE': dist.get('TYPE')}
            if dist.find('CONDSET') is not None:
                distribution[variable_name]['CONDSET'] = [var.get('NAME') for var in dist.find('CONDSET').findall('CONDELEM')]
            distribution[variable_name]['DPIS'] = np.array([list(map(float, dpi.text.split())) for dpi in dist.find('DPIS')])

        return distribution


class XBNWriter:
    """
    Base class for writing XML Belief Network file format.
    """
    def __init__(self):
        """
        Initializer for XBNWriter class
        """
        self.network = etree.Element('ANALYSISNOTEBOOK')
        self.bnmodel = etree.SubElement(self.network, 'BNMODEL')

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
        >>>                             ROOT='Cancer')
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
        self.bnmodel.set('NAME', str(name))

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
        static_prop = etree.SubElement(self.bnmodel, 'STATICPROPERTIES')
        for key, value in data.items():
            etree.SubElement(static_prop, key, attrib={'VALUE': value})

    def set_variables(self, data):
        """
        Set variables for the network.

        Parameters
        ----------
        data: dict
            data is dict in the form:
                {'a': {'TYPE': 'discrete', 'XPOS': '13495',
                       'YPOS': '10465', 'DESCRIPTION': '(a) Metastatic Cancer',
                       'STATES': ['Present', 'Absent']}
                'b': {'TYPE': 'discrete', 'XPOS': '11290',
                       'YPOS': '11965', 'DESCRIPTION': '(b) Serum Calcium Increase',
                       'STATES': ['Present', 'Absent']}, ...

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_variables({'a': {'TYPE': 'discrete', 'XPOS': '13495',
        >>>                             'YPOS': '10465', 'DESCRIPTION': '(a) Metastatic Cancer',
        >>>                             'STATES': ['Present', 'Absent']}
        >>>                       'b': {'TYPE': 'discrete', 'XPOS': '11290',
        >>>                             'YPOS': '11965', 'DESCRIPTION': '(b) Serum Calcium Increase',
        >>>                             'STATES': ['Present', 'Absent']}})
        """
        variables = etree.SubElement(self.bnmodel)
        for var in data:
            variable = etree.SubElement(variables, 'VAR', attrib={'NAME': var, 'TYPE': data[var]['TYPE'],
                                                                  'XPOS': data[var]['XPOS'], 'YPOS': data[var]['YPOS']})
            etree.SubElement(variable, 'DESCRIPTION', data[var]['DESCRIPTION'])
            for state in data[var]['STATES']:
                etree.SubElement(variable, 'STATENAME').text = state

    def set_edges(self, edge_list):
        """
        Set edges/arc in the network.

        Parameters
        ----------
        edge_list: array_like
            list, tuple, dict or set whose each elements has two values (parent, child).

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_edges([('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'), ('c', 'e')])
        """
        structure = etree.SubElement(self.bnmodel, 'STRUCTURE')
        for edge in edge_list:
            etree.SubElement(structure, 'ARC', attrib={'PARENT': edge[0], 'CHILD': edge[1]})

    def set_distributions(self, data):
        """
        Set distributions in the network.

        Parameters
        ----------
        data: dict
            dict in the form of:
                {'a': {'TYPE': 'discrete', 'DPIS': array([[ 0.2,  0.8]])},
                 'e': {'TYPE': 'discrete', 'DPIS': array([[ 0.8,  0.2],
                                                          [ 0.6,  0.4]]),
                       'CONDSET': ['c']}, ... }

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer =XBNWriter()
        >>> writer.set_distributions({'a': {'TYPE': 'discrete', 'DPIS': array([[ 0.2,  0.8]])},
        >>>                           'e': {'TYPE': 'discrete', 'DPIS': array([[ 0.8,  0.2],
        >>>                                                                    [ 0.6,  0.4]]),
        >>>                                 'CONDSET': ['c']}})
        """
        distributions = etree.SubElement(self.bnmodel, 'DISTRIBUTIONS')
        for var in data:
            dist = etree.SubElement(distributions, 'DIST', attrib=data[var]['TYPE'])
            etree.SubElement(dist, 'PRIVATE', attrib={'NAME': var})
            if 'CONDSET' in data:
                condset = etree.SubElement(dist, 'CONDSET')
                for cond in data[var]['CONDSET']:
                    etree.SubElement(condset, 'CONDELEM', attrib={'NAME': cond})
            #TODO: Add DPIS.