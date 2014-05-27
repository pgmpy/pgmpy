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
            self.network = etree.parse(path).getroot()
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
        import numpy as np
        distribution = {}
        for dist in self.bnmodel.find('DISTRIBUTIONS'):
            variable_name = dist.find('PRIVATE').get('NAME')
            distribution[variable_name] = {'TYPE': dist.get('TYPE')}
            if dist.find('CONDSET') is not None:
                distribution[variable_name]['CONDSET'] = [var.get('NAME') for var in dist.find('CONDSET').findall('CONDELEM')]
            distribution[variable_name]['DPIS'] = np.array([list(map(float, dpi.text.split())) for dpi in dist.find('DPIS')])

        return distribution
