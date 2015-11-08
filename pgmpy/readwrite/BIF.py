#import numpy as np
#from pgmpy.models import BayesianModel
#from pgmpy.factors import TabularCPD, State
#from pgmpy.extern.six.moves import range
import re

# Defining some regular expressions

block_comment_regex = re.compile('/\*[^/]*/')   # A regular expression to check for block comments
line_comment_regex = re.compile(r'//[^"\n"]*')  # A regular expression to check for line comments

class BifReader(object):
    
    
    """
    Base class for reading network file in bif format
    """
    def __init__(self, path = None, string = None):
        
        """
        Initialisation of BifReader object

        Parameters
        ----------------
        path : file or str
                File of bif data
        string : str
                String of bif data
        Examples
        -----------------
        # dog-problem.bif file is present at
        # http://www.cs.cmu.edu/~javabayes/Examples/DogProblem/dog-problem.bif
        >>> reader = BifReader("bif_test.bif")
        """
        if path:

            path = open(path, 'r').read().replace('"', '')  # Opening the file and replacing qoutes by null string
            
            if '/*' or '//' in FILE:
                
                path = block_comment_regex.sub('', path)
                path = line_comment_regex.sub('', path)      # Striping comments off
            
            self.network = path

        elif string:

            string = string.replace('"', '')
            
            if '/*' or '//' in string:
                
                string = block_comment_regex.sub('', string)
                string = line_comment_regex.sub('', string) #Striping comments off
            
            self.network = string

        else:
            
            raise ValueError("Must specify either path or string")
        
        self.get_variables_info()
    
    
    def get_variables_info(self):
        
        """
        Functions gets all type of variable information
        """
        variable_block_starts = [x.end()+1 for x in re.finditer('variable', self.network)]
        variable_block = []

        for i in variable_block_starts:
            
            variable_block_end = self.network.find('}\n', i)
            variable_block.append( self.network[ i : variable_block_end ] )
        
        self.network = self.network[ variable_block_end:]
        variable_names = []
        variable_states = {}
        variable_properties = {}
        state_pattern = re.compile('[\{\};,]*')
        
        for block in variable_block:
            
            block = block.split('\n')
            name = block[0].split()[0]
            variable_names.append(name)
            block = block[1:]
            variable_properties[name] = []
            for line in block:

                if 'type' in line:
                    
                    k = line.find('{')
                    line = state_pattern.sub('', line[k:])
                    variable_states[name] = [x for x in line.split() if x!= '']

                elif 'property' in line :
                    k = line.find('property')
                    variable_properties[name].append(line[k+8:-1].strip())
        
        self.variable_names = variable_names
        self.variable_states = variable_states
        self.variable_properties = variable_properties

    def network_name(self):
        
        """
        Retruns the name of the network

        Examples
        ---------------
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.network_name()
        'Dog-Problem'
        """
    
    def get_variables(self):
        
        """
        Returns list of variables of the network

        Examples
        -------------
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.get_variables()
        ['light-on','bowel_problem','dog-out','hear-bark','family-out']
        """
        return self.variable_names

    def get_edges(self):
        
        """
        Returns the edges of the network

        Examples
        ------------
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.get_edges()
        [['family-out','light-on'],
         ['family-out','dog-out'],
         ['bowel-problem','dog-out'],
         ['dog-out','hear-bark']]
        """
    
    def get_states(self):
        
        """
        Returns the states of variables present in the network

        Examples
        -----------
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.get_states()
        {'bowel-problem': ['true','false'],
         'dog-out': ['true','false'],
         'family-out': ['true','false'],
         'hear-bark': ['true','false'],
         'light-on': ['true','false']}
        """
        
        return self.variable_states
    
    def get_property(self):
        
        """
        Returns the property of the variable

        Examples
        -------------
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.get_property()
        {{'bowel-problem': ['position = (335, 99)'],
          'dog-out': ['position = (300, 195)'],
          'family-out': ['position = (257, 99)'],
          'hear-bark': ['position = (296, 268)'],
          'light-on': ['position = (218, 195)']}
        """
        
        return self.variable_properties
    
    def get_cpd(self):
        
        """
        Returns the CPD of the variables present in the network

        Examples
        --------
        >>> reader = BIF.BifReader("bif_test.bif")
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
