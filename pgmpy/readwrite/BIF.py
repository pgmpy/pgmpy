import numpy as np
#from pgmpy.models import BayesianModel
#from pgmpy.factors import TabularCPD, State
#from pgmpy.extern.six.moves import range
import re

# Defining some regular expressions

block_comment_regex = re.compile('/\*[^/]*/')   # A regular expression to check for block comments
line_comment_regex = re.compile(r'//[^"\n"]*')  # A regular expression to check for line comments
remove_multipule_spaces = re.compile(r'[" ""\t""\r""\f"][" ""\t""\r""\f"]*')
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
            path = remove_multipule_spaces.sub(' ', path)
            if '/*' or '//' in FILE:
                
                path = block_comment_regex.sub('', path)
                path = line_comment_regex.sub('', path)      # Striping comments off
            
            self.network = path

        elif string:

            string = string.replace('"', '')
            string = remove_multipule_spaces.sub(' ', string)
            if '/*' or '//' in string:
                
                string = block_comment_regex.sub('', string)
                string = line_comment_regex.sub('', string)     #Striping comments off
            
            self.network = string

        else:
            
            raise ValueError("Must specify either path or string")
        
        self.network_name()
        self.get_variables_info()
        self.get_cpd()
    
    def network_name(self):
        
        """
        Retruns the name of the network

        Examples
        ---------------
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.network_name()
        'Dog-Problem'
        """
        start = self.network.find('network')
        end = self.network.find('{',start)
        self.network_name = self.network[start+8:end].strip()
        return self.network_name

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
        {'bowel-problem': np.array([[0.01],
                                   [0.99]]),
         'dog-out': np.array([[0.99, 0.97, 0.9, 0.3],
                              [0.01, 0.03, 0.1, 0.7]]),
         'family-out': np.array([[0.15],
                                 [0.85]]),
         'hear-bark': np.array([[0.7, 0.01],
                                [0.3, 0.99]]),
         'light-on': np.array([[0.6, 0.05],
                               [0.4, 0.95]])}
        """
        probability_starts = [x.end()+1 for x in re.finditer('probability',self.network)]
        probability_block = []
        strip_parenthesis = re.compile('[\(\)\|]*')
        strip_characters = re.compile(r'[a-zA-Z\(\);,]*')

        for i in probability_starts:
            probability_end = self.network.find('}\n',i)
            probability_block.append(self.network[i:probability_end])

        variable_cpds = {}

        for block in probability_block :
            block = block.split('\n')
            block[0] = strip_parenthesis.sub('', block[0])
            block[0] = block[0].strip()
            name = block[0].split()[0]
            block = block[1:]
            cpd = []

            for line in block:
                line = line.strip()
                line = strip_characters.sub('', line)
                cpd.extend([float(x) for x in line.split() if x!= ''])

            arr = np.array(cpd)
            arr = arr.reshape(len(self.variable_states[name])
                            ,arr.size//len(self.variable_states[name]))
            variable_cpds[name] = arr

        self.variable_cpds = variable_cpds
        return variable_cpds
