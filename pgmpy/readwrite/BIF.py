import numpy as np
from pyparsing import Word,alphanums,Suppress,Optional,CharsNotIn,Group,nums,alphas,ZeroOrMore
# from pgmpy.models import BayesianModel
# from pgmpy.factors import TabularCPD, State
# from pgmpy.extern.six.moves import range
import re

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
        # Defining some regular expressions

        cpp_comment_regex = re.compile('/\*[^/]*/')                                     # A regular expression to check for block comments
        line_comment_regex = re.compile(r'//[^"\n"]*')                                  # A regular expression to check for line comments
        remove_multipule_spaces = re.compile(r'[" ""\t""\r""\f"][" ""\t""\r""\f"]*')    # A regular expression to check for multiple spaces

        if path:
            path = open(path, 'r').read()                                               

            if '"' in path:
                path = path.replace('"', ' ')
                """ 
                Replacing quotes by spaces to remove case sensitivity like:
                "Dog-problem" and Dog-problem
                or "true""false" and "true" "false" and true false
                """
            path = remove_multipule_spaces.sub(' ', path)                               # replacing multiple spaces or tabs by one space
            if '/*' or '//' in path:
                path = cpp_comment_regex.sub('', path)
                path = line_comment_regex.sub('', path)                                 # Striping comments off both type of comments

            self.network = path
        elif string:
            if '"' in string:                                                           # replacing quotes by white-space
                string = string.replace('"', ' ')

            string = remove_multipule_spaces.sub(' ', string)                           # replacing mulitple spaces or tabs by one space
            if '/*' or '//' in string:
                string = cpp_comment_regex.sub('', string)
                string = line_comment_regex.sub('', string)                             # Striping comments off both types of comments

            self.network = string
        else:
            raise ValueError("Must specify either path or string")

        self.network_name()
        self.get_variables_info()

    def network_name(self):

        """
        Retruns the name of the network
        
        network attributes are of the format
            network <name_of_network> {
                attirbutes
            }
        Example of network attribute
        ------------------------------
        network "Dog-Problem" { //5 variables and 5 probability distributions
	        property "credal-set constant-density-bounded 1.1" ;
        }
        
        Useage
        ---------------
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.network_name()
        'Dog-Problem'
        """
        network_attribute = Suppress('network') +Word(alphanums+'_'+'-') +'{'                            # Creating a network attribute 
        temp = network_attribute.searchString(self.network)
        self.network_name = temp[0][0]
        return self.network_name

    def get_variables_info(self):

        """
        Functions gets all type of variable information
        
        variable block is of format
        
        variable <name_of_variable> {
            attribute1
            attribute2
            ...
        }
        Example of variable attribute
        ----------
        variable  "light-on" { //2 values
	        type discrete[2] {  "true"  "false" };
	        property "position = (218, 195)" ;
        }
        """
        variable_block_starts = [x.start() for x in re.finditer('variable', self.network)]  # Finding the beginning of variable block
        variable_block = []

        for i in variable_block_starts:
            variable_block_end = self.network.find('}\n', i)                                # Finding the end of variable block
            variable_block.append( self.network[ i : variable_block_end ] )                 # Appending the complete variable block in a list

        variable_names = []                                                                 # Creating an empty list for variable names
        variable_states = {}                                                                # Creating an empty dictionary for variable states
        variable_properties = {}                                                            # Creating an empty dictionary for variable prop   
        word_expr = Word(alphanums+'_'+'-')                                                 # Defining a expression for valid word
        name_expr = Suppress('variable')+ word_expr + Suppress('{')                         # Creating a expression for finding variable name
        state_expr= ZeroOrMore( word_expr + Optional( Suppress(",") ) )

        # Defining a variable state expression
        variable_state_expr = Suppress('type') + Suppress(word_expr) + Suppress('[') + Suppress(Word(nums))+\
        Suppress(']') + Suppress('{') + Group(state_expr) + Suppress('}') + Suppress(';')
        # variable states is of the form type description [args] { val1, val2 }; (comma may or may not be present)

        property_expr = Suppress('property') + CharsNotIn(';') + Suppress(';')              # Creating a expr to find property

        for block in variable_block:
            temp = name_expr.searchString(block)                                            # Finding the string of the format name_expr
            name = temp[0][0]                                                               # Assigning name the value of variable name
            variable_names.append(name) 
            temp = variable_state_expr.searchString(block)                                  # Finding the variable states
            variable_states[name] = list(temp[0][0])                                        # Assigning the variable states in form of list
            properties = property_expr.searchString(block)                                  # Getting the properties of variable                
            variable_properties[name] = [ x[0].strip() for x in properties ]

        self.variable_names = variable_names
        self.variable_states = variable_states
        self.variable_properties = variable_properties
        return

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
