import numpy as np
from pyparsing import Word, alphanums, Suppress, Optional, CharsNotIn, Group, nums, ZeroOrMore, OneOrMore, cppStyleComment
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
import re


class BifReader(object):

    """
    Base class for reading network file in bif format
    """
    def __init__(self, path=None, string=None):

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
        remove_multipule_spaces = re.compile(r'[" ""\t""\r""\f"][" ""\t""\r""\f"]*')    # A regular expression to check for multiple spaces

        if path:
            self.network = open(path, 'r').read()                                               

        elif string:
            self.network = string        

        else:
            raise ValueError("Must specify either path or string")

        if '"' in self.network:
            """
            Replacing quotes by spaces to remove case sensitivity like:
            "Dog-Problem" and Dog-problem
            or "true""false" and "true" "false" and true false
            """
            self.network = self.network.replace('"', ' ')

        self.network = remove_multipule_spaces.sub(' ', self.network)                # replacing mulitple spaces or tabs by one space

        if '/*' or '//' in self.network:
            self.network = cppStyleComment.suppress().transformString(self.network)  # removing comments from the file

        self.get_network_name()
        self.get_variables_info()
        self.get_cpd()
        self.get_edges()
        self.model = self.get_model()

    def get_network_name(self):

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

        Sample run
        ---------------
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.network_name()
        'Dog-Problem'
        """
        network_attribute = Suppress('network') + Word(alphanums+'_'+'-') + '{'                            # Creating a network attribute 
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
        state_expr = ZeroOrMore( word_expr + Optional( Suppress(",") ) )

        # Defining a variable state expression
        variable_state_expr = Suppress('type') + Suppress(word_expr) + Suppress('[') + Suppress(Word(nums)) + \
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
            variable_properties[name] = [x[0].strip() for x in properties ]

        self.variable_names = variable_names
        self.variable_states = variable_states
        self.variable_properties = variable_properties
        return

    def get_variables(self):

        """
        Returns list of variables of the network

        Sample run
        -------------
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.get_variables()
        ['light-on','bowel_problem','dog-out','hear-bark','family-out']
        """
        return self.variable_names

    def get_states(self):

        """
        Returns the states of variables present in the network

        Sample run
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

        Sample run
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

        probability attribute is of the form
        probability (<args>){
            Optional(table) probabilities1
            probabilities 2
            ..
        }
        Example of probability attribute
        ---------------------------------
        probability (  "light-on"  "family-out" ) { //2 variable(s) and 4 values
	    table 0.6 0.05 0.4 0.95 ;
        }    

        Sample run
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

        probability_block_starts = [x.start() for x in re.finditer('probability', self.network)]
        probability_block = []
        for i in probability_block_starts:
            probability_block_end = self.network.find('}\n',i)
            probability_block.append(self.network[i:probability_block_end])

        word_expr = Word(alphanums + '-' + '_') + Suppress(Optional("|")) + Suppress(Optional(","))
        num_expr = Word(nums + '-' + '+' + 'e' +'E' +'.') + Suppress(Optional(","))
        probability_expr = Suppress('probability') + Suppress('(') + OneOrMore(word_expr) + Suppress(')')
        optional_expr = Suppress('(') + Suppress(OneOrMore(word_expr)) + Suppress(')')
        probab_attributes = optional_expr | Suppress('table')
        cpd_expr = probab_attributes + OneOrMore( num_expr)

        variable_parents = {}
        variable_cpds ={}

        for block in probability_block:
            names = probability_expr.searchString(block)
            names = names[0]
            variable = names[0]
            names = names[1:]
            variable_parents[variable] = names
            temp = cpd_expr.searchString(block)
            arr = [float(j) for i in temp for j in i]
            arr = np.array(arr)
            arr = arr.reshape((len(self.variable_states[variable]), arr.size//len(self.variable_states[variable])))
            variable_cpds[variable] = arr

        self.variable_parents = variable_parents
        self.variable_cpds = variable_cpds
        return self.variable_cpds

    def get_parents(self):

        """
        Returns the parents of the variables present in the network

        Sample run
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_parents()
        {'bowel-problem': [],
         'dog-out': ['family-out', 'bowel-problem'],
         'family-out': [],
         'hear-bark': ['dog-out'],
         'light-on': ['family-out']}
        """
        return self.variable_parents

    def get_edges(self):

        """
        Returns the edges of the network

        Sample run
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_edges()
        [['family-out', 'light-on'],
         ['family-out', 'dog-out'],
         ['bowel-problem', 'dog-out'],
         ['dog-out', 'hear-bark']]
        """
        self.edges = [[value, key] for key in self.variable_parents.keys()
                     for value in self.variable_parents[key]]
        return self.edges

    def get_model(self):

        """
        Returns the fitted bayesian model
        """
        model = BayesianModel(self.edges)
        model.name = self.network_name

        tabular_cpds = []
        for var, values in self.variable_cpds.items():
            cpd = TabularCPD(var, len(self.variable_states[var]), values,
                             evidence = self.variable_parents[var],
                             evidence_card = [len(self.variable_states[evidence_var])
                                            for evidence_var in self.variable_parents[var]])
            tabular_cpds.append(cpd)

        model.add_cpds(*tabular_cpds)

        for node, properties in self.variable_properties.items():
            for prop in properties:
                prop_name, prop_value = map(lambda t: t.strip(), prop.split('='))
                model.node[node][prop_name] = prop_value

        return model
