__all__ = ['readuai',
	   'UAIReader',       
	  ]

import warnings
import networkx as nx
import os.path
from collections import OrderedDict as od
#import pgmpy import MarkovModel as pm
from pgmpy import BayesianModel as bm

warnings.warn("Not Complete. Please use only for "
              "reading and writing Bayesian Models.")

def readuai(path):
    ''' given a filepath to UAI format file, return corresponding PGM(Bayesian Network, Markov Model).
    Parameters:
    	path : file path of UAI format file
    
    Output:
    	Corresponding PGM
    '''
    reader = UAIReader(path=path)
    return reader.make_network()



class UAIReader(object):
   
    def __init__(self, path=None):
        if os.path.exists(os.path.expanduser(path)):
	    self.path = path
        else:    
	    raise ValueError("Must specify valid 'path' as parameter.")

    def make_network(self):
	fp = open(self.path, 'r') #fp->file pointer
	lines = fp.read().splitlines()
        lines = [line for line in lines if line !=''] 
	G = None
	if lines[0] == "BAYES" :
            G = bm.BayesianModel()
	#elif lines[0] == "MARKOV" :
        #    G = pm.MarkovModel()
        
	#Add nodes
        for variable in range(int(lines[1])):   # line 1 represents number of variables
            self.add_node(G, variable,lines[2].split(' ')[variable])

        #Add edges
	noofcpd=int(line[3])
        for i in range(1,noofcpd+1):
            edge = line[3+i]
	    edge = edge.split(' ')
	    child = edge[-1]
	    if int(edge[0] != 1):
	        parents = edge[1:-1]
	        for parent in parents:
	    	    self.add_edge(G, parent, child)

        #Add CPD
        #TODO: parse potential
	fp.close()
        return G

    @staticmethod
    def add_node(G, variable,nstates):
        ''' nstates represents domain size of variable '''
        name = "node" + str(variable)
        G.add_node(name)
        G.set_states({name: ["state"+str(state) for state in range(0,int(nstates))]})


    @staticmethod
    def add_edge(G, u, v):
        ''' edge added from parent u to child v'''
	var1 = "node" + str(u)
	var2 = "node" + str(v)
        G.add_edge(var1, var2)
        
  
