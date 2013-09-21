#Test push
#!/usr/bin/env python3

import networkx as nx
import numpy as np

class BayesianModel(nx.DiGraph):
    """ Documentation Pending."""
    #__init__ is inherited      
        
    def add_nodes(self, *args):
        """Adds nodes to graph with node-labels as provided in function.
        """
        self._nodes_unsorted = [node_name for node_name in args]
        self._nodes = sorted(self._nodes_unsorted, key=str.lower)
        #using alphabetical order for all parameters internally
        
        self.add_nodes_from({node:tuple() 
                                    for node in self._nodes})  

        
    def add_edges(self, tail, head):
        """Takes two tuples of nodes as input. All nodes in 'tail' are 
        joint to all nodes in 'head' with the direction of each edge is
        from a node in 'tail' to a node in 'head'.
        
        CAUTION: When only one node is present in either 'tail' or 
        'head', append with a comma.
        eg: graph.add_edges(("node1",),(node2,node3))
        """
        for start_node in tail:
            for end_node in head:
                self.add_edge(start_node, end_node)
                
        
    def add_cpd(self, node, parents, states, cpd):
        """Adds CPD to node with order of CPD as specified in parents 
        and states.
        
        Adds variable-states to given node from tuple 'states'.
        Takes order of column name of 2D-matrix-CPD from 'parents'.
        Takes order of row name of 2D-matrix-CPD from 'states'.
        
        eg:
        student.add_cpd("intel", (), ("dumb", "avg", "smart"),
                                      [[0.5], [0.3], [0.2]]) 
        #dumb=0.5
        #avg=0.3
        #smart=0.2
        
        eg:
        student.add_cpd("grade", ("diff","intel"), 
                          ("A", "B", "C"), 
                          [[0.1,0.1,0.1,0.1,0.1,0.1],
                          [0.1,0.1,0.1,0.1,0.1,0.1], 
                          [0.8,0.8,0.8,0.8,0.8,0.8]])

        #diff:       easy                 hard
        #intel: dumb   avg   smart    dumb  avg   smart
        #gradeA: 0.1    0.1    0.1     0.1  0.1    0.1  
        #gradeB: 0.1    0.1    0.1     0.1  0.1    0.1
        #gradeC: 0.8    0.8    0.8     0.8  0.8    0.8
        
        """
        #checking if parents in tuple are really parents
        if set(parents) != set(self.predecessors(node)):
            raise ParentsMismatch

if __name__=='__main__':
    
    student = BayesianModel()
    student.add_nodes("diff","intel","grades")    
    student.add_edges(("diff","intel"),("grades",))
    print(sorted(student.edges()))

