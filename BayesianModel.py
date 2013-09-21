#!/usr/bin/env python3

import networkx as nx
import numpy as np
import ExceptionsPgmPy as epp

def string_to_tuple(string):
    """Converts a single string into a tuple with a string element."""
    return (string,)

class BayesianModel(nx.DiGraph):
    """ Public Methods
    --------------
    Graph:
    add_nodes(*args)
    add_edges(tail, head)
    
    Node:
    add_states(node, states)
    
    add_rule_for_states(node, states)
    rule_for_states(node)
    print_states(node)
    
    add_rule_for_parents(node, parents)
    rule_for_parents(node)
    print_parents(node)
    
    add_cpd(node, cpd)
    
    
    """
    #__init__ is inherited      
    def add_nodes(self, *args):
        """Adds nodes to graph with node-labels as provided in function.
        """
        for item in args:
            if type(item) != type('str'): 
                raise TypeError("Name of nodes must be strings.")
                
        self.add_nodes_from(args)
        
    def add_edges(self, tail, head):
        """Takes two tuples of nodes as input. All nodes in 'tail' are 
        joint to all nodes in 'head' with the direction of each edge is
        from a node in 'tail' to a node in 'head'.
        """
        if type(tail) == 'str': tail = string_to_tuple(tail)
        if type(head) == 'str': head = string_to_tuple(head)
        
        for end_node in head:
            for start_node in tail:
                self.add_edge(start_node, end_node)

            self.node[end_node]['parents'] = sorted(self.predecessors(end_node))
            self.node[end_node]['_rule_for_parents'] = (index
                                                    for index
                                                    in range(len(head)))

        
                
    def add_states(self, node, states):
        """Adds state-name from 'state' tuple to 'node'."""
        self.node[node]['states'] = [[state, 0] 
                                    for state in sorted(states)]
        self.node[node]['_rule_for_states'] = (n 
                                               for n 
                                               in range(len(states))
                                               )
        #internal storage = [['a',0],['b',0],['c',0],]
        #user-given order = ('c','a','b')
        #_rule_for_states = (2,0,1)
        #Rule will contain indices with which internal order should be
        #accessed
        
        
    #def add_state(self, node, state):
        
        
    def add_rule_for_states(self, node, states):
        """Sets new rule for order of states"""
        #check whether all states are mentioned?
        
        _order = list()
        for user_given_state in states:
            for state in self.node[node]['states']:
                if state == user_given_state:
                    _order.append(self.node[node]['states'].index(state))
                    break
        self.node[node]['_rule_for_states'] = tuple(_order)     
            
    def rule_for_states(self, node):
        """Returns tuple with states in user-defined order"""
        _order = list()
        for index in self.node[node]['_rule_for_states']:
            _order.append(self.node[node]['states'][index][0])
        return _order
        
    def states(self, node):
        """Prints states in alphabetical order"""
        return [self.node[node]['states'][index][0]
               for index in range(len(self.node[node]['states']))]
        
    def add_rule_for_parents(self, node, parents):
        #check if all parents are mentioned and no extra parents are
        #present
        #_extra = set(parents) - set(self.predecessors(node))
        #_missing = set(self.predecessors(node)) - set(parents)
        #if not len(_missing):
            #raise epp.MissingParentsError(_missing)
            
        #if not len(_extra):
            #raise epp.ExtraParentsError(_extra)
            
        _order = list()
        for user_given_parent in parents:
            for parent in self.node[node]['parents']:
                if parent == user_given_parent:
                    _order.append(self.node[node]['parents'].index(parent))
                    break
        self.node[node]['_rule_for_parents'] = tuple(_order)
            
    def rule_for_parents(self, node):
        """Returns tuple with parents in order"""
        _order = list()
        for index in self.node[node]['_rule_for_parents']:
            _order.append(self.node[node]['parents'][index])
            print(self.node[node]['parents'][index])
        return _order
        
    def parents(self, node):
        """Prints parents in alphabetical order"""
        return self.node[node]['parents']
     
    def add_cpd(self, node, cpd):
        """Adds given CPD to node as numpy.array

        It is expected that CPD given will be a 2D array such that
        the order of probabilities in the array will be in according 
        to the rules specified for parent and states.
        An example is shown below.
        
        EXAMPLE
        -------
        student.add_states('grades', ('A','C','B'))
        student.add_rule_for_parents('grades', ('diff', 'intel'))
        student.add_rule_for_states('grades', ('A', 'B', 'C'))
        student.add_cpd('grade',
                        [[0.1,0.1,0.1,0.1,0.1,0.1],
                        [0.1,0.1,0.1,0.1,0.1,0.1], 
                        [0.8,0.8,0.8,0.8,0.8,0.8]]
                        )

        #diff:       easy                 hard
        #intel: dumb   avg   smart    dumb  avg   smart
        #gradeA: 0.1    0.1    0.1     0.1  0.1    0.1  
        #gradeB: 0.1    0.1    0.1     0.1  0.1    0.1
        #gradeC: 0.8    0.8    0.8     0.8  0.8    0.8
        """
        self.node[node]['cpd'] = np.array(cpd)
        


if __name__=='__main__':
    
    student = BayesianModel()
    student.add_nodes('diff', 'intel', 'grades')    
    student.add_edges(('diff', 'intel'), ('grades',))
    
    print(sorted(student.edges()))
    
    student.add_states('diff',('hard','easy'))
    print(student.states('diff'))
    
    student.add_states('intel',('smart', 'avg', 'dumb'))
    print(student.states('intel'))
    
    student.add_states('grades', ('A', 'B', 'C'))
    print(student.states('grades'))
    print(student.node['grades']['parents'])
    
    student.add_rule_for_parents('grades', ('intel', 'diff'))
    print(">>", student.node['grades']['_rule_for_parents'])
    student.add_rule_for_states('grades', ('C', 'A', 'B'))
    print(student.rule_for_parents)
    print(student.rule_for_states)
    
    
    
