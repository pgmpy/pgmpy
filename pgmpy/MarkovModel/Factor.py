from .Exceptions import *

__author__ = 'navin'



class Factor:

    def __init__(self, node_objects, potentials):
        self.node_objects = node_objects
        self.potentials = potentials

    def number_of_nodes(self):
        return len(self.node_objects)

    def get_pos_from_index_list(self,list):
        index = 0
        for i in range(0,len(list)):
            index *= self.state_size_by_node_index(i)
            index += list[i]
        return index

    def get_pos_from_state_list(self, state_list):
        ind_list = []
        for i in range(0, state_list):
            flag=False
            for j in range(0, len(self.states_by_node_index(i))):
                if self.states_by_node_index(i)[j]==state_list[i]:
                    ind_list.append(j)
                    flag=True
                    break
            if not flag:
                raise ObservationNotFound("")



    def get_potential_from_state_list(self, list):
        return self.potentials[self.get_pos_from_state_list(list)]

    def states_by_node_index(self,i):
        return self.node_objects[i][1]['_states']

    def state_size_by_node_index(self, i):
        return len(self.states_by_node_index(i))

    def print_potential_h(self, pos, ind_list, str_list):
        if pos == self.number_of_nodes():
            for node_str in str_list:
                print(node_str,end="\t")
            index = self.get_pos_from_index_list(ind_list)
            print(self.potentials[index])
        else:
            for i in range(0,self.state_size_by_node_index(pos)):
                ind_list[pos]=i
                str_list[pos]=self.states_by_node_index(pos)[i]
                self.print_potential_h(pos+1, ind_list, str_list)

    def print_potential(self):
        ind_list=[]
        str_list=[]
        print("-------------------------")
        for node_str_objects in self.node_objects:
            print(node_str_objects[0], end="\t")
            ind_list.append(None)
            str_list.append(None)
        print("potential")
        print("-------------------------")
        self.print_potential_h(0, ind_list, str_list)
        print("-------------------------")

    def get_potential_state_assignment(self, assignment):
        state_list=[]
        for node_str_obj in self.node_objects:
            node_str=node_str_obj[0]
            state_list.append(assignment[node_str])
        return self.get_potential_from_state_list(state_list)

    def get_potential_index_assignment(self, assignment):
        index_list=[]
        for node_str_obj in self.node_objects:
            node_str=node_str_obj[0]
            index_list.append(assignment[node_str])
        return self.potentials[self.get_pos_from_index_list(index_list)]


