from math import log
import itertools
import numpy as np
from operator import sub
from operator import add
import queue
import math


class MPLPAlg:
    """
    Class for performing inference using Max-Product Linear Programming Relaxations.
    """

    def __init__(self, all_cardinalities, all_factors, all_lambdas):

        # For containing the cardinalities of all the variables present in the model
        self.m_var_sizes = all_cardinalities
        # All the factors are kept here
        self.m_all_intersect = all_factors
        # All the messages per factor is kept here. It is updated in each iteration of run_mplp
        self.objective = []
        # It contain the Region objects for all the factors which are made of more than one variable
        self.m_all_regions = []
        # Initial messages for more than single node regions are kept here. These don't change.
        self.m_region_lambdas = []
        # Maximum integral solution of the single nodes makes the m_best_val
        self.m_best_val = -1000000
        # This is to be used for calculating the breaking point of RunMPLP
        self.last_obj = 1e40
        self.m_intersect_map = []
        for i in range(len(all_factors)):
            self.objective.append([log(y) for y in all_lambdas[i]])
            if len(all_factors[i]) > 1:
                self.m_region_lambdas.append(self.objective[i])
                curr_region = Region(all_factors[i], all_factors[:i+1], all_factors[i], all_cardinalities, i)
                self.m_all_regions.append(curr_region)
            if len(all_factors[i]) == 2:
                self.m_intersect_map.append((all_factors[i], i))

            # Initialise output vector
            self.m_decoded_res = [0]*len(all_cardinalities)
            self.m_best_decoded_res = [0]*len(all_cardinalities)

    def _get_flat_ind(self, base_ind, base_card):
        """
        Converts:
        base_ind=[1 0 0] to 4 for base_cardinality=[2 2 2]
        base_ind=[1 0 0] to 6 for base_cardinality=[4 3 2]
        """
        return [list(i) for i in itertools.product(*[list(range(k)) for k in base_card])].index(base_ind)

    def _integer_val(self):
        """
        The original factor potentials gives us one of the integer solutions.
        """
        int_val = 0
        for i in range(len(self.m_all_regions)):
            tmpvec = []
            base_sizes = []
            for j in self.m_all_regions[i].rvariables_inds:
                tmpvec.append(self.m_decoded_res[j])
                base_sizes.append(self.m_var_sizes[j])
            int_val += self.m_region_lambdas[i][self._get_flat_ind(tmpvec, base_sizes)]
        return int_val

    def _local_decode(self):
        """
        Finds the best assignment from a set of assignment by locally maximizing the single node beliefs
        """
        obj = 0
        for i in range(len(self.objective)):
            obj += max(self.objective[i])
            index_of_maximum = self.objective[i].index(max(self.objective[i]))
            if len(self.m_all_intersect[i]) == 1:
                self.m_decoded_res[self.m_all_intersect[i][0]] = index_of_maximum

        # update the m_best_val if it is smaller than int_val
        int_val = self._integer_val()
        print('int_val: "{}"'.format(int_val))
        if int_val > self.m_best_val:
            self.m_best_decoded_res = self.m_decoded_res
            self.m_best_val = int_val
        return obj

    def run_mplp(self, niter, obj_del_thr, int_gap_thr):
        """
        Perform the GMPLP updates.
        """
        for i in range(niter):
            # For each of the 24 regions
            for r in self.m_all_regions:
                self.objective = self._update_msgs(self.objective, r)
            obj = self._local_decode()
            obj_del = self.last_obj-obj
            self.last_obj = obj
            int_gap = obj-self.m_best_val

            print(i)

            if (obj_del < obj_del_thr) and (i > len(self.m_var_sizes)):
                break
            if int_gap < int_gap_thr:
                break

    def _break_list(self, big_message, region):
        """
        Break the region's message into sub-messages for each of it's intersection
        """
        lst = [list(i) for i in itertools.product(*[list(range(k)) for k in region.region_var_sizes])]
        subsets = [[-1e09]*len(i) for i in region.m_msgs_from_region]
        no_subsets = len(region.rintersects_relative_pos)
        for vi in range(len(big_message)):
            for si in range(no_subsets):
                ind = self._get_idx(lst[vi], region.rintersects_relative_pos[si], region)
                subsets[si][ind] = max(big_message[vi], subsets[si][ind])
        return subsets

    def _get_idx(self, bit_value_big, active_index, region):
        """
        Converts the bit value's active indexes into decimal number corresponding to the region's cardinality.
        For eg:
        for a bit_value_big=[1, 0, 1] and active_index=[1, 0] and region's card=[3, 3, 2], it returns 1
        for a bit_value_big=[1, 1, 1] and active_index=[1, 2] and region's card=[3, 3, 2], it returns 3
        """
        bit_values = [bit_value_big[i] for i in active_index]
        card_values = [region.region_var_sizes[i] for i in active_index]
        multiplier = [np.product([j for j in card_values[i+1:]]) for i in range(len(card_values)-1)]+[1]
        decimal_value = np.sum([bit_values[i]*multiplier[i] for i in range(len(card_values))])
        return decimal_value

    def _expand_and_operate(self, big_message, curr_inds_of_intersect, small_message, operation, region):
        """
        To Add/Sub smaller messages on a bigger messages.
        """
        ind_to_replaced = []
        for lst in [list(i) for i in itertools.product(*[list(range(k)) for k in region.region_var_sizes])]:
            ind = self._get_idx(lst, curr_inds_of_intersect, region)
            ind_to_replaced.append(ind)
        expanded_message = [small_message[y] for y in ind_to_replaced]

        if operation == 'Add':
            big_message = list(map(add, big_message, expanded_message))
        elif operation == 'Sub':
            big_message = list(map(sub, big_message, expanded_message))

        return big_message

    def _update_msgs(self, objective, region):
        """
        Updates all the messages related to the current Region.
        """
        n_intersects = len(region.rintersect_inds)
        orig = objective[region.region_loc]

        for i in range(n_intersects):
            orig = self._expand_and_operate(orig, region.rintersects_relative_pos[i], region.m_msgs_from_region[i],
                                            'Add', region)

        # Iterating over the intersects present in the region.
        total_msgs_minus_region = []
        for i in range(n_intersects):

            curr_intersect_idx = region.rintersect_inds[i]
            # Store the total messages that are going into the intersect but not emanating from the current region
            total_msgs_minus_region.\
                append(list(map(sub, objective[curr_intersect_idx], region.m_msgs_from_region[i])))

            # if the no. of variables in the present region is==the no. of variables in the present intersect
            if len(region.rvariables_inds) == len(region.rintersects_relative_pos[i]):
                objective[region.region_loc] = list(map(add, objective[region.region_loc],
                                                        objective[curr_intersect_idx]))
            else:
                objective[region.region_loc] = self.\
                    _expand_and_operate(objective[region.region_loc], region.rintersects_relative_pos[i],
                                        objective[curr_intersect_idx], 'Add', region)

        region.m_msgs_from_region = self._break_list(objective[region.region_loc], region)
        sc = n_intersects
        for si in range(n_intersects):
            current_intersect_idx = region.rintersect_inds[si]
            region.m_msgs_from_region[si] = [x*1/sc for x in region.m_msgs_from_region[si]]
            objective[current_intersect_idx] = region.m_msgs_from_region[si]
            region.m_msgs_from_region[si] = list(map(sub, region.m_msgs_from_region[si], total_msgs_minus_region[si]))
            orig = self._expand_and_operate(orig, region.rintersects_relative_pos[si],
                                            region.m_msgs_from_region[si], 'Sub', region)
        objective[region.region_loc] = orig
        return objective

    def find_intersection_set(self, intersection_ind):
        if intersection_ind in self.m_all_intersect:
            return self.m_all_intersect.index(intersection_ind)
        else:
            return -1

    def add_intersection_set(self, new_intersection_ind):
        self.m_all_intersect.append(new_intersection_ind)
        if len(new_intersection_ind) == 2:
            self.m_intersect_map.append((new_intersection_ind, len(self.m_all_intersect) - 1))
        self.objective.append([0]*np.product([self.m_var_sizes[ind] for ind in new_intersection_ind]))
        return len(self.m_all_intersect)-1

    def add_region(self, inds_of_vars, intersect_inds):
        region_intersection_set = self.add_intersection_set(inds_of_vars)
        new_region = Region(inds_of_vars, self.m_all_intersect, intersect_inds, self.m_var_sizes,
                            region_intersection_set)
        self.m_all_regions.append(new_region)
        self.m_region_lambdas.append([0]*(2**len(inds_of_vars)))
        return region_intersection_set

    def create_k_projection_graph(self):

        # Initialize the projection graph.
        projection_map = []
        projection_edge_weights = []
        projection_node_iter = 0
        for node in range(len(self.m_var_sizes)):
            i_temp = []
            for state in range(self.m_var_sizes[node]):
                i_temp.append(projection_node_iter)
                projection_node_iter += 1
            projection_map.append(i_temp)

        projection_adjacency_list = [[] for i in range(projection_node_iter)]
        list_of_sij = []

        # Now we will be building the inverse map(or the iMap)
        projection_imap_var = []
        partition_imap = []
        num_projection_nodes = projection_node_iter
        for node in range(len(self.m_var_sizes)):
            for state in range(self.m_var_sizes[node]):
                projection_imap_var.append(node)
                partition_imap.append([state])

        # Iterate over all of the edges
        import heapq

        kk = -1
        my_intersect_map = self.m_intersect_map.copy()
        my_intersect_map = sorted(my_intersect_map, key=lambda x: x[1])
        my_intersect_map = sorted(my_intersect_map, key=lambda x: x[0][0])
        for it in my_intersect_map:
            print("it: {}".format(it))
            kk += 1

            # Get the 2 nodes i & j and the edge intersection set. Put in right order.
            i = it[0][0]
            j = it[0][1]
            ij_intersect_loc = it[1]
            edge_belief = self.objective[ij_intersect_loc]

            # We swap i and j if we find that it doesn't match m_all_intersects exactly.
            # Check to see if i and j have at-least 2 states each, other wise cannot be part of frustrated edges.
            max_j_bij_not_xj = np.zeros([self.m_var_sizes[i], self.m_var_sizes[j]])
            max_i_bij_not_xi = np.zeros([self.m_var_sizes[i], self.m_var_sizes[j]])
            max_ij_bij_not_xi_xj = np.zeros([self.m_var_sizes[i], self.m_var_sizes[j]])

            for state1 in range(self.m_var_sizes[i]):
                # Find the largest and the 2nd largest val over state 2
                tmp_val_list = []
                for state2 in range(self.m_var_sizes[j]):
                    inds = [state1, state2]
                    tmp_val = edge_belief[self._get_flat_ind(inds, [2, 2])]
                    tmp_val_list.append(tmp_val)

                [largest_val, second_largest_val] = heapq.nlargest(2, tmp_val_list)
                largest_ind = tmp_val_list.index(largest_val)

                # assign values
                for state2 in range(self.m_var_sizes[j]):
                    max_j_bij_not_xj[state1, state2] = largest_val
                max_j_bij_not_xj[state1][largest_ind] = second_largest_val

                # Find the largest and the 2nd largest val over state 2
                tmp_val_list = []
                for state2 in range(self.m_var_sizes[j]):
                    inds = [state2, state1]
                    tmp_val = edge_belief[self._get_flat_ind(inds, [2, 2])]
                    tmp_val_list.append(tmp_val)

                [largest_val, second_largest_val] = heapq.nlargest(2, tmp_val_list)
                largest_ind = tmp_val_list.index(largest_val)

                # assign values
                for state2 in range(self.m_var_sizes[j]):
                    max_i_bij_not_xi[state2][state1] = largest_val
                max_i_bij_not_xi[largest_ind][state1] = second_largest_val

            for state1 in range(self.m_var_sizes[i]):
                # decompose: max_{x_j!=x_j}max_{x_i != x_i'}. Then, use the above computations.
                tmp_val_list = []
                for state2 in range(self.m_var_sizes[j]):
                    tmp_val = max_i_bij_not_xi[state1][state2]
                    tmp_val_list.append(tmp_val)

                [largest_val, second_largest_val] = heapq.nlargest(2, tmp_val_list)
                largest_ind = tmp_val_list.index(largest_val)

                # assign values
                for state2 in range(self.m_var_sizes[j]):
                    max_ij_bij_not_xi_xj[state1][state2] = largest_val
                max_ij_bij_not_xi_xj[state1][largest_ind] = second_largest_val

            # For each of their states
            for xi in range(self.m_var_sizes[i]):
                m = projection_map[i][xi]
                for xj in range(self.m_var_sizes[j]):
                    n = projection_map[j][xj]
                    inds = [xi, xj]
                    tmp_val = edge_belief[self._get_flat_ind(inds, [2, 2])]
                    # Compute s_mn for this edge
                    val_s = max(tmp_val, max_ij_bij_not_xi_xj[xi][xj]) -\
                            max(max_i_bij_not_xi[xi][xj], max_j_bij_not_xj[xi][xj])

                    if val_s != 0:
                        projection_adjacency_list[m].append((n, val_s))
                        projection_adjacency_list[n].append((m, val_s))
                        list_of_sij.append(abs(val_s))

                    # Insert into edge weight map
                    projection_edge_weights.append(((n, m), val_s))
                    projection_edge_weights.append(((m, n), val_s))

        projection_adjacency_list = [sorted(i, key=lambda tup: tup[0]) for i in projection_adjacency_list]
        set_of_sij = sorted(set(list_of_sij))
        from collections import namedtuple
        Result = namedtuple('Result', ['projection_map', 'num_projection_nodes', 'projection_imap_var', 'partition_imap'
            ,'projection_edge_weights', 'projection_adjacency_list', 'set_of_sij'])
        result = Result(projection_map, num_projection_nodes, projection_imap_var, partition_imap,
                                       projection_edge_weights, projection_adjacency_list, set_of_sij)

        return result, num_projection_nodes

    def find_optimal_R(self, projection_graph):

        binary_search_lower_bound = 0
        binary_search_upper_bound = len(projection_graph.set_of_sij)
        sij_min = -1
        num_projection_nodes = len(projection_graph.projection_adjacency_list)
        while binary_search_lower_bound <= binary_search_upper_bound:

            # compute the mid point
            mid_position = math.floor((binary_search_lower_bound + binary_search_upper_bound)/2)
            R = projection_graph.set_of_sij[mid_position]
            # Does there exist an odd signed cycle using just edges with sij >= R?
            # If yes, then go up.
            # else, go down.
            found_odd_signed_cycle = False

            # Zero denotes that node "not yet seen"
            node_sign = np.zeros(num_projection_nodes)

            # Graph may be disconnected, so check all the nodes
            current_node = 0

            while (found_odd_signed_cycle is False) and (current_node < num_projection_nodes):

                if node_sign[current_node] == 0:

                    # Set it as the root node.
                    node_sign[current_node] = 1
                    q = queue.Queue()

                    # Put the current node
                    q.put(current_node)

                    while (q.empty() is False) and (found_odd_signed_cycle is False):
                        front_node = q.get()
                        for adj_node in projection_graph.projection_adjacency_list[front_node]:
                            edge_weight = adj_node[1]

                            # Ignore edges with weights less than R
                            if abs(edge_weight) < R:
                                continue

                            adj_node_number = adj_node[0]
                            sign_adj_node = np.sign(edge_weight)

                            # Travel to the adjacent node
                            if node_sign[adj_node_number] == 0:
                                node_sign[adj_node_number] = node_sign[front_node] * sign_adj_node
                                q.put(adj_node_number)
                            elif node_sign[adj_node_number] == -node_sign[front_node] * sign_adj_node:
                                found_odd_signed_cycle = True
                                break
                current_node += 1

            if found_odd_signed_cycle is True:
                sij_min = R
                binary_search_lower_bound = mid_position + 1
            else:
                binary_search_upper_bound = mid_position - 1

        return sij_min

    def add_cycle(self, cycle, projection_imap_var, triplet_set, num_projection_nodes):

        nClustersAdded = 0

        # Number of clusters we are adding here is length_cycle-2
        nNewClusters = len(cycle)-2
        cluster_index = 0

        # Found the violated cycle. Now we triangulate and add the relaxation.
        tripletcluster_array=[]
        i = 0
        while i < (len(cycle)-3)/2:
            tripletcluster = {'i':0, 'j':0, 'k':0, 'ij_loc':0, 'jk_loc':0, 'ki_loc':0,}
            tripletcluster['i'] = projection_imap_var[np.int(cycle[i])]
            tripletcluster['j'] = projection_imap_var[np.int(cycle[i+1])]
            tripletcluster['k'] = projection_imap_var[np.int(cycle[len(cycle)-2-i])]
            tripletcluster_array.append(tripletcluster)
            i += 1

        i = len(cycle)-1
        while i > len(cycle)/2:
            tripletcluster = {'i':0, 'j':0, 'k':0, 'ij_loc':0, 'jk_loc':0, 'ki_loc':0,}
            tripletcluster['i'] = projection_imap_var[np.int(cycle[i])]
            tripletcluster['j'] = projection_imap_var[np.int(cycle[i-1])]
            tripletcluster['k'] = projection_imap_var[np.int(cycle[len(cycle)-1-i])]
            tripletcluster_array.append(tripletcluster)
            i -= 1

        # Add the top nclus_to_add clsuters to the relaxation
        for clusterId  in range(nNewClusters):
            # Check that these clusters and intersection sets haven't already been added
            temp = [tripletcluster_array[clusterId]['i'], tripletcluster_array[clusterId]['j'],
                    tripletcluster_array[clusterId]['k']]

            # we pass this iteration if the current cluster contains non unique elements
            if len(temp) > len(set(temp)):
                continue
            temp = list(np.sort(temp))

            # If you cannot find temp in the triplet set, then thats good.
            if temp not in triplet_set:
                triplet_set.append(temp)

            # Find the intersection sets for this triangle. If it doesn't then add!
            if self.find_intersection_set([temp[0], temp[1]]) == -1:
                tripletcluster_array[clusterId]['ij_loc'] = self.add_intersection_set([temp[0], temp[1]])
            else:
                tripletcluster_array[clusterId]['ij_loc'] = self.find_intersection_set([temp[0], temp[1]])

            if self.find_intersection_set([temp[1], temp[2]]) == -1:
                tripletcluster_array[clusterId]['jk_loc'] = self.add_intersection_set([temp[1], temp[2]])
            else:
                tripletcluster_array[clusterId]['jk_loc'] = self.find_intersection_set([temp[1], temp[2]])

            if self.find_intersection_set([temp[0], temp[2]]) == -1:
                tripletcluster_array[clusterId]['ki_loc'] = self.add_intersection_set([temp[0], temp[2]])
            else:
                tripletcluster_array[clusterId]['ki_loc'] = self.find_intersection_set([temp[0], temp[2]])

            ijk_inds = [tripletcluster_array[clusterId]['i'], tripletcluster_array[clusterId]['j'],
                        tripletcluster_array[clusterId]['k']]
            ijk_intersect_inds = [tripletcluster_array[clusterId]['ij_loc'], tripletcluster_array[clusterId]['jk_loc'],
                                  tripletcluster_array[clusterId]['ki_loc']]

            # Now write the code for AddRegion here
            self.add_region(ijk_inds, ijk_intersect_inds)
            nClustersAdded += 1
        return nClustersAdded

    def find_cycles(self, projection_graph, optimal_R):
        num_projection_nodes = len(projection_graph.projection_adjacency_list)
        node_sign = np.zeros(num_projection_nodes)
        node_depth = np.ones(num_projection_nodes) * -1
        node_parent = np.ones(num_projection_nodes) * -1

        # Construct the rooted spanning tree(s)[when the graph is disconnected we have more than 1 spanning tree]
        for node in range(num_projection_nodes):
            if node_sign[node] == 0:
                # Set it as the root node.
                node_sign[node] = 1
                # Parent of a root node is that itself.
                node_parent[node] = node
                # Distance between the current node and the Root
                node_depth[node] = 0
                q = queue.Queue()
                # Put the current node
                q.put(node)
                # Construct the spanning tree for this root.
                while q.empty() is False:
                    front_node = q.get()
                    for neighbour_node in projection_graph.projection_adjacency_list[front_node]:
                        neighbour_node_no = neighbour_node[0]
                        # If this neighbour node has a greater weight than the Optimal and has not been traversed:
                        if node_sign[neighbour_node_no] == 0:
                            neighbour_node_wt = neighbour_node[1]
                            if abs(neighbour_node_wt) >= optimal_R:
                                sign_wt = np.sign(neighbour_node_wt)
                                # Assign the properties for this neighbourhood node which qualifies for Spanning tree.
                                node_sign[neighbour_node_no] = node_sign[front_node] * sign_wt
                                node_parent[neighbour_node_no] = front_node
                                node_depth[neighbour_node_no] = node_depth[front_node] + 1
                                # Put the neighbourhood node here
                                q.put(neighbour_node_no)

        # Now since the spanning tree has been formed, we need to get the most out of it.
        # Construct the Edge set that contains edges that are not parts of the tree
        total = 0
        edge_set = []
        node_i_no = -1
        for node_i in projection_graph.projection_adjacency_list:
            node_i_no += 1
            node_j_no = -1
            for node_j_tuple in node_i:
                node_j_no += 1
                if abs(node_j_tuple[1]) >= optimal_R:
                    node_j = node_j_tuple[0]
                    node_j_weight = node_j_tuple[1]
                    # If cycle is found
                    if node_sign[node_i_no] == -node_sign[node_j] * np.sign(node_j_weight):
                        print("cycle is found")
                        total += 1
                        temp_di = node_depth[node_i_no]
                        temp_dj = node_depth[node_j]

                        anc_i = node_i_no
                        anc_j = node_j

                        while temp_dj > temp_di:
                            anc_j = node_parent[anc_j]
                            temp_dj -= 1

                        while temp_di > temp_dj:
                            anc_i = node_parent[anc_i]
                            temp_di -= 1

                        print(temp_di)
                        while temp_di >= 0:
                            if anc_i == anc_j:

                                # LCA is found
                                temp = []
                                temp.append(node_i_no)
                                temp.append(node_j)
                                temp.append(node_depth[node_i_no] + node_depth[node_j] - temp_di - temp_dj)
                                print(temp)
                                edge_set.append(temp)
                                break
                            anc_i = node_parent[anc_i]
                            anc_j = node_parent[anc_j]
                            temp_di -= 1
                            temp_dj -= 1
        edge_set.sort(key=lambda x: x[2], reverse=True)
        cycle_set = []
        for i in edge_set:
            if len(cycle_set) < 50:

                # Get next edge (and depth) from the sorted list
                temp = edge_set.pop()

                # Find the least common ancestor
                left = temp[0]
                right = temp[1]
                left_d = node_depth[left]
                right_d = node_depth[right]
                left_anc = left
                right_anc = right

                # Find the first common ancestor
                while left_d > right_d:
                    left_anc = node_parent[left_anc]
                    left_d -= 1

                while right_d > left_d:
                    right_anc = node_parent[right_anc]
                    right_d -= 1

                while left_anc != right_anc:
                    left_anc = node_parent[left_anc]
                    right_anc = node_parent[right_anc]

                ancestor = left_anc
                cycle = []

                # Backtrace the found cycle.
                if left == ancestor:
                    cycle.append(left)
                else:
                    while node_parent[left] != ancestor:
                        cycle.append(left)
                        left = node_parent[left]
                    cycle.append(left)
                    cycle.append(node_parent[left])
                a = []

                if right != ancestor:
                    while node_parent[right] != ancestor:

                        a.append(right)
                        right = node_parent[right]

                    a.append(right)

                print(a[::-1])
                print(ancestor)
                cycle = a[::-1] + cycle
                cycle_set.append(cycle)

        return cycle_set

    def tighten_cycle(self):
        projection_graph, num_projection_nodes = self.create_k_projection_graph()
        # Now we go for the "find_optimal_R"
        nClustersAdded = 0
        optimal_R = self.find_optimal_R(projection_graph)

        # Look for cycles. Some will be discarded.
        # Given an undirected graph, finds an odd-signed cycle.
        # This works by breadth first search.
        cycle_set = self.find_cycles(projection_graph, optimal_R)

        # Add all the cycles that we have found to the relaxation.
        triplet_set = []
        for z in cycle_set:
            if nClustersAdded < 50:
                for it in z:
                    print(projection_graph.projection_imap_var[np.int(it)], end="")
                    temp = projection_graph.partition_imap[np.int(it)]
                    for s in temp[:]:
                        print("({}, ".format(s),  end="")
                    print("{}), ".format(len(temp)-1), end="")

                print("")

                # Add cycles to the relaxation
                nClustersAdded += self.add_cycle(z, projection_graph.projection_imap_var, triplet_set,
                                            num_projection_nodes)
        return nClustersAdded


class Region:
    """
    This is class which holds a group of factors in it.
    Factors and intersects are the same thing.
    """
    def __init__(self, rvariables_inds, all_intersects, rintersect_inds, all_var_sizes, region_loc):
        self.rvariables_inds = rvariables_inds
        self.rintersect_inds = rintersect_inds
        self.region_loc = region_loc
        self.region_var_sizes = [all_var_sizes[i] for i in rvariables_inds]
        self.m_msgs_from_region = []
        self.rintersects_relative_pos = []

        # We iterate over all the intersects in the present region.
        for i in self.rintersect_inds:
            curr_intersect = all_intersects[i]
            product = np.product([all_var_sizes[i] for i in curr_intersect])
            self.m_msgs_from_region.append([0]*product)
            self.rintersects_relative_pos.\
                append([rvariables_inds.index(i) for i in curr_intersect])


"""
    Use case:

    >>> reader=UAIReader(path="grid.uai")
    >>> all_lambdas = reader.params
    >>> all_factors=[]
    >>> for i in range(reader.no_of_functions):
    >>>     all_factors.append(list(map(int, reader.function_vars[i])))
    >>> all_cardinalities=reader.cardinality
    >>> mplp=MPLPAlg(all_cardinalities, all_factors, all_lambdas)
    >>> mplp.RunMPLP(1000, 0.0002, 0.0002) # 31 is the number of iterations here.
    >>> mplp.tighten_cycle()
"""