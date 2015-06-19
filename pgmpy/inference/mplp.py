from math import log
import itertools
import numpy as np
from operator import sub
from operator import add


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
        for i in range(len(all_factors)):
            self.objective.append([log(y) for y in all_lambdas[i]])
            if len(all_factors[i]) > 1:
                self.m_region_lambdas.append(self.objective[i])
                curr_region = Region(all_factors[i], all_factors[:i+1], all_factors[i], all_cardinalities, i)
                self.m_all_regions.append(curr_region)

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
"""
