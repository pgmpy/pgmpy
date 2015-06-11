from math import log
import itertools
import numpy as np
from operator import sub
from operator import add

class MPLPAlg:

    """
    The MAP problem asks us to maximize some potentials. (call this primal).
    If we take the dual fo the above and maximize it, we get an upper bound.
    With each iteration of GMPLP, we decrease the dual objective so that it comes near the MAP(primal) solution.
    """

    def __init__(self, all_cardinalities, all_factors, reader):

        # For containing the cardinalities of all the variables present in the model
        self.m_var_sizes = all_cardinalities
        # All the factors are kept here
        self.m_all_intersect = all_factors
        # All the messages per factor is kept here. It is updated in each iteration of RunMPLP
        self.m_sum_into_intersects = []
        # It contain the Region objects for all the factors which are made of more than one variable
        self.m_all_regions = []
        # Initial messages for more than single node regions are kept here. These don't change.
        self.m_region_lambdas = []
        # Maximum integral solution of the single nodes makes the m_best_val
        self.m_best_val =- 1000000
        # This is to be used for calculating the breaking point of RunMPLP
        self.last_obj = 1e40
        for i in range(reader.no_of_functions):
            self.m_sum_into_intersects.append([log(y) for y in all_lambdas[i]])
            if len(all_factors[i])>1:
                self.m_region_lambdas.append(self.m_sum_into_intersects[i])
                curr_region = Region(all_factors[i], all_factors[:i+1], all_factors[i], all_cardinalities, i)
                self.m_all_regions.append(curr_region)

            # Initialise output vector
            self.m_decoded_res = [0]*len(all_cardinalities)
            self.m_best_decoded_res = [0]*len(all_cardinalities)

    # To convert a [1 0 0] to its flat value 4 if the cardinality for each variable is [2 2 2]
    # Similarly to convert a [1 0 0] to 6 if the cardinality foreach variable is [4 3 2]
    def GetFlatInd(self, base_ind_for_multi_index, region_base_sizes):
        return [list(i) for i in itertools.product(*[list(range(k)) for k in region_base_sizes])].index(base_ind_for_multi_index)

    # To see if m_decoded _res indeed is the m_best_decoded_res, we need to calculate int_val.
    def IntVal(self):
        int_val=0
        for i in range(len(self.m_all_regions)):
            tmpvec = []
            base_sizes = []
            for j in self.m_all_regions[i].m_region_inds:
                tmpvec.append(self.m_decoded_res[j])
                base_sizes.append(self.m_var_sizes[j])
            int_val += self.m_region_lambdas[i][self.GetFlatInd(tmpvec, base_sizes)]
        return int_val

    # Find an integral solution x by locally maximizing the single node beliefs.
    # Here x = m_best_decoded_res
    def LocalDecode(self):
        obj=0
        for i in range(len(self.m_sum_into_intersects)):
            obj+=max(self.m_sum_into_intersects[i])
            index_of_maximum=self.m_sum_into_intersects[i].index(max(self.m_sum_into_intersects[i]))
            if(len(self.m_all_intersect[i])==1):
                self.m_decoded_res[self.m_all_intersect[i][0]]=index_of_maximum

        # update the m_best_val if it is smaller than int_val
        int_val=self.IntVal()
        print('int_val: "{}"'.format(int_val))

        if(int_val>self.m_best_val):
            self.m_best_decoded_res=self.m_decoded_res
            self.m_best_val=int_val
        return obj

    def RunMPLP(self, niter, obj_del_thr, int_gap_thr):
        # Perform the GMPLP updates.
        for i in range(niter):
            # For each of the 24 regions
            for r in self.m_all_regions:
                self.m_sum_into_intersects=r.UpdateMsgs(self.m_sum_into_intersects)
            obj=self.LocalDecode()
            obj_del= self.last_obj-obj
            self.last_obj=obj
            int_gap=obj-self.m_best_val

            if (obj_del<obj_del_thr) and (i>16) :
                break
            if int_gap<int_gap_thr:
                break


class Region:

    def __init__(self, m_region_inds, all_intersects, m_intersect_inds, var_sizes, m_region_intersect):
        self.m_region_inds = m_region_inds
        self.m_intersect_inds = m_intersect_inds
        self.m_region_intersect = m_region_intersect
        self.m_msgs_from_region = []
        self.region_var_sizes=[var_sizes[i] for i in m_region_inds]
        for i in self.m_intersect_inds:
            curr_intersect=all_intersects[i]
            product=1
            for j in curr_intersect:
                product = product*var_sizes[j]
            self.m_msgs_from_region.append([0]*product)

        # In a region you may have [[a, b],[b, d], [d, a]] where [a, b], [b, d] and [d, a] are 3 of its intersects.
        # In the m_region_inds, you have a list of index of all the variables present in the region=[1, 0, 3]
        # {0 for a, 1 for b, 3 for d}[The order is user provided.]According to this b comes first.
        # m_inds_of_intersect for this region then would be=[[1, 0],[0, 2],[2, 1]]

        # These are region variables for the region index
        region_vars = [i for i in range(len(var_sizes))]
        actual_variables_corresponding_to_region_idx = [region_vars[i] for i in m_region_inds]
        self.m_inds_of_intersect = []
        for i in self.m_intersect_inds:
            curr_intersect = all_intersects[i]
            self.m_inds_of_intersect.append([actual_variables_corresponding_to_region_idx.index(j) for j in curr_intersect])

    def get_idx(self,the_curr_base_bits, current_base_index):

        if len(current_base_index)==1:
            idx=the_curr_base_bits[current_base_index[0]]
        elif len(current_base_index)==2:
            print("is this even true")
            # The below 2 is actually the cardinality of the 2nd variable of
            # that intersect to which this message is passed to!
            # So we must know the intersect we are actually passing our getidx.
            idx = the_curr_base_bits[current_base_index[0]]*2 + the_curr_base_bits[current_base_index[1]]

        return idx

    # ExpandAndAdd works on that message which is directed from the region to one of its intersects.
    def ExpandAndAdd(self, orig, curr_inds_of_intersect, m_msgs_from_region_to_this_intersect):
        # we will try to get the size of the m_base_size of orig from the current regions
        # length of m_region_index for a region will be the will be the length of index of big_array
        ind_to_replaced = []
        for lst in [list(i) for i in itertools.product(*[list(range(k)) for k in self.region_var_sizes])]:
            ind=self.get_idx(lst, curr_inds_of_intersect)
            ind_to_replaced.append(ind)

        new_message_to_added=[m_msgs_from_region_to_this_intersect[y] for y in ind_to_replaced]
        orig = list(map(add, orig, new_message_to_added))

        return orig

    # ExpandAndSubtract works on that message which is directed from the region to one of its intersects.
    def ExpandAndSubtract(self, orig, curr_inds_of_intersect, m_msgs_from_region_to_this_intersect):
        # we will try to get the size of the m_base_size of orig from the current regions
        # length of m_region_index for a region will be the will be the length of index of big_array
        ind_to_replaced = []
        for lst in [list(i) for i in itertools.product(*[list(range(k)) for k in self.region_var_sizes])]:
            ind = self.get_idx(lst, curr_inds_of_intersect)
            ind_to_replaced.append(ind)

        new_message_to_sub = [m_msgs_from_region_to_this_intersect[y] for y in ind_to_replaced]
        orig = list(map(sub, orig, new_message_to_sub))

        return orig

    def UpdateMsgs(self, sum_into_intersects):
        """
        First do the expansion:
        1. Take out the message into the intersection set from the current cluster
        2. Expand it to the size of the region
        3. Add this for all intersection sets
        """
        # orig will have the messages
        orig = sum_into_intersects[self.m_region_intersect]
        # we iterate over all the intersects in the region.

        for i in range(len(self.m_intersect_inds)):
            orig = self.ExpandAndAdd(orig, self.m_inds_of_intersect[i], self.m_msgs_from_region[i])

        # Iterating over the intersects present in the region.
        total_msgs_minus_region = []
        for i in range(len(self.m_intersect_inds)):

            # Index of the current_intersect from the all_factors table
            curr_intersect_idx = self.m_intersect_inds[i]

            # Store the total messages that are going into the intersect but not emanating from the current region
            total_msgs_minus_region.append\
                (list(map(sub, sum_into_intersects[curr_intersect_idx], self.m_msgs_from_region[i])))

            # We will see if the no. of variables in the present region is==the no. of variables in the present intersect
            if len(self.region_var_sizes) == len(self.m_inds_of_intersect[i]):
                # No need to expand and add
                sum_into_intersects[self.m_region_intersect] = list(map(add,\
                        sum_into_intersects[self.m_region_intersect], sum_into_intersects[curr_intersect_idx]))
            else:
                # Now we will update the sum_into_intersect[16 or 17 or 18 ...]
                # using sum_into_intersect[0, 1, 2 .., 15]
                # Here we are assuming that
                sum_into_intersects[self.m_region_intersect] = self.ExpandAndAdd\
                    (sum_into_intersects[self.m_region_intersect], self.m_inds_of_intersect[i],
                     sum_into_intersects[curr_intersect_idx])

        # We update here the sum_into_intersect for those variables which are in
        # the present region i.e [0,1] or [1,2] ...
        # We set m_msgs_from_region here
        self.m_msgs_from_region = self.max_into_multiple_subsets_special(sum_into_intersects[self.m_region_intersect])

        sC = len(self.m_intersect_inds)
        for si in range(sC):
            current_intersect_idx = self.m_intersect_inds[si]
            self.m_msgs_from_region[si] = [x*1/sC for x in self.m_msgs_from_region[si]]
            sum_into_intersects[current_intersect_idx]=self.m_msgs_from_region[si]
            self.m_msgs_from_region[si] = list(map(sub, self.m_msgs_from_region[si], total_msgs_minus_region[si]))
            orig=self.ExpandAndSubtract(orig, self.m_inds_of_intersect[si], self.m_msgs_from_region[si])
        sum_into_intersects[self.m_region_intersect]=orig

        return sum_into_intersects

    def max_into_multiple_subsets_special(self, sum_into_intersect_for_a_region):
        lst = [list(i) for i in itertools.product(*[list(range(k)) for k in self.region_var_sizes])]
        a = [[-1e09]*len(i) for i in self.m_msgs_from_region]
        for vi in range(len(sum_into_intersect_for_a_region)):
            nsubsets = len(self.m_inds_of_intersect)

            for si in range(nsubsets):
                if self.m_inds_of_intersect[si] != len(self.region_var_sizes):
                    ind = self.get_idx(lst[vi], self.m_inds_of_intersect[si])
                    a[si][ind] = max(sum_into_intersect_for_a_region[vi], a[si][ind])
        return a

"""
    Use case:

    >>> reader=UAIReader(path="grid.uai")
    >>> all_lambdas = reader.params
    >>> all_factors=[]
    >>> for i in range(reader.no_of_functions):
    >>>     all_factors.append(list(map(int, reader.function_vars[i])))
    >>> all_cardinalities=reader.cardinality
    >>> mplp=MPLPAlg(all_cardinalities, all_factors, reader)
    >>> mplp.RunMPLP(1000, 0.0002, 0.0002) # 31 is the number of iterations here.
"""
