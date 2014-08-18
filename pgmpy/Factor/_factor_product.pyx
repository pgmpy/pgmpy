cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_t, ndim=2] matrix_gen(np.ndarray[DTYPE_t, ndim=1] cardinality):
    cdef np.ndarray[DTYPE_t, ndim=1] index = np.arange(np.prod(cardinality))
    return (np.floor(np.tile(np.atleast_2d(index).T, (1, cardinality.shape[0])) / 
                     np.tile(np.cumprod(np.concatenate(([1], cardinality[:-1]))),
                             (index.shape[0], 1))) % np.tile(cardinality, (index.shape[0], 1))).astype(DTYPE)

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef pattern_gen(np.ndarray[DTYPE_t, ndim=1] x_card,
                 np.ndarray[DTYPE_t, ndim=1] y_card,
                 np.ndarray[DTYPE_t, ndim=2] common_index):
    cdef:
        np.ndarray[DTYPE_t, ndim=1] common_index_y, common_index_x
        np.ndarray[DTYPE_t, ndim=1] cum_y_card
        np.ndarray[DTYPE_t, ndim=2] delta_common_index
        np.ndarray[DTYPE_t, ndim=2] pattern_0, pattern_3
        np.ndarray[DTYPE_t, ndim=1] pattern_1
        np.ndarray[DTYPE_t, ndim=1] delta_non_common_index_y
        np.ndarray[DTYPE_t, ndim=1] non_common_x, non_common_y 
        unsigned int left_x, left_y
    
    common_index_y = np.array([index[1] for index in common_index], dtype=DTYPE)
    common_index_x = np.array([index[0] for index in common_index], dtype=DTYPE)
    cum_y_card =  np.cumprod(np.concatenate(([1], y_card[:-1])))
    delta_common_index = cum_y_card[common_index_y].reshape(
        (common_index_y.shape[0], 1))
    non_common_x = np.array([i for i in range(x_card.shape[0]) 
                             if i not in common_index_x], dtype=DTYPE)
    non_common_y = np.array([i for i in range(y_card.shape[0]) 
                             if i not in common_index_y], dtype=DTYPE)
    
    left_x = np.prod(x_card[non_common_x])
    left_y = np.prod(y_card[non_common_y])
    pattern_0 = np.tile(np.dot(matrix_gen(y_card[common_index_y]), 
                               delta_common_index).astype(DTYPE), 
                        (1, left_y)).astype(DTYPE)
    delta_non_common_index_y = cum_y_card[non_common_y]
    pattern_1 = np.dot(matrix_gen(y_card[non_common_y]), 
                       delta_non_common_index_y).astype(DTYPE)
    pattern_3 = np.tile(pattern_0 + pattern_1, (left_x, 1)).astype(DTYPE)
    return pattern_3.flatten(order='F'), left_y


@cython.boundscheck(False)
@cython.nonecheck(False)
def _factor_product(np.ndarray[DTYPE_t, ndim=1] card_prod,DTYPE_t size,
                    np.ndarray[double, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] card_x, np.ndarray[DTYPE_t,ndim=1] ref_x,
                    np.ndarray[double, ndim=1] y, np.ndarray[DTYPE_t, ndim=1] card_y, np.ndarray[DTYPE_t,ndim=1] ref_y):

    cdef:
        np.ndarray[double, ndim=1] product_arr = np.zeros(size)
        np.ndarray[DTYPE_t, ndim=1] prod_indices = np.array([0] * card_prod.shape[0], dtype=DTYPE)
        int i, x_index, y_index

    x_index = 0
    y_index = 0
    for i in range(size):
        product_arr[i] = x[x_index] * y[y_index]
        j=card_prod.shape[0]-1
        flag=0
        while True:
            old_value = prod_indices[j]
            prod_indices[j]+=1
            if prod_indices[j] == card_prod[j]:
                prod_indices[j] =0
            else:
                flag=1
            if ref_x[j] != -1:
                x_index += (prod_indices[j]-old_value) * card_x[ref_x[j]]
            if ref_y[j] != -1:
                y_index += (prod_indices[j]-old_value) * card_y[ref_y[j]]
            j-=1
            if flag==1:
                break
    return product_arr

@cython.boundscheck(False)
@cython.nonecheck(False)
def _factor_product_orig(np.ndarray[double, ndim=1] x,
                    np.ndarray[double, ndim=1] y,
                    DTYPE_t size,
                    np.ndarray[DTYPE_t, ndim=2] common_index=None,
                    np.ndarray[DTYPE_t, ndim=1] x_card=None,
                    np.ndarray[DTYPE_t, ndim=1] y_card=None):

    cdef:
        np.ndarray[double, ndim=1] product_arr = np.zeros(size)
        unsigned int count = 0
        unsigned int xmax = x.shape[0]
        unsigned int ymax = y.shape[0]
        unsigned int i, j, left_y
        np.ndarray[DTYPE_t, ndim=1] x_iter, y_iter

    cdef bint CHECK = 1
    if common_index is None and x_card is None and y_card is None:
        CHECK = 0

    if CHECK:
        y_iter, left_y = pattern_gen(x_card, y_card, common_index)
        x_iter = np.tile(np.arange(np.prod(x_card)), left_y).astype(DTYPE)
        for i in range(x_iter.shape[0]):
            print(x_iter[i],y_iter[i])
        for i, j in zip(x_iter, y_iter):
            product_arr[count] = x[i]*y[j]
            count += 1
    else:
        for i in range(ymax):
            for j in range(xmax):
                product_arr[count] = x[j]*y[i]
                count += 1

    return product_arr

@cython.boundscheck(False)
@cython.nonecheck(False)
def _factor_divide(np.ndarray[double, ndim=1] x,
                    np.ndarray[double, ndim=1] y,
                    DTYPE_t size,
                    np.ndarray[DTYPE_t, ndim=2] common_index=None,
                    np.ndarray[DTYPE_t, ndim=1] x_card=None,
                    np.ndarray[DTYPE_t, ndim=1] y_card=None):

    cdef:
        np.ndarray[double, ndim=1] product_arr = np.zeros(size)
        unsigned int count = 0
        unsigned int xmax = x.shape[0]
        unsigned int ymax = y.shape[0]
        unsigned int i, j, left_y
        np.ndarray[DTYPE_t, ndim=1] x_iter, y_iter


    y_iter, left_y = pattern_gen(x_card, y_card, common_index)
    x_iter = np.tile(np.arange(np.prod(x_card)), left_y).astype(DTYPE)
    for i, j in zip(x_iter, y_iter):
        product_arr[count] = x[i]/y[j]
        count += 1


    return product_arr