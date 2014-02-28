cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t


@cython.boundscheck(False)
@cython.nonecheck(False)
cdef assignment(DTYPE_t index, np.ndarray[DTYPE_t, ndim=1] card):
    return np.mod(np.floor(np.divide(np.tile(index, card.shape[0]),
                  np.cumprod(np.concatenate(([1], card[:-1]))))), card)


@cython.boundscheck(False)
@cython.nonecheck(False)
cdef assignment_match(DTYPE_t x_index, DTYPE_t y_index,
                      np.ndarray[DTYPE_t, ndim=2] common_index,
                      np.ndarray[DTYPE_t, ndim=1] x_card,
                      np.ndarray[DTYPE_t, ndim=1] y_card):
    for index in common_index:
        if assignment(x_index, x_card)[index[0]] != assignment(y_index, y_card)[index[1]]:
            return False
    return True


@cython.boundscheck(False)
@cython.nonecheck(False)
def _factor_product(np.ndarray[double, ndim=1] x,
                    np.ndarray[double, ndim=1] y,
                    DTYPE_t size,
                    np.ndarray[DTYPE_t, ndim=2] common_index=None,
                    np.ndarray[DTYPE_t, ndim=1] x_card=None,
                    np.ndarray[DTYPE_t, ndim=1] y_card=None):

    cdef np.ndarray[double, ndim = 1] product_arr = np.zeros(size)
    cdef unsigned int count = 0
    cdef unsigned int xmax = x.shape[0]
    cdef unsigned int ymax = y.shape[0]
    cdef unsigned int i, j

    cdef bint CHECK = 1
    if common_index is None and x_card is None and y_card is None:
        CHECK = 0

    if CHECK:
        for i in range(ymax):
            for j in range(xmax):
                if assignment_match(j, i, common_index, x_card, y_card):
                    product_arr[count] = x[j]*y[i]
                    count += 1
    else:
        for i in range(ymax):
            for j in range(xmax):
                product_arr[count] = x[j]*y[i]
                count += 1

    return product_arr
