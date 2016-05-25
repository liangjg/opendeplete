"""Zernike module.

Utility class for zernike polynomials
"""

import numpy as np
from math import pi

# b_matrix and c_matrix are assuming 
b_matrix = np.zeros((3,3,3))
b_matrix[0,0,0]=3.14159265358979
b_matrix[0,1,1]=0.785398163397448
b_matrix[0,2,2]=0.785398163397448
b_matrix[1,0,1]=0.785398163397448
b_matrix[1,1,0]=0.785398163397448
b_matrix[2,0,2]=0.785398163397448
b_matrix[2,2,0]=0.785398163397448

c_matrix = np.zeros(3)
c_matrix[0]=3.14159265358979
c_matrix[1]=0.785398163397448
c_matrix[2]=0.785398163397448

import numpy as np

def num_poly(n):
    return int(1/2 * (n+1) * (n+2))

def zern_to_ind(n,m):
    ind = num_poly(n-1)
    ind += (m + n) // 2
    return int(ind)

def form_b_matrix(p, pp, rate):
    # Yields the sum
    # ret = sum_r Int[P_p * P_pp * P_r * rate_r] / Int[P_pp^2]

    order = len(rate)

    v1 = b_matrix[p,pp,0:order]
    return np.dot(v1, rate)/c_matrix[pp]
