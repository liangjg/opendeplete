"""Zernike module.

Utility class for zernike polynomials
"""

import numpy as np
from math import pi
import numpy as np
from zerndata import b_matrix, c_matrix

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

class ZernikePolynomial:
    ''' ZernikePolynomial class
    
    This class contains the data that fully describes a Zernike polynomial
    and supporting functions.

    Attributes
    ----------
    order : int
        The maximum order of the polynomial
    radial_norm : flaot
        The radial normalization factor
    name : str
        The name of the polynomial
    n_coeffs : int
        The number of coefficients for a given order
    coeffs : List[flaot]
        The coefficients of the polynomial ordered as (m,n) 
        (0,0) (-1,1) (1,1) (-2,2) (0,2) (2,2) ...
    p_coeffs : List[float]
        Precomputed polynomial coefficients so that factorials
        do not need to be evaluated for funciton evaluation.
    '''
    
    import zernint as zni
    
    def __init__(self, order, coeffs, radial_norm=1.0, sqrt_normed=False):
        self._order = order
        self._radial_norm = radial_norm
        self._name = ''
        self._coeffs = coeffs
        self._n_coeffs = int(1/2 * (order+1) * (order+2))
        self._sqrt_normed = sqrt_normed
        self._p_coeffs = self.precompute_zn_coeffs()
        
    @property
    def order(self):
        return self._order

    @property
    def radial_norm(self):
        return self._radial_norm
    
    @property
    def coeffs(self):
        return self._coeffs

    @property
    def n_coeffs(self):
        return self._n_coeffs
    
    @property
    def name(self):
        return self._name

    @property
    def sqrt_normed(self):
        return self._sqrt_normed

    @order.setter
    def order(self, order):
        self._order = order

    @coeffs.setter
    def coeffs(self, coeffs):
        self._coeffs = coeffs

    @n_coeffs.setter
    def n_coeffs(self, n_coeffs):
        self._n_coeffs = n_coeffs
            
    @radial_norm.setter
    def radial_norm(self, radial_norm):
        self._radial_norm = radial_norm
        
    @name.setter
    def name(self, name):
        self._name = name

    @sqrt_normed.setter
    def sqrt_normed(self, sqrt_normed):
        self._sqrt_normed = sqrt_normed


    def order_to_index(self, n, m):
        ''' Returns the index for accessing coefficients based 
            on the (n,m) values.  
        '''
        return int(1/2 * (n) * (n+1))  + (m + n) // 2

    def precompute_zn_coeffs(self):
        ''' Precompute the zernike polynomial leading coefficeints

            Note that all FETs in OpenMC and reconstruction in
            MOOSE assume that the square root of the normalization
            constant is included.  These are embedded in poly_coeffs.
        '''

        poly_coeffs = []

        for n in range(0,(self.order+1)):
            for m in range(-n,(n+1),2):
                for s in range(0,(n-abs(m))//2+1):
                    poly_coeffs.append(self.R_m_n_s(n,m,s,self.radial_norm))

        return poly_coeffs

    def get_poly_value(self, r, theta):
        ''' Compute the value of a polynomial at a point.

            Note that the precomputed leading coefficients,
            p_coeffs[], assuems that the square root of the
            normalization constant is included in the coefficients
            coeffs[]

            Parameters
            ----------
            r : float
                 The radial point.  Not normalizated.
            theta : float
                 The theta value in radians.

        '''

        import math
        
        p_coeff_num = 0
        val = 0.0

        for n in range(0,(self.order+1)):
            for m in range(-n,(n+1)):
                if ((n-m) % 2 == 0):

                    if( (n-m) % 2 == 0 and m == 0 ):
                        azim_factor = 1.0
                    elif ( (n-m) % 2 == 0 and m < 0):
                        azim_factor = math.sin(abs(m) * theta)
                    elif ( (n-m) % 2 == 0 and m > 0):
                        azim_factor = math.cos(m * theta)
                    else:
                        print("n = "+str(n)+", m = "+str(m))
                        print("Invalid value of m and n");

                    for s in range(0,(n-abs(m))//2+1):
                        val = val + self.coeffs[self.order_to_index(n,m)] * \
                              self._p_coeffs[p_coeff_num] * \
                              (r/self.radial_norm)**(int(n-2*s)) * \
                              azim_factor
                        p_coeff_num = p_coeff_num + 1

        return val

    def compute_integral(self, r_min, r_max, theta_min, theta_max):
        ''' Compute the integral of the zernike polynomial over some 
        subset of the unit disk

        Note that the normalization factor is included because
        it is assumed that the polynomials coefficients include
        the square root of the normalization constants but
        integrate_wedge() does not use this multiplicate constant.

        Parameters
        ----------
        r_min : float
            The inner radius
        r_max : float
            The outer radius
        theta_min : float
            The minimum theta value
        theta_max : float
            The maxmium theta value
        '''

        import math
        import zernint as zni
        import sys

        val = 0.0

        for n in range(0, self.order+1):
            for m in range(-n,(n+1),2):
                norm_factor = self.get_norm_factor(n,m)
                val += self.coeffs[self.order_to_index(n,m)] * norm_factor * \
                       zni.integrate_wedge(n,m,r_min,r_max,theta_min, theta_max)

        return val

    def plot_disk(self, n_rings, n_sectors, fname):
        ''' This function plots the volume averaged value of the
        Zernike polynonmials in radial rings and azimuthal
        sectors.

        Parameters
        ----------
        n_rings : int
             The number of rings to split the disk into.
        n_sectors : int
             The number of azimuthal sectors in each ring.
        fname : str
             The name of the file into which to save the plot.
        
        Returns
        -------
        patch_vals : List
            List of patch values used in plot.
        '''

        import math
        import numpy
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Wedge, Polygon
        from matplotlib.collections import PatchCollection
        
        vol_per_ring = self.radial_norm * self.radial_norm * math.pi / n_rings       
        ring_radii = [0.0]
        for i in range(0, n_rings):
            ring_radii.append( math.sqrt(vol_per_ring / math.pi \
                            + ring_radii[-1] * ring_radii[-1]) )

        theta_cuts = [0.0]
        for i in range(0, n_sectors):
            theta_cuts.append(theta_cuts[-1] + \
                              math.pi * 2.0 / n_sectors)

        patches = []
        patch_vals = []
        for i in range(0, n_rings):
            for j in range(0, n_sectors):
                
                thickness = ring_radii[i+1] - ring_radii[i]
                if(j == n_sectors-1):  # Loop around the sectors
                    patch_vals.append(self.compute_integral(ring_radii[i]/self.radial_norm, ring_radii[i+1]/self.radial_norm,\
                                                            theta_cuts[j], 2*math.pi))
                    wedge = Wedge( (0.0,0.0), ring_radii[i+1], theta_cuts[j]*180.0/math.pi, theta_cuts[0]*180.0/math.pi, width=thickness)
                else:
                    patch_vals.append(self.compute_integral(ring_radii[i]/self.radial_norm, ring_radii[i+1]/self.radial_norm,\
                                                            theta_cuts[j], theta_cuts[j+1]))
                    wedge = Wedge( (0.0,0.0), ring_radii[i+1], theta_cuts[j]*180.0/math.pi, theta_cuts[j+1]*180.0/math.pi, width=thickness)
                patches.append(wedge)
                
        fig, ax = plt.subplots()
        ax.set_xlim([-self.radial_norm,self.radial_norm])
        ax.set_ylim([-self.radial_norm,self.radial_norm])
        p = PatchCollection(patches, cmap=plt.cm.jet)
        p.set_array(numpy.array(patch_vals))
        ax.add_collection(p)
        plt.colorbar(p)
        fig.savefig(fname)
        plt.close()

        return patch_vals

    def plot_over_line(self, theta, fname):
        ''' This funciton plots the polynomial over the entire disk
            along a specific theta value

        Parameters
        ----------
        theta : float
             The theta value over which to plot from [0, R_max]
        fname : str
             The filename to use when saving the figure
        
        '''

        import matplotlib.pyplot as plt

        # We will use a linearly spaced set of points for now to plot
        r_vals = np.linspace(0.0, self.radial_norm, num=1000)
        vals = []

        for r in r_vals:
            vals.append(self.get_poly_value(r,theta))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(r_vals, vals)
        fig.savefig(fname)
        plt.close()

    def remove_fet_sqrt_normalization(self):
        ''' This function removes the sqrt(2(n+1)) or sqrt(n+1)
        normalization that is applied in OpenMC FETs.
        '''
        import math

        for n in range(0, self.order+1):
            for m in range(-n,(n+1),2):
                if (m == 0):
                    self.coeffs[self.order_to_index(n,m)] *= math.sqrt(n+1.0)
                else:
                    self.coeffs[self.order_to_index(n,m)] *= math.sqrt(2.0*n+2.0)

        self.sqrt_normed = False
        # Since we might have changed the normalization state, we need
        # to recompute the precomputed polynomial coefficients
        self._p_coeffs = self.precompute_zn_coeffs()

    def normalize_coefficients(self):
        ''' This function normalizes coefficients by (n+1) / pi or
        (2n+2) / pi.
        '''
        import math

        for n in range(0, self.order+1):
            for m in range(-n,(n+1),2):
                if (m == 0):
                    self.coeffs[self.order_to_index(n,m)] *= (n+1.0)
                else:
                    self.coeffs[self.order_to_index(n,m)] *= (2.0*n+2.0)

                self.coeffs[self.order_to_index(n,m)] /= math.pi

        self.sqrt_normed = False
        # Since we might have changed the normalization state, we need
        # to recompute the precomputed polynomial coefficients
        self._p_coeffs = self.precompute_zn_coeffs()

    def scale_coefficients(self, scale_value):
        ''' This function scales every coefficient in the expansion
        by a given value

        Parameters
        ----------
        scale_value : float
             The scaling value to apply
        '''

        for n in range(0, self.order+1):
            for m in range(-n,(n+1),2):
                self.coeffs[self.order_to_index(n,m)] *= scale_value


    def get_norm_factor(self, n, m):
        ''' This function determines the normalization factor
        to be applied

        Parameters
        ----------
        n : int
             The radial moment number
        m : int
             The azimuthal moment number
        '''
        import math
        
        if (m == 0 and self.sqrt_normed):
            return (1.0 / math.pi * math.sqrt(n + 1.0))
        elif (m != 0 and self.sqrt_normed):
            return (1.0 / math.pi * math.sqrt(2.0 * n + 2.0))
        elif (not self.sqrt_normed):
            return 1.0
        else:
            sys.exit("Invalid state when calculating normalization factor")

    def R_m_n_s(self, n, m, s, r_max=1.0):
        ''' This function calculates the R_{n,m}(r)
        coefficients.  Note that this funciton does not check if
        n, m, or s are valid.

        Parameters
        ----------
        n : int
             The azimuthal moment number
        m : int
             The azimuthal moment number
        s : int
             One of the summation terms that defines R_{n,m}
             s is defined as [0,(n-abs(m))/2]
        r_max : float
             The maximum radius of the disk

        '''

        import math

        norm_factor = self.get_norm_factor(n,m)

        return (1.0 / (r_max * r_max) * math.sqrt(norm_factor) * \
                math.pow(-1,s) * math.factorial(n-s) / \
                ( math.factorial(s) * math.factorial((n+m)//2 - s) * \
                  math.factorial( (n-m)//2 - s) ) )
