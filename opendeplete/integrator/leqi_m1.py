""" The LE/QI integrator.

Implements the LE/QI Predictor-Corrector algorithm.

This algorithm is mathematically defined as:

.. math:
    y' = A(y, t) y(t)
    A_m1 = A(y_n-1, t_n-1)
    A_0 = A(y_n, t_n)
    A_l(t) linear extrapolation of A_m1, A_0
    Integrate to t_n+1 to get y_p
    A_c = A(y_p, y_n+1)
    A_q(t) quadratic interpolation of A_m1, A_0, A_c

It is initialized using the CE/LI algorithm.
"""

import copy
import os
import time

from mpi4py import MPI

from .celi_m1 import celi_m1_inner
from .cram import CRAM48
from .save_results import save_results

def leqi_m1(operator, m=5, print_out=True):
    """ Performs integration of an operator using the LE/QI M1 algorithm.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    m : Int, optional, default 5
        Number of substeps to perform
    print_out : bool, optional
        Whether or not to print out time.
    """

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = operator.initial_condition()

    n_mats = len(vec)

    t = 0.0

    # Perform single step of CE/LI M1
    dt_l = operator.settings.dt_vec[0]
    vec, t, rates_last = celi_m1_inner(operator, m, vec, 0, t, dt_l, print_out)

    # Perform remaining LE/QI
    for i, dt in enumerate(operator.settings.dt_vec[1::]):
        # Create vectors
        x = [copy.copy(vec)]
        seeds = []
        eigvls = []
        rates_array = []

        eigvl, rates, seed = operator.eval(x[0])

        eigvls.append(eigvl)
        seeds.append(seed)
        rates_array.append(copy.deepcopy(rates))

        x_result = []

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrices
            f1 = operator.form_matrix(rates_last, mat)
            f2 = operator.form_matrix(rates_array[0], mat)

            # Perform substepping
            x_new = copy.copy(x[0][mat])
            for j in range(m):
                a = j / m
                b = (j + 1) / m
                c1 = (a - b) * (a + b) * dt**2 / (2 * dt_l)
                c2 = -(a - b) * dt * ((a + b) * dt + 2 * dt_l) / (2 * dt_l)

                x_new = CRAM48(f1*c1 + f2*c2, x_new, 1.0)

            x_result.append(x_new)

        t_end = time.time()
        if MPI.COMM_WORLD.rank == 0:
            if print_out:
                print("Time to matexp: ", t_end - t_start)

        x.append(x_result)

        eigvl, rates, seed = operator.eval(x[1])

        eigvls.append(eigvl)
        seeds.append(seed)
        rates_array.append(copy.deepcopy(rates))

        x_result = []

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrices
            f1 = operator.form_matrix(rates_last, mat)
            f2 = operator.form_matrix(rates_array[0], mat)
            f3 = operator.form_matrix(rates_array[1], mat)

            # Perform substepping
            x_new = copy.copy(x[0][mat])
            for j in range(m):
                a = j / m
                b = (j + 1) / m
                c1 = (((3 - 2 * a) * a**2 + b**2 * (-3 + 2 * b)) * dt**3)/(6 * dt_l * (dt + dt_l))
                c2 = ((a - b) * dt * ((2 * a**2 + a * (-3 + 2 * b) + b * (-3 + 2 * b)) * dt + 3 * (-2 + a + b) * dt_l))/(6 * dt_l)
                c3 = (dt * (-2 * a**3 * dt - 3 * a**2 * dt_l + b**2 * (2 * b * dt + 3 * dt_l)))/(6 * (dt + dt_l))
                x_new = CRAM48(f1*c1 + f2*c2 + f3*c3, x_new, 1.0)

            x_result.append(x_new)

        t_end = time.time()
        if MPI.COMM_WORLD.rank == 0:
            if print_out:
                print("Time to matexp: ", t_end - t_start)

        # Create results, write to disk
        save_results(operator, x, rates_array, eigvls, seeds, [t, t + dt], i + 1)

        rates_last = copy.deepcopy(rates_array[0])
        t += dt
        dt_l = dt
        vec = copy.deepcopy(x_result)

    # Perform one last simulation
    x = [copy.copy(vec)]
    seeds = []
    eigvls = []
    rates_array = []
    eigvl, rates, seed = operator.eval(x[0])

    eigvls.append(eigvl)
    seeds.append(seed)
    rates_array.append(rates)

    # Create results, write to disk
    save_results(operator, x, rates_array, eigvls, seeds, [t, t],
                 len(operator.settings.dt_vec))

    # Return to origin
    os.chdir(dir_home)
