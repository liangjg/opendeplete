""" The Stochastic Implicit Euler algorithm.

This implements the SIE algorithm with nuclide relaxation.
"""

import copy
import os
import time

from mpi4py import MPI

from .cram import CRAM48
from .save_results import save_results

def sie_NR(operator, m=10, print_out=True):
    """ SIE with NR.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    m : Int
        Number of iterations.
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

    for i, dt in enumerate(operator.settings.dt_vec):
        # Create vectors
        x = [copy.deepcopy(vec)]
        seeds = []
        eigvls = []
        rates_array = []

        eigvl, rates, seed = operator.eval(x[0])

        eigvls.append(eigvl)
        seeds.append(seed)
        rates_array.append(copy.deepcopy(rates))

        # Create results, write to disk
        save_results(operator, x, rates_array, eigvls, seeds, [t, t + dt], i)

        x_result = []

        t_start = time.time()

        for mat in range(n_mats):
            f = operator.form_matrix(rates_array[0], mat)

            x_result.append(CRAM48(f, x[0][mat], dt))

        t_end = time.time()
        if MPI.COMM_WORLD.rank == 0:
            if print_out:
                print("Time to matexp: ", t_end - t_start)

        # Iterate on implicit partition
        x_result_avg = copy.deepcopy(x_result)

        for j in range(m):
            eigvl, rates, seed = operator.eval(x_result_avg)

            t_start = time.time()

            for mat in range(n_mats):
                f = operator.form_matrix(rates, mat)

                x_mat = CRAM48(f, x[0][mat], dt)

                # Average together
                x_result_avg[mat] += (x_mat - x_result_avg[mat]) / (j + 2)

            t_end = time.time()
            if MPI.COMM_WORLD.rank == 0:
                if print_out:
                    print("Time to matexp: ", t_end - t_start)

        t += dt
        vec = copy.deepcopy(x_result_avg)

    # Perform one last simulation
    x = [copy.deepcopy(vec)]
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
