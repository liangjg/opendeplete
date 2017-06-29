""" The CE/LI M1 integrator.

Implements the CE/LI Predictor-Corrector algorithm using Magnus integrator.

This algorithm is mathematically defined as:

.. math:
    y' = A(y, t) y(t)
    A_p = A(y_n, t_n)
    y_p = expm(A_p h) y_n
    A_c = A(y_p, t_n)
    A(t) = t/dt * A_c + (dt - t)/dt * A_p

Here, A(t) is integrated by averaging A(t) over a substep and using the matrix
exponent.
"""

import copy
import os
import time

import numpy as np
from mpi4py import MPI

from .cram import CRAM48
from .save_results import save_results

def celi_m1(operator, m=5, print_out=True):
    """ Performs integration of an operator using the CE/LI M1 algorithm.

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

    t = 0.0

    for i, dt in enumerate(operator.settings.dt_vec):
        vec, t, _ = celi_m1_inner(operator, m, vec, i, t, dt, print_out)

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

def celi_m1_inner(operator, m, vec, iter_index, t, dt, print_out, renormalize_power=False):
    """ The inner loop of CE/LI M1.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    m : Int
        Number of substeps to perform
    vec : list of numpy.array
        Nuclide vector, beginning of time.
    iter_index : Int
        Current iteration number.
    t : Float
        Time at start of step.
    dt : Float
        Time step.
    print_out : bool
        Whether or not to print out time.
    renormalize_power : bool, optional
        Whether or not to renormalize power for each substep.

    Returns
    -------
    x_result : list of numpy.array
        Nuclide vector, end of time.
    Float
        Next time
    ReactionRates
        Reaction rates from beginning of step.
    """

    # Compute true energy deposition for time step
    if renormalize_power == True:
        edep_true = operator.settings.power * dt

    n_mats = len(vec)

    # Create vectors
    x = [copy.deepcopy(vec)]
    seeds = []
    eigvls = []
    rates_array = []

    eigvl, rates, seed = operator.eval(x[0])

    eigvls.append(eigvl)
    seeds.append(seed)
    rates_array.append(copy.deepcopy(rates))

    if renormalize_power:
        # Compute power deposition for all nuclides
        energy_array = np.zeros(n_mats)

        for mat in range(n_mats):
            # Form matrix
            f = operator.form_matrix(rates_array[0], mat, compute_energy=True)

            n_nuc = len(x[0][mat])
            x_i = np.zeros(n_nuc + 1)
            x_i[:-1] = copy.deepcopy(x[0][mat])

            x_new = CRAM48(f, x_i, dt)

            energy_array[mat] = x_new[-1]

        # Broadcast power
        total_energy = 0.0
        if MPI.COMM_WORLD.rank == 0:
            total_energy += np.sum(energy_array)
            for i in range(1, MPI.COMM_WORLD.size):
                total_energy += MPI.COMM_WORLD.recv(source=i, tag=i)
        else:
            MPI.COMM_WORLD.send(np.sum(energy_array), dest=0, tag=MPI.COMM_WORLD.rank)

        total_energy = MPI.COMM_WORLD.bcast(total_energy, root=0)

        scale = edep_true / total_energy

        if MPI.COMM_WORLD.rank == 0:
            print("Scaling parameter = ", scale)
    else: 
        scale = 1.0

    x_result = []

    t_start = time.time()
    for mat in range(n_mats):
        # Form matrix
        f = operator.form_matrix(rates_array[0], mat, scale=scale)

        x_new = CRAM48(f, x[0][mat], dt)

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

    x_result = copy.deepcopy(x[0])
    x_result_new = copy.deepcopy(x[0])

    t_start = time.time()
    for j in range(m):
        # Polynomial weights
        a = j / m
        b = (j + 1) / m
        c1 = 1/2 * (a - b) * (-2 + a + b) * dt
        c2 = 1/2 * (-a**2 + b**2) * dt

        # First, compute scaling parameter
        if renormalize_power:
            # Compute power deposition for all nuclides
            energy_array = np.zeros(n_mats)

            for mat in range(n_mats):
                # Form matrix
                f1 = operator.form_matrix(rates_array[0], mat, compute_energy=True)
                f2 = operator.form_matrix(rates_array[1], mat, compute_energy=True)

                n_nuc = len(x_result[mat])
                x_i = np.zeros(n_nuc + 1)
                x_i[:-1] = copy.deepcopy(x_result[mat])

                x_new = CRAM48(f1*c1 + f2*c2, x_i, 1.0)

                energy_array[mat] = x_new[-1]

            # Broadcast power
            total_energy = 0.0
            if MPI.COMM_WORLD.rank == 0:
                total_energy += np.sum(energy_array)
                for i in range(1, MPI.COMM_WORLD.size):
                    total_energy += MPI.COMM_WORLD.recv(source=i, tag=i)
            else:
                MPI.COMM_WORLD.send(np.sum(energy_array), dest=0, tag=MPI.COMM_WORLD.rank)

            total_energy = MPI.COMM_WORLD.bcast(total_energy, root=0)

            scale = edep_true / total_energy / m

            if MPI.COMM_WORLD.rank == 0:
                print("Scaling parameter = ", scale)
        else: 
            scale = 1.0

        for mat in range(n_mats):
            # Form matrix
            f1 = operator.form_matrix(rates_array[0], mat, scale=scale)
            f2 = operator.form_matrix(rates_array[1], mat, scale=scale)

            x_result_new[mat] = CRAM48(f1*c1 + f2*c2, x_result[mat], 1.0)

        x_result = copy.deepcopy(x_result_new)

    t_end = time.time()
    if MPI.COMM_WORLD.rank == 0:
        if print_out:
            print("Time to matexp: ", t_end - t_start)

    # Create results, write to disk
    save_results(operator, x, rates_array, eigvls, seeds, [t, t + dt], iter_index)

    return x_result, t + dt, rates_array[0]