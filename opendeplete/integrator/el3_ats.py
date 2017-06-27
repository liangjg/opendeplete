""" The EL3 integrator.

Implements the EL3 Predictor-Corrector algorithm.
"""

import copy
import os
import pickle
import time

import numpy as np
import scipy.special as special
from mpi4py import MPI

from .cram import CRAM48
from .save_results import save_results

def el3_ats(operator, tol, print_out=True):
    """ Performs integration of an operator using the EL3 algorithm.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    tol : float
        Tolerance.
    print_out : bool, optional
        Whether or not to print out time.
    """

    comm = MPI.COMM_WORLD

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = copy.deepcopy(operator.initial_condition())

    n_mats = len(vec)

    t = 0.0

    dt = 36000.0
    nps = int(250)

    n_steps = 10

    operator.settings.particles = nps

    t_final = sum(operator.settings.dt_vec)

    neutrons_total = 0.0
    rejected_steps = 0.0
    kept_steps = 0.0

    step_ind = 0

    while t < t_final * (1 - 1.0e-12):

        # Ensure no overruns
        if t + dt > t_final:
            dt = t_final - t

        # Avg/std storage
        rates_avg = []
        x_avg = []
        relative_diff_M = []
        relative_diff_S = []
        eigvl_avg = 0.0

        for i in range(n_steps):
            # Compute next step
            x_result, xhat_result, rates_array, eigvl = el3_inner(operator, vec, t, dt, print_out)

            relative_diff = compute_relative_diff(x_result, xhat_result)

            # Compress using Wellford's method
            if i == 0:
                rates_avg = copy.deepcopy(rates_array)
                x_avg = copy.deepcopy(x_result)
                relative_diff_M = copy.deepcopy(relative_diff)
                relative_diff_S = [np.zeros(len(relative_diff[mat])) for mat in range(n_mats)]
                eigvl_avg = eigvl
            else:
                rates_avg.rates += (rates_array.rates - rates_avg.rates) / (i + 1)
                eigvl_avg += (eigvl - eigvl_avg) / (i + 1)
                for mat in range(n_mats):
                    x_avg[mat] += (x_result[mat] - x_avg[mat]) / (i + 1)
                    old_M = copy.deepcopy(relative_diff_M[mat])
                    relative_diff_M[mat] += (relative_diff[mat] - relative_diff_M[mat]) / (i + 1)
                    relative_diff_S[mat] += (relative_diff[mat] - relative_diff_M[mat]) * (relative_diff[mat] - old_M)

        # Compute standard deviation of mean
        for mat in range(n_mats):
            relative_diff_S[mat] = np.sqrt(relative_diff_S[mat] / ((n_steps - 1) * n_steps) )

        # Compute / Broadcast RMSE of mean / standard deviation
        sum_M, sum_S, nuc_count = compute_sum_of_squares(relative_diff_M, relative_diff_S, x_avg, 1.0e6)

        success = False
        next_dt = 0.0
        next_nps = 0

        if comm.rank == 0:
            overall_sum_M = sum_M
            overall_sum_S = sum_S
            overall_nuc_count = nuc_count
            for i in range(1, comm.size):
                sum_M_i, sum_S_i, nuc_count_i = comm.recv(source=i, tag=i)
                overall_sum_M += sum_M_i
                overall_sum_S += sum_S_i
                overall_nuc_count += nuc_count_i

            mu = np.sqrt(overall_sum_M / overall_nuc_count)
            std = np.sqrt(overall_sum_S / overall_nuc_count)

            # Compute success or failure
            p_within = 1/2 * (-special.erf((mu - tol) / (np.sqrt(2) * std)) + special.erf((mu + tol) / (np.sqrt(2) * std)))

            c = 0.95
            theta_h = 0.5
            theta_n = 0.1
            ratio = 4.0

            if p_within > c:
                success = True
                kept_steps += 1
            else:
                rejected_steps += 1
            neutrons_total += n_steps * 3 * nps * operator.settings.batches

            # Compute new dt, new nps
            m = np.abs(mu) + np.sqrt(2) * special.erfinv(c) * std
            h_0 = dt * (theta_h * tol / m)**(1/3)
            n_0 = nps * h_0 / dt * (std / (theta_n * tol))**2

            next_dt = max(dt / ratio, min(dt * ratio, h_0))
            next_nps = max(100, max(nps / ratio, min(nps * ratio, n_0)))
            next_nps = int(next_nps)
        else:
            comm.send((sum_M, sum_S, nuc_count), dest=0, tag=comm.rank)
        
        # Broadcast failure state, new time step
        success = comm.bcast(success, root=0)
        next_dt = comm.bcast(next_dt, root=0)
        next_nps = comm.bcast(next_nps, root=0)

        if comm.rank == 0:
            print("Success?", success)
            print("dt?", next_dt)
            print("nps?", next_nps)
            print("mu?", mu)
            print("std?", std)
            print("total?", neutrons_total)

        if success:
            # Write to disk
            save_results(operator, [vec], [rates_avg], [eigvl_avg], [0], [t, t + dt], step_ind)

            t += dt
            dt = next_dt
            nps = next_nps
            operator.settings.particles = next_nps
            step_ind += 1
            vec = copy.deepcopy(x_avg)

    # Perform one last simulation
    operator.settings.particles *= n_steps
    x = [copy.deepcopy(vec)]
    seeds = []
    eigvls = []
    rates_array = []
    eigvl, rates, seed = operator.eval(x[0])

    eigvls.append(eigvl)
    seeds.append(seed)
    rates_array.append(rates)

    # Create results, write to disk
    save_results(operator, x, rates_array, eigvls, seeds, [t, t], step_ind)

    # Return to origin
    os.chdir(dir_home)

def compute_sum_of_squares(a, b, ref, cutoff):
    """ Computes the sum of squares of a list of np.arrays."""

    n_mats = len(a)

    sum_a = 0.0
    sum_b = 0.0
    count = 0

    for i in range(n_mats):
        mask = (ref[i] > cutoff)

        count += sum(mask)

        sum_a += np.sum(a[i][mask]**2)
        sum_b += np.sum(b[i][mask]**2)

    return sum_a, sum_b, count

def compute_relative_diff(a, b):
    """ Computes the relative difference between distribution a and b

    """

    n_mats = len(a)

    result = []

    for i in range(n_mats):
        x1 = copy.deepcopy(a[i])
        x2 = b[i]

        x1[x1 == 0.0] = 1.0e-24

        result.append((x1 - x2) / x1)

    return result

def el3_inner(operator, vec, t, dt, print_out):
    """ The inner loop of EL3.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    vec : list of numpy.array
        Nuclide vector, beginning of time.
    t : Float
        Time at start of step.
    dt : Float
        Time step.
    print_out : bool
        Whether or not to print out time.

    Returns
    -------
    x_result : list of numpy.array
        Nuclide vector, end of time, 3rd order
    xhat_result : list of numpy.array
        Nuclide vector, end of time, 2nd order
    ReactionRates
        Reaction rates from beginning of step.
    eigvl : float
        Eigenvalue from beginning of step.
    """

    n_mats = len(vec)

    # Coefficients
    d11 = 1.0
    d21 = 0.4917209077482205 
    d22 = 0.5082790922517794
    d31 = 0.0203785627902480
    d32 = 0.5023605249880540
    d33 = 0.4772609122216979
    a111 = 0.4546892735758679
    a211 = -0.0935787504499672
    a212 = 0.8796664339946539
    a221 = -0.5901222600691500
    a222 = 0.9215206700379688
    a311 = 0.2323861655916654
    a312 = 0.1815990852230668
    a313 = 0.5860147491852676
    a321 = 0.0110577763886061
    a322 = 0.0278227885012972
    a323 = 0.5064301615342287
    a331 = 0.0272124023499768
    a332 = -0.1076902478515720
    a333 = 0.2943901619569085

    # ATS
    d41 = 0.6990258593378940649862735967080976455200406842493
    d42 = 0
    d43 = 0.30097414066210593501372640329190235447995931575069
    a411 = 0.38971952731069785507790259853117777873507238351692
    a412 = 0.075578336488621953387346103161918515430029393849860
    a413 = 0.53470213391874912702171801144164378522046861152633
    a421 = 0.42570023566807507892515063485779185284196902787003
    a422 = 0.67976107351925317542648508250031744461617142972538
    a423 = -0.25516485456486948176696770982599954145462683120568
    a431 = 0.21151234903676088640583995486342472618548058958015
    a432 = -0.58451128882910535972747400089587563831460173004155
    a433 = 0.58691126154754437516922996630514434877577877433675

    x = [copy.deepcopy(vec)]
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
        # Form matrix
        f = operator.form_matrix(rates_array[0], mat)

        x_new = d11 * CRAM48(a111 * f, x[0][mat], dt)

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
        # Form matrix
        f0 = operator.form_matrix(rates_array[0], mat)
        f1 = operator.form_matrix(rates_array[1], mat)

        x_new = (d21 * CRAM48(a211 * f0 + a212 * f1, x[0][mat], dt) 
                 + d22 * CRAM48(a221 * f0 + a222 * f1, x[1][mat], dt))

        x_result.append(x_new)

    t_end = time.time()
    if MPI.COMM_WORLD.rank == 0:
        if print_out:
            print("Time to matexp: ", t_end - t_start)

    x.append(x_result)

    eigvl, rates, seed = operator.eval(x[2])

    eigvls.append(eigvl)
    seeds.append(seed)
    rates_array.append(copy.deepcopy(rates))

    x_result = []
    xhat_result = []

    t_start = time.time()
    for mat in range(n_mats):
        # Form matrix
        f0 = operator.form_matrix(rates_array[0], mat)
        f1 = operator.form_matrix(rates_array[1], mat)
        f2 = operator.form_matrix(rates_array[2], mat)

        x_new = (d31 * CRAM48(a311 * f0 + a312 * f1 + a313 * f2, x[0][mat], dt) 
                 + d32 * CRAM48(a321 * f0 + a322 * f1 + a323 * f2, x[1][mat], dt)
                 + d33 * CRAM48(a331 * f0 + a332 * f1 + a333 * f2, x[2][mat], dt) )

        xhat_new = (d41 * CRAM48(a411 * f0 + a412 * f1 + a413 * f2, x[0][mat], dt) 
                 + d42 * CRAM48(a421 * f0 + a422 * f1 + a423 * f2, x[1][mat], dt)
                 + d43 * CRAM48(a431 * f0 + a432 * f1 + a433 * f2, x[2][mat], dt) )

        x_result.append(x_new)
        xhat_result.append(xhat_new)

    t_end = time.time()
    if MPI.COMM_WORLD.rank == 0:
        if print_out:
            print("Time to matexp: ", t_end - t_start)

    return x_result, xhat_result, rates_array[0], eigvls[0]