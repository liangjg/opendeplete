""" The EL3 integrator.

Implements the EL3 Predictor-Corrector algorithm.
"""

import copy
import os
import time

from mpi4py import MPI
import numpy as np

from .cram import CRAM48
from .save_results import save_results

def el3_stab(operator, print_out=True):
    """ Performs integration of an operator using the EL3 algorithm.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    print_out : bool, optional
        Whether or not to print out time.
    """

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

    c1 = a111
    c2 = a211 + a212
    c3 = a311 + a312 + a313

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = operator.initial_condition()

    n_mats = len(vec)

    ind_xe = operator.number.nuc_to_ind["Xe135"]
    ind_I = operator.number.nuc_to_ind["I135"]

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

            # Get inline Xenon 135, Iodine 135
            x_xe = CRAM48(f0, x[0][mat], dt * c2)

            x_new[ind_xe] = x_xe[ind_xe]
            x_new[ind_I] = x_xe[ind_I]

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

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrix
            f0 = operator.form_matrix(rates_array[0], mat)
            f1 = operator.form_matrix(rates_array[1], mat)
            f2 = operator.form_matrix(rates_array[2], mat)

            x_new = (d31 * CRAM48(a311 * f0 + a312 * f1 + a313 * f2, x[0][mat], dt) 
                     + d32 * CRAM48(a321 * f0 + a322 * f1 + a323 * f2, x[1][mat], dt)
                     + d33 * CRAM48(a331 * f0 + a332 * f1 + a333 * f2, x[2][mat], dt) )

            # Get inline Xenon 135, Iodine 135
            x_xe = CRAM48(f0, x[0][mat], dt * c3)

            x_new[ind_xe] = x_xe[ind_xe]
            x_new[ind_I] = x_xe[ind_I]

            x_result.append(x_new)

        t_end = time.time()
        if MPI.COMM_WORLD.rank == 0:
            if print_out:
                print("Time to matexp: ", t_end - t_start)

        # Create results, write to disk
        save_results(operator, x, rates_array, eigvls, seeds, [t, t + dt], i)

        t += dt
        vec = copy.deepcopy(x_result)

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

def xe_i_eq(x, f, ind_I, ind_Xe):
    # Compute sources

    x_no_IXe = copy.deepcopy(x)
    x_no_IXe[ind_I] = 0.0
    x_no_IXe[ind_Xe] = 0.0

    gen = f * x_no_IXe

    s = np.zeros((2))
    s[0] = gen[ind_I]
    s[1] = gen[ind_Xe]

    # Matrix components
    m = np.zeros((2,2))
    m[0, 0] = f[ind_I, ind_I]
    m[0, 1] = f[ind_I, ind_Xe]
    m[1, 0] = f[ind_Xe, ind_I]
    m[1, 1] = f[ind_Xe, ind_Xe]

    print(m)

    # Now we have m v + s = 0
    # v = -m^{-1}s
    v = np.linalg.solve(m, -s)

    return v[0], v[1]

def el3_stab2(operator, print_out=True):
    """ Performs integration of an operator using the EL3 algorithm.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    print_out : bool, optional
        Whether or not to print out time.
    """

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

    c1 = a111
    c2 = a211 + a212
    c3 = a311 + a312 + a313

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = operator.initial_condition()

    n_mats = len(vec)

    ind_xe = operator.number.nuc_to_ind["Xe135"]
    ind_I = operator.number.nuc_to_ind["I135"]

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

        x_result = []

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrix
            f = operator.form_matrix(rates_array[0], mat)

            x_new = d11 * CRAM48(a111 * f, x[0][mat], dt)

            # Enforce Xe, I equilibrium
            x_new[ind_I], x_new[ind_xe] = xe_i_eq(x_new, f, ind_I, ind_xe)

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

            # Enforce Xe, I equilibrium
            x_new[ind_I], x_new[ind_xe] = xe_i_eq(x_new, f0, ind_I, ind_xe)

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

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrix
            f0 = operator.form_matrix(rates_array[0], mat)
            f1 = operator.form_matrix(rates_array[1], mat)
            f2 = operator.form_matrix(rates_array[2], mat)

            x_new = (d31 * CRAM48(a311 * f0 + a312 * f1 + a313 * f2, x[0][mat], dt) 
                     + d32 * CRAM48(a321 * f0 + a322 * f1 + a323 * f2, x[1][mat], dt)
                     + d33 * CRAM48(a331 * f0 + a332 * f1 + a333 * f2, x[2][mat], dt) )

            # Enforce Xe, I equilibrium
            x_new[ind_I], x_new[ind_xe] = xe_i_eq(x_new, f0, ind_I, ind_xe)

            x_result.append(x_new)

        t_end = time.time()
        if MPI.COMM_WORLD.rank == 0:
            if print_out:
                print("Time to matexp: ", t_end - t_start)

        # Create results, write to disk
        save_results(operator, x, rates_array, eigvls, seeds, [t, t + dt], i)

        t += dt
        vec = copy.deepcopy(x_result)

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

def el3_stab3(operator, print_out=True):
    """ Performs integration of an operator using the EL3 algorithm.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    print_out : bool, optional
        Whether or not to print out time.
    """

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

    c1 = a111
    c2 = a211 + a212
    c3 = a311 + a312 + a313

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = operator.initial_condition()

    n_mats = len(vec)

    ind_xe = operator.number.nuc_to_ind["Xe135"]
    ind_I = operator.number.nuc_to_ind["I135"]

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

        x_result = []

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrix
            f = operator.form_matrix(rates_array[0], mat)

            x_new = d11 * CRAM48(a111 * f, x[0][mat], dt)

            # Enforce Xe, I equilibrium
            x_new[ind_I], x_new[ind_xe] = xe_i_eq(x[0][mat], f, ind_I, ind_xe)

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

            # Enforce Xe, I equilibrium
            i1, x1 = xe_i_eq(x[0][mat], f0, ind_I, ind_xe)
            i2, x2 = xe_i_eq(x[1][mat], f1, ind_I, ind_xe)

            x_new[ind_I] = (i1 + i2)/2
            x_new[ind_xe] = (x1 + x2)/2

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

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrix
            f0 = operator.form_matrix(rates_array[0], mat)
            f1 = operator.form_matrix(rates_array[1], mat)
            f2 = operator.form_matrix(rates_array[2], mat)

            x_new = (d31 * CRAM48(a311 * f0 + a312 * f1 + a313 * f2, x[0][mat], dt) 
                     + d32 * CRAM48(a321 * f0 + a322 * f1 + a323 * f2, x[1][mat], dt)
                     + d33 * CRAM48(a331 * f0 + a332 * f1 + a333 * f2, x[2][mat], dt) )

            # Enforce Xe, I equilibrium
            i1, x1 = xe_i_eq(x[0][mat], f0, ind_I, ind_xe)
            i2, x2 = xe_i_eq(x[1][mat], f1, ind_I, ind_xe)
            i3, x3 = xe_i_eq(x[2][mat], f2, ind_I, ind_xe)

            x_new[ind_I] = (i1 + i2 + i3)/3
            x_new[ind_xe] = (x1 + x2 + x3)/3

            x_result.append(x_new)

        t_end = time.time()
        if MPI.COMM_WORLD.rank == 0:
            if print_out:
                print("Time to matexp: ", t_end - t_start)

        # Create results, write to disk
        save_results(operator, x, rates_array, eigvls, seeds, [t, t + dt], i)

        t += dt
        vec = copy.deepcopy(x_result)

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

def el3_stab4(operator, print_out=True):
    """ Performs integration of an operator using the EL3 algorithm.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    print_out : bool, optional
        Whether or not to print out time.
    """

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

    c1 = a111
    c2 = a211 + a212
    c3 = a311 + a312 + a313

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = operator.initial_condition()

    n_mats = len(vec)

    ind_xe = operator.number.nuc_to_ind["Xe135"]
    ind_I = operator.number.nuc_to_ind["I135"]

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

            # Get inline Xenon 135, Iodine 135
            x_xe = CRAM48((f0 + f1)/2, x[0][mat], dt * c2)

            x_new[ind_xe] = x_xe[ind_xe]
            x_new[ind_I] = x_xe[ind_I]

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

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrix
            f0 = operator.form_matrix(rates_array[0], mat)
            f1 = operator.form_matrix(rates_array[1], mat)
            f2 = operator.form_matrix(rates_array[2], mat)

            x_new = (d31 * CRAM48(a311 * f0 + a312 * f1 + a313 * f2, x[0][mat], dt) 
                     + d32 * CRAM48(a321 * f0 + a322 * f1 + a323 * f2, x[1][mat], dt)
                     + d33 * CRAM48(a331 * f0 + a332 * f1 + a333 * f2, x[2][mat], dt) )

            # Get inline Xenon 135, Iodine 135
            x_xe = CRAM48((f0 + f1 + f2)/3, x[0][mat], dt * c3)

            x_new[ind_xe] = x_xe[ind_xe]
            x_new[ind_I] = x_xe[ind_I]

            x_result.append(x_new)

        t_end = time.time()
        if MPI.COMM_WORLD.rank == 0:
            if print_out:
                print("Time to matexp: ", t_end - t_start)

        # Create results, write to disk
        save_results(operator, x, rates_array, eigvls, seeds, [t, t + dt], i)

        t += dt
        vec = copy.deepcopy(x_result)

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