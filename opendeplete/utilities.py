""" The utilities module.

Contains functions that can be used to post-process objects that come out of
the results module.
"""

import copy
import gc
import os

import numpy as np
from .results import read_results

def evaluate_single_nuclide(results, cell, nuc):
    """ Evaluates a single nuclide in a single cell from a results list.

    Parameters
    ----------
    results : list of results
        The results to extract data from.  Must be sorted and continuous.
    cell : str
        Cell name to evaluate
    nuc : str
        Nuclide name to evaluate

    Returns
    -------
    time : numpy.array
        Time vector.
    concentration : numpy.array
        Total number of atoms in the cell.
    """

    n_points = len(results)
    time = np.zeros(n_points)
    concentration = np.zeros(n_points)

    # Evaluate value in each region
    for i, result in enumerate(results):
        time[i] = result.time[0]
        concentration[i] = result[0, cell, nuc]

    return time, concentration

def evaluate_reaction_rate(results, cell, nuc, rxn):
    """ Evaluates a single nuclide reaction rate in a single cell from a results list.

    Parameters
    ----------
    results : list of Results
        The results to extract data from.  Must be sorted and continuous.
    cell : str
        Cell name to evaluate
    nuc : str
        Nuclide name to evaluate
    rxn : str
        Reaction rate to evaluate

    Returns
    -------
    time : numpy.array
        Time vector.
    rate : numpy.array
        Reaction rate.
    """

    n_points = len(results)
    time = np.zeros(n_points)
    rate = np.zeros(n_points)
    # Evaluate value in each region
    for i, result in enumerate(results):
        time[i] = result.time[0]
        rate[i] = result.rates[0][cell, nuc, rxn] * result[0, cell, nuc]

    return time, rate

def evaluate_eigenvalue(results):
    """ Evaluates the eigenvalue from a results list.

    Parameters
    ----------
    results : list of Results
        The results to extract data from.  Must be sorted and continuous.

    Returns
    -------
    time : numpy.array
        Time vector.
    eigenvalue : numpy.array
        Eigenvalue.
    """

    n_points = len(results)
    time = np.zeros(n_points)
    eigenvalue = np.zeros(n_points)

    # Evaluate value in each region
    for i, result in enumerate(results):

        time[i] = result.time[0]
        eigenvalue[i] = result.k[0]

    return time, eigenvalue

def load_all(folder):
    """ Iterates through all results in a folder to compute results.

    Parameters
    ----------
    folder : string
        Location of results.h5 files

    Returns
    -------

    """

    # Get results count
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".h5")]
    n_results = len(files)
    n_steps = 0
    nuc_nuc_to_ind = []
    nuc_mat_to_ind = []
    rate_nuc_to_ind = []
    rate_mat_to_ind = []
    rate_react_to_ind = []

    ev_array = []
    nuc_array = []
    rate_array = []
    time = []

    for i, file in enumerate(files):
        result = read_results(file)
        # Allocate
        if n_steps == 0:
            n_steps = len(result)
            nuc_nuc_to_ind = copy.deepcopy(result[0].nuc_to_ind)
            nuc_mat_to_ind = copy.deepcopy(result[0].mat_to_ind)
            rate_nuc_to_ind = copy.deepcopy(result[0].rates[0].nuc_to_ind)
            rate_mat_to_ind = copy.deepcopy(result[0].rates[0].mat_to_ind)
            rate_react_to_ind = copy.deepcopy(result[0].rates[0].react_to_ind)

            n_mat = len(nuc_mat_to_ind)
            n_nuc = len(nuc_nuc_to_ind)

            nr_mat = len(rate_mat_to_ind)
            nr_nuc = len(rate_nuc_to_ind)
            nr_rate = len(rate_react_to_ind)

            ev_array = np.zeros((n_results, n_steps))
            nuc_array = np.zeros((n_results, n_steps, n_mat, n_nuc))
            rate_array = np.zeros((n_results, n_steps, nr_mat, nr_nuc, nr_rate))

        # Read data
        time, ev = evaluate_eigenvalue(result)
        if len(ev) > n_steps:
            ev_array[i, :] = ev[:n_steps]
        else:
            ev_array[i, :len(ev)] = ev
        for mat, j in nuc_mat_to_ind.items():
            for nuc, k in nuc_nuc_to_ind.items():
                time, n = evaluate_single_nuclide(result, mat, nuc)
                if len(n) > n_steps:
                    nuc_array[i, :, j, k] = n[:n_steps]
                else:
                    nuc_array[i, :len(n), j, k] = n
        for mat, j in rate_mat_to_ind.items():
            for nuc, k in rate_nuc_to_ind.items():
                for react, l in rate_react_to_ind.items():
                    time, r = evaluate_reaction_rate(result, mat, nuc, react)
                    if len(r) > n_steps:
                        rate_array[i, :, j, k, l] = r[:n_steps]
                    else:
                        rate_array[i, :len(r), j, k, l] = r
        del result
        gc.collect()

    return time, nuc_nuc_to_ind, nuc_mat_to_ind, rate_nuc_to_ind, rate_mat_to_ind, rate_react_to_ind, ev_array, nuc_array, rate_array

