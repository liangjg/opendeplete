"""ReactionRates module.

Just contains a dictionary of np.arrays to store reaction rates.
"""

import numpy as np
import zernike


class ReactionRates:
    """ The Nuclide class.

    Contains everything in a depletion chain relating to a single nuclide.

    Parameters
    ----------
    cell_to_ind : OrderedDict[int]
        A dictionary mapping cell ID as string to index.
    nuc_to_ind : OrderedDict[int]
        A dictionary mapping nuclide name as string to index.
    react_to_ind : OrderedDict[int]
        A dictionary mapping reaction name as string to index.

    Attributes
    ----------
    cell_to_ind : OrderedDict[int]
        A dictionary mapping cell ID as string to index.
    nuc_to_ind : OrderedDict[int]
        A dictionary mapping nuclide name as string to index.
    react_to_ind : OrderedDict[int]
        A dictionary mapping reaction name as string to index.
    n_cell : int
        Number of cells.
    n_nuc : int
        Number of nucs.
    n_react : int
        Number of reactions.
    rates : np.array
        Array storing rates indexed by the above dictionaries.
    """

    def __init__(self, cell_to_ind, nuc_to_ind, react_to_ind, max_poly_order):

        self.cell_to_ind = cell_to_ind
        self.nuc_to_ind = nuc_to_ind
        self.react_to_ind = react_to_ind

        self.n_cell = len(cell_to_ind)
        self.n_nuc = len(nuc_to_ind)
        self.n_react = len(react_to_ind)
        self.max_poly_order = max_poly_order
        self.n_poly = zernike.num_poly(max_poly_order)

        self.rates = np.zeros((self.n_cell, self.n_nuc, self.n_react, self.n_poly))

    def __getitem__(self, pos):
        """ Retrieves an item from reaction_rates.

        Parameters
        ----------
        pos : Tuple
            A four-length tuple containing a cell index, a nuc index, a
            reaction index, and a polynomial index.  These indexes can be
            strings (which get converted to integers via the dictionaries),
            integers used directly, or slices.

        Returns
        -------
        np.array
            The value indexed from self.rates.
        """

        cell, nuc, react, poly = pos
        if isinstance(cell, str):
            cell_id = self.cell_to_ind[cell]
        else:
            cell_id = cell
        if isinstance(nuc, str):
            nuc_id = self.nuc_to_ind[nuc]
        else:
            nuc_id = cell
        if isinstance(react, str):
            react_id = self.react_to_ind[react]
        else:
            react_id = react

        return self.rates[cell_id, nuc_id, react_id, poly]

    def __setitem__(self, pos, val):
        """ Sets an item from reaction_rates.

        Parameters
        ----------
        pos : Tuple
            A four-length tuple containing a cell index, a nuc index, a
            reaction index, and a polynomial index.  These indexes can be
            strings (which get converted to integers via the dictionaries),
            integers used directly, or slices.
        val : float
            The value to set the array to.
        """

        cell, nuc, react, poly = pos
        if isinstance(cell, str):
            cell_id = self.cell_to_ind[cell]
        else:
            cell_id = cell
        if isinstance(nuc, str):
            nuc_id = self.nuc_to_ind[nuc]
        else:
            nuc_id = cell
        if isinstance(react, str):
            react_id = self.react_to_ind[react]
        else:
            react_id = react

        self.rates[cell_id, nuc_id, react_id, poly] = val
