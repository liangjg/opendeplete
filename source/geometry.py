"""Geometry module.

This module contains information about the geometry necessary for the depletion
process and the OpenMC input file generation process.  Currently, the geometry
is hard-coded.  It is anticipated that this will change.

Example:
    To create a geometry, just instantiate it::

        $ geo = Geometry()
"""

import depletion_chain
import reaction_rates
import openmc
from collections import OrderedDict


class Operator:
    def __init__(self):
        self.geometry = None
        """openmc_wrapper.Geometry: The OpenMC geometry object."""
        self.settings = None
        """openmc_wrapper.Settings: Settings file"""

    def initialize(self, geo, vol, mat, settings):
        # First, load in depletion data
        self.load_depletion_data(settings.chain_file)

        # Then, create geometry
        self.geometry = openmc_wrapper.Geometry()
        self.geometry.geometry = geo
        self.geometry.materials = mat
        self.geometry.volume = vol
        self.geometry.initialize(settings)

        # Save settings
        self.settings = settings

    def eval(self, vec):
        return geometry.function_evaluation(vec, self.settings)

    def load_depletion_data(self, filename):
        """ Load self.depletion_data

        This problem has a hard-coded 100 batches, with 40 inactive batches. The
        number of neutrons per batch, however, is not hardcoded.

        Args:
            npb (int): Number of neutrons per batch to use.
        """
        # Create a depletion chain object, and then allocate the reaction rate objects
        self.geometry.chain = depletion_chain.DepletionChain()
        self.geometry.chain.xml_read(filename)
