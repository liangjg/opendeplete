""" Regression tests for cf4.py"""

import os
import unittest

import numpy as np
from mpi4py import MPI

import opendeplete
from opendeplete import results
from opendeplete import utilities
import test.dummy_geometry as dummy_geometry

class TestCF4Regression(unittest.TestCase):
    """ Regression tests for opendeplete.integrator.cf4 algorithm.

    These tests integrate a simple test problem described in dummy_geometry.py.
    """

    @classmethod
    def setUpClass(cls):
        """ Save current directory in case integrator crashes."""
        cls.cwd = os.getcwd()
        cls.results = "test_integrator_regression"

    def test_cf4(self):
        """ Integral regression test of integrator algorithm using CF4. """

        settings = opendeplete.Settings()
        settings.dt_vec = [0.75, 0.75]
        settings.output_dir = self.results

        op = dummy_geometry.DummyGeometry(settings)

        # Perform simulation using the MCNPX/MCNP6 algorithm
        opendeplete.cf4(op, print_out=False)

        # Load the files
        res = results.read_results(settings.output_dir + "/results.h5")

        _, y1 = utilities.evaluate_single_nuclide(res, "1", "1")
        _, y2 = utilities.evaluate_single_nuclide(res, "1", "2")

        # Mathematica solution
        s1 = [2.061016292915567, 1.377835877143979]
        s2 = [2.572413180088443, 2.637316295190417]

        tol = 1.0e-13

        self.assertLess(np.absolute(y1[1] - s1[0]), tol)
        self.assertLess(np.absolute(y2[1] - s1[1]), tol)

        self.assertLess(np.absolute(y1[2] - s2[0]), tol)
        self.assertLess(np.absolute(y2[2] - s2[1]), tol)

    @classmethod
    def tearDownClass(cls):
        """ Clean up files"""

        os.chdir(cls.cwd)

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            os.remove(os.path.join(cls.results, "results.h5"))
            os.rmdir(cls.results)


if __name__ == '__main__':
    unittest.main()
