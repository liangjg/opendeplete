""" Regression tests for leqi_cfq4.py"""

import os
import unittest

import numpy as np
from mpi4py import MPI

import opendeplete
from opendeplete import results
from opendeplete import utilities
import test.dummy_geometry as dummy_geometry

class TestLEQI_CFQ4Regression(unittest.TestCase):
    """ Regression tests for opendeplete.integrator.leqi_cfq4 algorithm.

    These tests integrate a simple test problem described in dummy_geometry.py.
    """

    @classmethod
    def setUpClass(cls):
        """ Save current directory in case integrator crashes."""
        cls.cwd = os.getcwd()
        cls.results = "test_integrator_regression"

    def test_leqi_cfq4(self):
        """ Integral regression test of integrator algorithm using LE/QI CFQ4. """

        settings = opendeplete.Settings()
        settings.dt_vec = [1/2, 1/3, 2/3]
        settings.output_dir = self.results

        op = dummy_geometry.DummyGeometry(settings)

        # Perform simulation using the LE/QI CFQ4 algorithm
        opendeplete.leqi_cfq4(op, print_out=False)

        # Load the files
        res = results.read_results(settings.output_dir + "/results.h5")

        _, y1 = utilities.evaluate_single_nuclide(res, "1", "1")
        _, y2 = utilities.evaluate_single_nuclide(res, "1", "2")

        # Mathematica solution
        s1 = [1.629483276498146, 1.141921621804422]
        s2 = [2.098356526334020, 1.327410934086008]
        s3 = [2.833459051424228, 2.139667147926196]

        tol = 1.0e-13

        self.assertLess(np.absolute(y1[1] - s1[0]), tol)
        self.assertLess(np.absolute(y2[1] - s1[1]), tol)

        self.assertLess(np.absolute(y1[2] - s2[0]), tol)
        self.assertLess(np.absolute(y2[2] - s2[1]), tol)

        self.assertLess(np.absolute(y1[3] - s3[0]), tol)
        self.assertLess(np.absolute(y2[3] - s3[1]), tol)

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
