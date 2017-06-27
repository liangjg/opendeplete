""" Tests for el3_ats.py """

import copy
import unittest

import numpy as np
from mpi4py import MPI

import test.dummy_geometry as dummy_geometry
from opendeplete import ReactionRates, integrator, Settings

class TestEL3_ATS(unittest.TestCase):
    """ Tests for el3_ats.py
    """

    def test_el3_compute_sum_rate(self):
        """ Ensure sums of reaction rates are computed properly. """

        # Create reaction rate

        mat_to_ind = {"10000" : 0, "10001" : 1}
        nuc_to_ind = {"U238" : 0, "U235" : 1}
        react_to_ind = {"fission" : 0, "(n,gamma)" : 1}

        rates = ReactionRates(mat_to_ind, nuc_to_ind, react_to_ind)

        scale = MPI.COMM_WORLD.rank + 1

        rates["10000", "U238", "fission"] = 1.0
        rates["10001", "U238", "fission"] = 2.0
        rates["10000", "U235", "fission"] = 3.0
        rates["10001", "U235", "fission"] = 4.0
        rates["10000", "U238", "(n,gamma)"] = 5.0
        rates["10001", "U238", "(n,gamma)"] = 6.0
        rates["10000", "U235", "(n,gamma)"] = 7.0
        rates["10001", "U235", "(n,gamma)"] = 8.0

        # Create diffs
        diff_M = []
        diff_M.append(np.array([1.0, 2.0]) * scale)
        diff_M.append(np.array([2.0, 3.0]) * scale)
        diff_S = []
        diff_S.append(np.array([2.0, 4.0]) * scale)
        diff_S.append(np.array([3.0, 4.0]) * scale)
        x_avg = []
        x_avg.append(np.array([1.0, 1.0]))
        x_avg.append(np.array([2.0, 2.0]))

        nuc_to_ind_b = {"U235" : 0, "U238" : 1}

        mu, std = integrator.el3_compute_sum_rate(diff_M, diff_S, x_avg, rates, nuc_to_ind_b)

        M_per_rank = (1.0 + 5.0) * 2.0 + (2.0 + 6.0) * 3.0 + (3.0 + 7.0) * 1.0 + (4.0 + 8.0) * 2.0
        S_per_rank = (1.0 + 5.0) * 4.0 + (2.0 + 6.0) * 4.0 + (3.0 + 7.0) * 2.0 + (4.0 + 8.0) * 3.0
        X_per_rank = (1.0 + 5.0) * 1.0 + (2.0 + 6.0) * 2.0 + (3.0 + 7.0) * 1.0 + (4.0 + 8.0) * 2.0

        scale_2 = np.sum(range(1, 1 + MPI.COMM_WORLD.size)) / MPI.COMM_WORLD.size

        self.assertEqual(mu, M_per_rank * scale_2 / X_per_rank)
        self.assertEqual(std, S_per_rank * scale_2 / X_per_rank)

    def test_el3_compute_rmse_nuc(self):
        """ Ensure sums of reaction rates are computed properly. """

        scale = MPI.COMM_WORLD.rank + 1

        # Create diffs
        diff_M = []
        diff_M.append(np.array([1.0, 2.0]) * scale)
        diff_M.append(np.array([2.0, 3.0]) * scale)
        diff_S = []
        diff_S.append(np.array([2.0, 4.0]) * scale)
        diff_S.append(np.array([3.0, 4.0]) * scale)
        x_avg = []
        x_avg.append(np.array([1.0e7, 1.0e7]))
        x_avg.append(np.array([1.0e5, 1.0e7]))

        mu, std = integrator.el3_compute_rmse_nuc(diff_M, diff_S, x_avg, 1.0e6)

        mu_val = np.sum([(1.0**2 + 2.0**2 + 3.0**2) * i**2 for i in range(1, 1 + MPI.COMM_WORLD.size)])
        std_val = np.sum([(2.0**2 + 4.0**2 + 4.0**2) * i**2 for i in range(1, 1 + MPI.COMM_WORLD.size)])

        self.assertEqual(mu, np.sqrt(mu_val / (3 * MPI.COMM_WORLD.size)))
        self.assertEqual(std, np.sqrt(std_val / (3 * MPI.COMM_WORLD.size)))

    def test_compute_sum_of_squares(self):
        """ Ensure sums of reaction rates are computed properly. """

        # Create diffs
        diff_M = []
        diff_M.append(np.array([1.0, 2.0]))
        diff_M.append(np.array([2.0, 3.0]))
        diff_S = []
        diff_S.append(np.array([2.0, 4.0]))
        diff_S.append(np.array([3.0, 4.0]))
        x_avg = []
        x_avg.append(np.array([1.0e7, 1.0e7]))
        x_avg.append(np.array([1.0e5, 1.0e7]))

        sum_M, sum_S, count = integrator.compute_sum_of_squares(diff_M, diff_S, x_avg, 1.0e6)

        self.assertEqual(sum_M, (1.0**2 + 2.0**2 + 3.0**2))
        self.assertEqual(sum_S, (2.0**2 + 4.0**2 + 4.0**2))
        self.assertEqual(count, 3)

    def test_compute_relative_diff(self):
        """ Ensure sums of reaction rates are computed properly. """

        # Create diffs
        a = []
        a.append(np.array([1.0, 2.0]))
        a.append(np.array([0.0, 4.0]))
        b = []
        b.append(np.array([2.0, 4.0]))
        b.append(np.array([3.0, 3.0]))

        diff = integrator.compute_relative_diff(a, b)

        self.assertEqual(diff[0][0], -1.0)
        self.assertEqual(diff[0][1], -1.0)
        self.assertEqual(diff[1][0], (1.0e-24 - 3.0) / 1.0e-24)
        self.assertEqual(diff[1][1], 0.25)

    def test_compute_diff(self):
        """ Ensure sums of reaction rates are computed properly. """

        # Create diffs
        a = []
        a.append(np.array([1.0, 2.0]))
        a.append(np.array([0.0, 4.0]))
        b = []
        b.append(np.array([2.0, 4.0]))
        b.append(np.array([3.0, 3.0]))

        diff = integrator.compute_diff(a, b)

        self.assertEqual(diff[0][0], -1.0)
        self.assertEqual(diff[0][1], -2.0)
        self.assertEqual(diff[1][0], -3.0)
        self.assertEqual(diff[1][1], 1.0)

    def test_el3(self):
        """ Integral regression test of integrator algorithm using EL3. """

        settings = Settings()
        settings.dt_vec = [0.75, 0.75]

        op = dummy_geometry.DummyGeometry(settings)
        vec = copy.deepcopy(op.initial_condition())

        # Perform simulation using the MCNPX/MCNP6 algorithm
        x, xhat, rates, eigvl = integrator.el3_inner(op, vec, 0.0, 0.75, print_out=False)

        # Mathematica solution
        s1 = [2.098374972718346, 1.409627547361825]
        s1hat = [2.146945500656890, 1.443474904774555]

        tol = 1.0e-13

        self.assertLess(np.absolute(x[0][0] - s1[0]), tol)
        self.assertLess(np.absolute(x[0][1] - s1[1]), tol)

        self.assertLess(np.absolute(xhat[0][0] - s1hat[0]), tol)
        self.assertLess(np.absolute(xhat[0][1] - s1hat[1]), tol)


if __name__ == '__main__':
    unittest.main()
