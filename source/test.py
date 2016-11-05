import results
import zernike
import numpy as np

order = 6

r = results.read_results("test_10_order/step1.pklz")

#~ con = r.num[0]

#~ zer = zernike.ZernikePolynomial(order, con["10000", "Xe-135"]  * 20 * 32 / (np.pi * 0.412275**2) / np.pi)

#~ zer.plot_disk(20, 32, "test.pdf")

rea = r.rates[0]

zer = rea.get_fet(["10000", "Xe-135", "(n,gamma)"])  * 20 * 32 / (np.pi * 0.412275**2) / np.pi * 1.0e24

zer.plot_disk(100, 80, "test2.pdf")
