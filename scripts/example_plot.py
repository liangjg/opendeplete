"""An example file showing how to plot data from a simulation."""

import matplotlib.pyplot as plt

from opendeplete import read_results, \
                        evaluate_single_nuclide, \
                        evaluate_reaction_rate, \
                        evaluate_eigenvalue

# Set variables for where the data is, and what we want to read out.
result_folder = "reference2"

# Load data
results = read_results(result_folder + "/results")

cells = ["10000", "10004", "10008", "10016", "10020", "10024", "10028", "10032", "10036"]
nuc = "Gd157"
rxn = "(n,gamma)"

# Total number of nuclides
plt.figure()
for cell in cells:
    # Pointwise data
    x, y = evaluate_single_nuclide(results, cell, nuc)
    plt.semilogy(x, y, label=cell)

plt.xlabel("Time, s")
plt.ylabel("Total Number")
plt.legend(loc="best")
plt.savefig("number2.pdf")

# Reaction rate
plt.figure()
for cell in cells:
    # Pointwise data
    x, y = evaluate_reaction_rate(results, cell, nuc, rxn)
    plt.plot(x, y, label=cell)
plt.xlabel("Time, s")
plt.ylabel("Reaction Rate, 1/s")
plt.legend(loc="best")
plt.savefig("rate2.pdf")

# Eigenvalue
plt.figure()
x, y = evaluate_eigenvalue(results)
plt.plot(x, y)
plt.xlabel("Time, s")
plt.ylabel("Eigenvalue")

plt.savefig("eigvl2.pdf")
