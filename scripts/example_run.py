"""An example file showing how to run a simulation."""

import numpy as np
import opendeplete
import sys

import example_geometry

# Load geometry from example
geometry, lower_left, upper_right = example_geometry.generate_cheap_geometry(n_rings=10)

# Create dt vector, 4 hour time steps for first 6 days, then 1 day time steps for 200 days.
dt_1 = np.repeat(3600.0 * 4, 24)
dt_2 = np.repeat(3600.0 * 24, 194)
dt = np.append(dt_1, dt_2)

# Create settings variable
settings = opendeplete.OpenMCSettings()

settings.openmc_call = "openmc"
# An example for mpiexec:
# settings.openmc_call = ["mpiexec", "openmc"]
settings.particles = 30000
settings.batches = 40
settings.inactive = 20
settings.lower_left = lower_left
settings.upper_right = upper_right

settings.power = 2.337e15*4  # MeV/second cm from CASMO
settings.dt_vec = dt
settings.output_dir = 'reference' + sys.argv[1]

op = opendeplete.OpenMCOperator(geometry, settings)

# Perform simulation using the MCNPX/MCNP6 algorithm
opendeplete.integrator.celi_m1(op)
