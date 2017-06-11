"""An example file showing how to run a simulation."""

import numpy as np
import opendeplete
import sys

import example_geometry

t_end = 150.0 * 24 * 3600 # 150 days, corresponding to peak l2 norm
algorithm = opendeplete.integrator.celi_m1
stages = 2

neutron_grid = [500, 1000, 2000]
step_grid = [1, 2, 3, 5, 6, 10, 15, 25]

for neutron in neutron_grid:
    for step in step_grid:
        folder_name = algorithm.__name__ + "/" + str(neutron) + "/" + str(step) + "/" + sys.argv[1]

        # Load geometry from example
        geometry, lower_left, upper_right = example_geometry.generate_cheap_geometry(n_rings=10)

        # Time grid
        dt = np.repeat(t_end / step, step)

        # Create settings variable
        settings = opendeplete.OpenMCSettings()

        neutron_actual = neutron * (4 / stages) * (step_grid[-1] / step)

        settings.openmc_call = "openmc"
        settings.particles = int(neutron_actual)
        settings.batches = 40
        settings.inactive = 20
        settings.lower_left = lower_left
        settings.upper_right = upper_right

        settings.power = 2.337e15  # MeV/second cm from CASMO
        settings.dt_vec = dt
        settings.output_dir = folder_name

        op = opendeplete.OpenMCOperator(geometry, settings)

        # Perform simulation using the MCNPX/MCNP6 algorithm
        opendeplete.integrator.celi_m1(op)
