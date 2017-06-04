"""An example file showing how to make a geometry.

This particular example creates a 3x3 geometry, with 8 regular pins and one
Gd-157 2 wt-percent enriched.  All pins are segmented.
"""

from collections import OrderedDict
import math

import numpy as np
import openmc

from opendeplete import density_to_mat


def generate_initial_number_density():
    """ Generates initial number density.

    These results were from a CASMO5 run in which the gadolinium pin was
    loaded with 2 wt percent of Gd-157.
    """

    # Concentration to be used for all fuel pins
    fuel_dict = OrderedDict()
    fuel_dict['U234'] = 8.40809941906e+18
    fuel_dict['U235'] = 1.04652755446e+21
    fuel_dict['U238'] = 2.19208402675e+22
    fuel_dict['O16'] = 4.58422147203e+22
    fuel_dict['O17'] = 1.74156381484e+19
    fuel_dict['O18'] = 9.19214843062e+19
    # fuel_dict['O18'] = 9.51352e19 # Does not exist in ENDF71, merged into 17

    # Concentration to be used for the gadolinium fuel pin
    fuel_gd_dict = OrderedDict()
    fuel_gd_dict['U234'] = 8.23993743068e+18
    fuel_gd_dict['U235'] = 1.02559700337e+21
    fuel_gd_dict['U238'] = 2.14824234621e+22
    fuel_gd_dict['O16'] = 4.49253704259e+22
    fuel_gd_dict['O17'] = 1.70673253854e+19
    fuel_gd_dict['O18'] = 9.008305462e+19
    fuel_gd_dict['Gd152'] = 1.57744481862e+18
    fuel_gd_dict['Gd154'] = 1.7194148523e+19
    fuel_gd_dict['Gd155'] = 1.16730916578e+20
    fuel_gd_dict['Gd156'] = 1.61451477186e+20
    fuel_gd_dict['Gd157'] = 1.23435057057e+20
    fuel_gd_dict['Gd158'] = 1.95918646473e+20
    fuel_gd_dict['Gd160'] = 1.72414718676e+20
    # There are a whole bunch of 1e-10 stuff here.

    # Concentration to be used for cladding
    clad_dict = OrderedDict()
    clad_dict['O16'] = 3.07427e20
    clad_dict['O17'] = 7.48868e17
    clad_dict['Cr50'] = 3.29620e18
    clad_dict['Cr52'] = 6.35639e19
    clad_dict['Cr53'] = 7.20763e18
    clad_dict['Cr54'] = 1.79413e18
    clad_dict['Fe54'] = 5.57350e18
    clad_dict['Fe56'] = 8.74921e19
    clad_dict['Fe57'] = 2.02057e18
    clad_dict['Fe58'] = 2.68901e17
    clad_dict['Cr50'] = 3.29620e18
    clad_dict['Cr52'] = 6.35639e19
    clad_dict['Cr53'] = 7.20763e18
    clad_dict['Cr54'] = 1.79413e18
    clad_dict['Ni58'] = 2.51631e19
    clad_dict['Ni60'] = 9.69278e18
    clad_dict['Ni61'] = 4.21338e17
    clad_dict['Ni62'] = 1.34341e18
    clad_dict['Ni64'] = 3.43127e17
    clad_dict['Zr90'] = 2.18320e22
    clad_dict['Zr91'] = 4.76104e21
    clad_dict['Zr92'] = 7.27734e21
    clad_dict['Zr94'] = 7.37494e21
    clad_dict['Zr96'] = 1.18814e21
    clad_dict['Sn112'] = 4.67352e18
    clad_dict['Sn114'] = 3.17992e18
    clad_dict['Sn115'] = 1.63814e18
    clad_dict['Sn116'] = 7.00546e19
    clad_dict['Sn117'] = 3.70027e19
    clad_dict['Sn118'] = 1.16694e20
    clad_dict['Sn119'] = 4.13872e19
    clad_dict['Sn120'] = 1.56973e20
    clad_dict['Sn122'] = 2.23076e19
    clad_dict['Sn124'] = 2.78966e19

    # Gap concentration
    # Funny enough, the example problem uses air.
    gap_dict = OrderedDict()
    gap_dict['O16'] = 7.86548e18
    gap_dict['O17'] = 2.99548e15
    gap_dict['N14'] = 3.38646e19
    gap_dict['N15'] = 1.23717e17

    # Concentration to be used for coolant
    # No boron
    cool_dict = OrderedDict()
    cool_dict['H1'] = 4.68063e22
    cool_dict['O16'] = 2.33427e22
    cool_dict['O17'] = 8.89086e18

    # Store these dictionaries in the initial conditions dictionary
    initial_density = OrderedDict()
    initial_density['fuel_gd'] = fuel_gd_dict
    initial_density['fuel'] = fuel_dict
    initial_density['gap'] = gap_dict
    initial_density['clad'] = clad_dict
    initial_density['cool'] = cool_dict

    # Set up libraries to use
    temperature = OrderedDict()
    sab = OrderedDict()

    # Toggle betweeen MCNP and NNDC data
    MCNP = False

    if MCNP:
        temperature['fuel_gd'] = 900.0
        temperature['fuel'] = 900.0
        # We approximate temperature of everything as 600K, even though it was
        # actually 580K.
        temperature['gap'] = 600.0
        temperature['clad'] = 600.0
        temperature['cool'] = 600.0
    else:
        temperature['fuel_gd'] = 293.6
        temperature['fuel'] = 293.6
        temperature['gap'] = 293.6
        temperature['clad'] = 293.6
        temperature['cool'] = 293.6

    sab['cool'] = 'c_H_in_H2O'

    # Set up burnable materials
    burn = OrderedDict()
    burn['fuel_gd'] = True
    burn['fuel'] = True
    burn['gap'] = False
    burn['clad'] = False
    burn['cool'] = False

    return temperature, sab, initial_density, burn

def segment_pin(n_rings, n_wedges, r_fuel, r_gap, r_clad):
    """ Calculates a segmented pin.

    Separates a pin with n_rings and n_wedges.  All cells have equal volume.
    Pin is centered at origin.
    """

    # Calculate all the volumes of interest
    v_fuel = math.pi * r_fuel**2
    v_gap = math.pi * r_gap**2 - v_fuel
    v_clad = math.pi * r_clad**2 - v_fuel - v_gap
    v_ring = v_fuel / n_rings
    v_segment = v_ring / n_wedges

    # Compute ring radiuses
    r_rings = np.zeros(n_rings)

    for i in range(n_rings):
        r_rings[i] = math.sqrt(1.0/(math.pi) * v_ring * (i+1))

    # Compute thetas
    theta = np.linspace(0, 2*math.pi, n_wedges + 1)

    # Compute surfaces
    fuel_rings = [openmc.ZCylinder(x0=0, y0=0, R=r_rings[i])
                  for i in range(n_rings)]

    fuel_wedges = [openmc.Plane(A=math.cos(theta[i]), B=math.sin(theta[i]))
                   for i in range(n_wedges)]

    gap_ring = openmc.ZCylinder(x0=0, y0=0, R=r_gap)
    clad_ring = openmc.ZCylinder(x0=0, y0=0, R=r_clad)

    # Create cells
    fuel_cells = []
    if n_wedges == 1:
        for i in range(n_rings):
            cell = openmc.Cell(name='fuel')
            if i == 0:
                cell.region = -fuel_rings[0]
            else:
                cell.region = +fuel_rings[i-1] & -fuel_rings[i]
            fuel_cells.append(cell)
    else:
        for i in range(n_rings):
            for j in range(n_wedges):
                cell = openmc.Cell(name='fuel')
                if i == 0:
                    if j != n_wedges-1:
                        cell.region = (-fuel_rings[0]
                                       & +fuel_wedges[j]
                                       & -fuel_wedges[j+1])
                    else:
                        cell.region = (-fuel_rings[0]
                                       & +fuel_wedges[j]
                                       & -fuel_wedges[0])
                else:
                    if j != n_wedges-1:
                        cell.region = (+fuel_rings[i-1]
                                       & -fuel_rings[i]
                                       & +fuel_wedges[j]
                                       & -fuel_wedges[j+1])
                    else:
                        cell.region = (+fuel_rings[i-1]
                                       & -fuel_rings[i]
                                       & +fuel_wedges[j]
                                       & -fuel_wedges[0])
                fuel_cells.append(cell)

    # Gap ring
    gap_cell = openmc.Cell(name='gap')
    gap_cell.region = +fuel_rings[-1] & -gap_ring
    fuel_cells.append(gap_cell)

    # Clad ring
    clad_cell = openmc.Cell(name='clad')
    clad_cell.region = +gap_ring & -clad_ring
    fuel_cells.append(clad_cell)

    # Moderator
    mod_cell = openmc.Cell(name='cool')
    mod_cell.region = +clad_ring
    fuel_cells.append(mod_cell)

    # Form universe
    fuel_u = openmc.Universe()
    fuel_u.add_cells(fuel_cells)

    return fuel_u, v_segment, v_gap, v_clad

def generate_cheap_geometry(n_rings):
    pitch = 1.26197
    r_fuel = 0.412275
    r_gap = 0.418987
    r_clad = 0.476121

    vol_fuel = np.pi * r_fuel**2

    temperature, sab, initial_density, burn = generate_initial_number_density()

    # Walls
    left = openmc.XPlane(x0=0.0, name='left')
    right = openmc.XPlane(x0=1.5*pitch, name='right')
    bottom = openmc.YPlane(y0=0.0, name='bottom')
    top = openmc.YPlane(y0=1.5*pitch, name='top')

    left.boundary_type = 'reflective'
    right.boundary_type = 'reflective'
    top.boundary_type = 'reflective'
    bottom.boundary_type = 'reflective'

    radii_gd = []
    radii_ul = []
    radii_ur = []
    radii_lr = []

    cells = []

    # Fill in fuels
    for n in range(n_rings):
        vol_outer = vol_fuel * (n + 1) / n_rings
        r = np.sqrt(vol_outer / np.pi)
        radii_gd.append(openmc.ZCylinder(x0=0.0, y0=0.0, R=r))
        radii_ul.append(openmc.ZCylinder(x0=0.0, y0=pitch, R=r))
        radii_ur.append(openmc.ZCylinder(x0=pitch, y0=pitch, R=r))
        radii_lr.append(openmc.ZCylinder(x0=pitch, y0=0.0, R=r))

        fuel_gd_i = openmc.Cell(name="Gd Ring " + str(n))
        fuel_ul_i = openmc.Cell(name="UL Ring " + str(n))
        fuel_ur_i = openmc.Cell(name="UR Ring " + str(n))
        fuel_lr_i = openmc.Cell(name="LR Ring " + str(n))
        if n == 0:
            fuel_gd_i.region = -radii_gd[n] & +left & +bottom
            fuel_ul_i.region = -radii_ul[n] & +left
            fuel_ur_i.region = -radii_ur[n]
            fuel_lr_i.region = -radii_lr[n] & +bottom
        else:
            fuel_gd_i.region = +radii_gd[n-1] & -radii_gd[n] & +left & +bottom
            fuel_ul_i.region = +radii_ul[n-1] & -radii_ul[n] & +left
            fuel_ur_i.region = +radii_ur[n-1] & -radii_ur[n]
            fuel_lr_i.region = +radii_lr[n-1] & -radii_lr[n] & +bottom

        fuel_gd_i.fill = density_to_mat(initial_density["fuel_gd"])

        if "fuel_gd" in sab:
            fuel_gd_i.fill.add_s_alpha_beta(sab["fuel_gd"])
        fuel_gd_i.fill.temperature = temperature["fuel_gd"]
        fuel_gd_i.fill.depletable = burn["fuel_gd"]
        fuel_gd_i.fill.volume = vol_fuel / n_rings / 4.0

        fuel_ul_i.fill = density_to_mat(initial_density["fuel"])

        if "fuel" in sab:
            fuel_ul_i.fill.add_s_alpha_beta(sab["fuel"])
        fuel_ul_i.fill.temperature = temperature["fuel"]
        fuel_ul_i.fill.depletable = burn["fuel"]
        fuel_ul_i.fill.volume = vol_fuel / n_rings / 2.0

        fuel_ur_i.fill = density_to_mat(initial_density["fuel"])

        if "fuel" in sab:
            fuel_ur_i.fill.add_s_alpha_beta(sab["fuel"])
        fuel_ur_i.fill.temperature = temperature["fuel"]
        fuel_ur_i.fill.depletable = burn["fuel"]
        fuel_ur_i.fill.volume = vol_fuel / n_rings

        fuel_lr_i.fill = density_to_mat(initial_density["fuel"])

        if "fuel" in sab:
            fuel_lr_i.fill.add_s_alpha_beta(sab["fuel"])
        fuel_lr_i.fill.temperature = temperature["fuel"]
        fuel_lr_i.fill.depletable = burn["fuel"]
        fuel_lr_i.fill.volume = vol_fuel / n_rings / 2.0

        cells.append(fuel_gd_i)
        cells.append(fuel_ul_i)
        cells.append(fuel_ur_i)
        cells.append(fuel_lr_i)

    # Gap
    rg1 = openmc.ZCylinder(x0=0.0, y0=0.0, R=r_gap)
    rg2 = openmc.ZCylinder(x0=0.0, y0=pitch, R=r_gap)
    rg3 = openmc.ZCylinder(x0=pitch, y0=pitch, R=r_gap)
    rg4 = openmc.ZCylinder(x0=pitch, y0=0.0, R=r_gap)
    v_gap = np.pi * (r_gap**2 - r_fuel**2)

    gap_gd = openmc.Cell(name="Gd Gap")
    gap_ul = openmc.Cell(name="UL Gap")
    gap_ur = openmc.Cell(name="UR Gap")
    gap_lr = openmc.Cell(name="LR Gap")

    gap_gd.region = +radii_gd[n_rings-1] & -rg1 & +left & +bottom
    gap_ul.region = +radii_ul[n_rings-1] & -rg2 & +left
    gap_ur.region = +radii_ur[n_rings-1] & -rg3
    gap_lr.region = +radii_lr[n_rings-1] & -rg4 & +bottom

    gap_gd.fill = density_to_mat(initial_density["gap"])

    if "gap" in sab:
        gap_gd.fill.add_s_alpha_beta(sab["gap"])
    gap_gd.fill.temperature = temperature["gap"]
    gap_gd.fill.depletable = burn["gap"]
    gap_gd.fill.volume = v_gap / 4.0

    gap_ul.fill = density_to_mat(initial_density["gap"])

    if "gap" in sab:
        gap_ul.fill.add_s_alpha_beta(sab["gap"])
    gap_ul.fill.temperature = temperature["gap"]
    gap_ul.fill.depletable = burn["gap"]
    gap_ul.fill.volume = v_gap / 2.0

    gap_ur.fill = density_to_mat(initial_density["gap"])

    if "gap" in sab:
        gap_ur.fill.add_s_alpha_beta(sab["gap"])
    gap_ur.fill.temperature = temperature["gap"]
    gap_ur.fill.depletable = burn["gap"]
    gap_ur.fill.volume = v_gap

    gap_lr.fill = density_to_mat(initial_density["gap"])

    if "gap" in sab:
        gap_lr.fill.add_s_alpha_beta(sab["gap"])
    gap_lr.fill.temperature = temperature["gap"]
    gap_lr.fill.depletable = burn["gap"]
    gap_lr.fill.volume = v_gap / 2.0

    cells.append(gap_gd)
    cells.append(gap_ul)
    cells.append(gap_ur)
    cells.append(gap_lr)

    # Clad
    rclad1 = openmc.ZCylinder(x0=0.0, y0=0.0, R=r_clad)
    rclad2 = openmc.ZCylinder(x0=0.0, y0=pitch, R=r_clad)
    rclad3 = openmc.ZCylinder(x0=pitch, y0=pitch, R=r_clad)
    rclad4 = openmc.ZCylinder(x0=pitch, y0=0.0, R=r_clad)
    v_clad = np.pi * (r_clad**2 - r_gap**2)

    clad_gd = openmc.Cell(name="Gd Clad")
    clad_ul = openmc.Cell(name="UL Clad")
    clad_ur = openmc.Cell(name="UR Clad")
    clad_lr = openmc.Cell(name="LR Clad")

    clad_gd.region = +rg1 & -rclad1 & +left & +bottom
    clad_ul.region = +rg2 & -rclad2 & +left
    clad_ur.region = +rg3 & -rclad3
    clad_lr.region = +rg4 & -rclad4 & +bottom

    clad_gd.fill = density_to_mat(initial_density["clad"])

    if "clad" in sab:
        clad_gd.fill.add_s_alpha_beta(sab["clad"])
    clad_gd.fill.temperature = temperature["clad"]
    clad_gd.fill.depletable = burn["clad"]
    clad_gd.fill.volume = v_clad / 4.0

    clad_ul.fill = density_to_mat(initial_density["clad"])

    if "clad" in sab:
        clad_ul.fill.add_s_alpha_beta(sab["clad"])
    clad_ul.fill.temperature = temperature["clad"]
    clad_ul.fill.depletable = burn["clad"]
    clad_ul.fill.volume = v_clad / 2.0

    clad_ur.fill = density_to_mat(initial_density["clad"])

    if "clad" in sab:
        clad_ur.fill.add_s_alpha_beta(sab["clad"])
    clad_ur.fill.temperature = temperature["clad"]
    clad_ur.fill.depletable = burn["clad"]
    clad_ur.fill.volume = v_clad

    clad_lr.fill = density_to_mat(initial_density["clad"])

    if "clad" in sab:
        clad_lr.fill.add_s_alpha_beta(sab["clad"])
    clad_lr.fill.temperature = temperature["clad"]
    clad_lr.fill.depletable = burn["clad"]
    clad_lr.fill.volume = v_clad / 2.0

    cells.append(clad_gd)
    cells.append(clad_ul)
    cells.append(clad_ur)
    cells.append(clad_lr)

    # Water
    water = openmc.Cell(name="cool")
    water.region = +left & +bottom & -top & -right & +rclad1 & +rclad2 & +rclad3 & +rclad4

    water.fill = density_to_mat(initial_density["cool"])

    if "cool" in sab:
        water.fill.add_s_alpha_beta(sab["cool"])
    water.fill.temperature = temperature["cool"]
    water.fill.depletable = burn["cool"]
    water.fill.volume = (1.5*pitch)**2 - (0.25 + 0.5 * 2 + 1.0) * np.pi * r_clad**2

    cells.append(water)

    # Instantiate Universe
    root = openmc.Universe(universe_id=0, name='root universe')

    # Register Cells with Universe
    root.add_cells(cells)

    # Instantiate a Geometry, register the root Universe, and export to XML
    geometry = openmc.Geometry(root)
    geometry.export_to_xml()

    # Run this if you want to verify the geometry
    lower_left = [0.0, 0.0, -0.5]
    upper_right = [1.5*pitch, 1.5*pitch, +0.5]
    # vol_calc = openmc.VolumeCalculation(cells, 100000000, lower_left, upper_right)

    # settings = openmc.Settings()
    # settings.volume_calculations = [vol_calc]
    # settings.export_to_xml()

    # openmc.calculate_volumes()
    # vol_calc.load_results("volume_1.h5")

    # for cell in cells:
    #     print(cell.name, cell.fill.volume, vol_calc.volumes[cell.id][0], np.abs(cell.fill.volume - vol_calc.volumes[cell.id][0])/ cell.fill.volume)

    # exit()

    plot = openmc.Plot()
    plot.basis = 'xy'
    plot.origin = (0.75*pitch, 0.75*pitch, 0.0)
    plot.width = (1.5*pitch, 1.5*pitch)
    plot.pixels = (400, 400)
    plots = openmc.Plots()
    plots.append(plot)
    plots.export_to_xml()

    return geometry, lower_left, upper_right

def generate_geometry(n_rings, n_wedges):
    """ Generates example geometry.

    This function creates the initial geometry, a 9 pin reflective problem.
    One pin, containing gadolinium, is discretized into sectors.

    In addition to what one would do with the general OpenMC geometry code, it
    is necessary to create a dictionary, volume, that maps a cell ID to a
    volume. Further, by naming cells the same as the above materials, the code
    can automatically handle the mapping.

    Parameters
    ----------
    n_rings : int
        Number of rings to generate for the geometry
    n_wedges : int
        Number of wedges to generate for the geometry
    """

    pitch = 1.26197
    r_fuel = 0.412275
    r_gap = 0.418987
    r_clad = 0.476121

    n_pin = 3

    # This table describes the 'fuel' to actual type mapping
    # It's not necessary to do it this way.  Just adjust the initial conditions
    # below.
    mapping = ['fuel', 'fuel', 'fuel',
               'fuel', 'fuel_gd', 'fuel',
               'fuel', 'fuel', 'fuel']

    # Form pin cell
    fuel_u, v_segment, v_gap, v_clad = segment_pin(n_rings, n_wedges, r_fuel, r_gap, r_clad)

    # Form lattice
    all_water_c = openmc.Cell(name='cool')
    all_water_u = openmc.Universe(cells=(all_water_c, ))

    lattice = openmc.RectLattice()
    lattice.pitch = [pitch]*2
    lattice.lower_left = [-pitch*n_pin/2, -pitch*n_pin/2]
    lattice_array = [[fuel_u for i in range(n_pin)] for j in range(n_pin)]
    lattice.universes = lattice_array
    lattice.outer = all_water_u

    # Bound universe
    x_low = openmc.XPlane(x0=-pitch*n_pin/2, boundary_type='reflective')
    x_high = openmc.XPlane(x0=pitch*n_pin/2, boundary_type='reflective')
    y_low = openmc.YPlane(y0=-pitch*n_pin/2, boundary_type='reflective')
    y_high = openmc.YPlane(y0=pitch*n_pin/2, boundary_type='reflective')
    z_low = openmc.ZPlane(z0=-10, boundary_type='reflective')
    z_high = openmc.ZPlane(z0=10, boundary_type='reflective')

    # Compute bounding box
    lower_left = [-pitch*n_pin/2, -pitch*n_pin/2, -10]
    upper_right = [pitch*n_pin/2, pitch*n_pin/2, 10]

    root_c = openmc.Cell(fill=lattice)
    root_c.region = (+x_low & -x_high
                     & +y_low & -y_high
                     & +z_low & -z_high)
    root_u = openmc.Universe(universe_id=0, cells=(root_c, ))
    geometry = openmc.Geometry(root_u)

    v_cool = pitch**2 - (v_gap + v_clad + n_rings * n_wedges * v_segment)

    # Store volumes for later usage
    volume = {'fuel': v_segment, 'gap':v_gap, 'clad':v_clad, 'cool':v_cool}

    return geometry, volume, mapping, lower_left, upper_right

def generate_problem(n_rings=5, n_wedges=8):
    """ Merges geometry and materials.

    This function initializes the materials for each cell using the dictionaries
    provided by generate_initial_number_density.  It is assumed a cell named
    'fuel' will have further region differentiation (see mapping).

    Parameters
    ----------
    n_rings : int, optional
        Number of rings to generate for the geometry
    n_wedges : int, optional
        Number of wedges to generate for the geometry
    """

    # Get materials dictionary, geometry, and volumes
    temperature, sab, initial_density, burn = generate_initial_number_density()
    geometry, volume, mapping, lower_left, upper_right = generate_geometry(n_rings, n_wedges)

    # Apply distribmats, fill geometry
    cells = geometry.root_universe.get_all_cells()
    for cell_id in cells:
        cell = cells[cell_id]
        if cell.name == 'fuel':

            omc_mats = []

            for cell_type in mapping:
                omc_mat = density_to_mat(initial_density[cell_type])

                if cell_type in sab:
                    omc_mat.add_s_alpha_beta(sab[cell_type])
                omc_mat.temperature = temperature[cell_type]
                omc_mat.depletable = burn[cell_type]
                omc_mat.volume = volume['fuel']

                omc_mats.append(omc_mat)

            cell.fill = omc_mats
        elif cell.name != '':
            omc_mat = density_to_mat(initial_density[cell.name])

            if cell.name in sab:
                omc_mat.add_s_alpha_beta(sab[cell.name])
            omc_mat.temperature = temperature[cell.name]
            omc_mat.depletable = burn[cell.name]
            omc_mat.volume = volume[cell.name]

            cell.fill = omc_mat

    return geometry, lower_left, upper_right
