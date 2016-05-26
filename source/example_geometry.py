"""An example file showing how to make a geometry.

This particular example creates a 2x2 geometry, with 3 regular pins and one
Gd-157 2 wt-percent enriched.  The geometry is reflected, with reflections
going through the center of the Gd-157 pin.  Said pin is split into 5 rings.
"""

import function
import openmc
import openmc_wrapper
import numpy as np
import math
from collections import OrderedDict


def generate_initial_number_density():
    """ Generates initial number density.

    These results were from a CASMO5 run in which the gadolinium pin was
    loaded with 2 wt percent of Gd-157.
    """

    # Concentration to be used for all fuel pins
    fuel_dict = OrderedDict()
    fuel_dict['U-235'] = 1.05692e21
    fuel_dict['U-234'] = 1.00506e19
    fuel_dict['U-238'] = 2.21371e22
    fuel_dict['O-16'] = 4.62954e22
    fuel_dict['O-17'] = 1.127684e20
    fuel_dict['I-135'] = 1.0e10
    fuel_dict['Xe-135'] = 1.0e10
    fuel_dict['Xe-136'] = 1.0e10
    fuel_dict['Cs-135'] = 1.0e10
    fuel_dict['Gd-156'] = 1.0e10
    fuel_dict['Gd-157'] = 1.0e10
    # fuel_dict['O-18'] = 9.51352e19 # Does not exist in ENDF71, merged into 17

    # Concentration to be used for the gadolinium fuel pin
    fuel_gd_dict = OrderedDict()
    fuel_gd_dict['U-235'] = 1.03579e21
    fuel_gd_dict['U-238'] = 2.16943e22
    fuel_gd_dict['Gd-156'] = 3.95517E+10
    fuel_gd_dict['Gd-157'] = 1.08156e20
    fuel_gd_dict['O-16'] = 4.64035e22
    fuel_dict['I-135'] = 1.0e10
    fuel_dict['Xe-136'] = 1.0e10
    fuel_dict['Xe-135'] = 1.0e10
    fuel_dict['Cs-135'] = 1.0e10
    # There are a whole bunch of 1e-10 stuff here.

    # Concentration to be used for cladding
    clad_dict = OrderedDict()
    clad_dict['O-16'] = 3.07427e20
    clad_dict['O-17'] = 7.48868e17
    clad_dict['Cr-50'] = 3.29620e18
    clad_dict['Cr-52'] = 6.35639e19
    clad_dict['Cr-53'] = 7.20763e18
    clad_dict['Cr-54'] = 1.79413e18
    clad_dict['Fe-54'] = 5.57350e18
    clad_dict['Fe-56'] = 8.74921e19
    clad_dict['Fe-57'] = 2.02057e18
    clad_dict['Fe-58'] = 2.68901e17
    clad_dict['Cr-50'] = 3.29620e18
    clad_dict['Cr-52'] = 6.35639e19
    clad_dict['Cr-53'] = 7.20763e18
    clad_dict['Cr-54'] = 1.79413e18
    clad_dict['Ni-58'] = 2.51631e19
    clad_dict['Ni-60'] = 9.69278e18
    clad_dict['Ni-61'] = 4.21338e17
    clad_dict['Ni-62'] = 1.34341e18
    clad_dict['Ni-64'] = 3.43127e17
    clad_dict['Zr-90'] = 2.18320e22
    clad_dict['Zr-91'] = 4.76104e21
    clad_dict['Zr-92'] = 7.27734e21
    clad_dict['Zr-94'] = 7.37494e21
    clad_dict['Zr-96'] = 1.18814e21
    clad_dict['Sn-112'] = 4.67352e18
    clad_dict['Sn-114'] = 3.17992e18
    clad_dict['Sn-115'] = 1.63814e18
    clad_dict['Sn-116'] = 7.00546e19
    clad_dict['Sn-117'] = 3.70027e19
    clad_dict['Sn-118'] = 1.16694e20
    clad_dict['Sn-119'] = 4.13872e19
    clad_dict['Sn-120'] = 1.56973e20
    clad_dict['Sn-122'] = 2.23076e19
    clad_dict['Sn-124'] = 2.78966e19

    # Gap concentration
    # Funny enough, the example problem uses air.
    gap_dict = OrderedDict()
    gap_dict['O-16'] = 7.86548e18
    gap_dict['O-17'] = 2.99548e15
    gap_dict['N-14'] = 3.38646e19
    gap_dict['N-15'] = 1.23717e17

    # Concentration to be used for coolant
    # No boron
    cool_dict = OrderedDict()
    cool_dict['H-1'] = 4.68063e22
    cool_dict['O-16'] = 2.33427e22
    cool_dict['O-17'] = 8.89086e18

    # Store these dictionaries in the initial conditions dictionary
    initial_density = OrderedDict()
    initial_density['fuel_gd'] = fuel_gd_dict
    initial_density['fuel'] = fuel_dict
    initial_density['gap'] = gap_dict
    initial_density['clad'] = clad_dict
    initial_density['cool'] = cool_dict

    # Set up libraries to use
    library = OrderedDict()
    library_sab = OrderedDict()
    sab = OrderedDict()

    # Toggle betweeen MCNP and NNDC data
    MCNP = False

    if MCNP:
        library['fuel_gd'] = '82c'
        library['fuel'] = '82c'
        # We approximate temperature of everything as 600K, even though it was
        # actually 580K.
        library['gap'] = '81c'
        library['clad'] = '81c'
        library['cool'] = '81c'

        sab['cool'] = 'lwtr'

        library_sab['cool'] = '26t'
    else:
        library['fuel_gd'] = '71c'
        library['fuel'] = '71c'
        library['gap'] = '71c'
        library['clad'] = '71c'
        library['cool'] = '71c'

        sab['cool'] = 'HH2O'

        library_sab['cool'] = '71t'

    # Set up burnable materials
    burn = OrderedDict()
    burn['fuel_gd'] = True
    burn['fuel'] = False
    burn['gap'] = False
    burn['clad'] = False
    burn['cool'] = False

    materials = openmc_wrapper.Materials()
    materials.library = library
    materials.library_sab = library_sab
    materials.sab = sab
    materials.initial_density = initial_density
    materials.burn = burn

    return materials


def generate_geometry():
    """ Generates example geometry.

    This function creates the initial geometry, a 4 pin reflective problem.
    One pin, containing gadolinium, is discretized into 5 radial cells of
    equal volume.  Reflections go through the center of this pin.

    In addition to what one would do with the general OpenMC geometry code, it
    is necessary to create a dictionary, volume, that maps a cell ID to a
    volume. Further, by naming cells the same as the above materials, the code
    can automatically handle the mapping.
    """

    import math
    import numpy as np

    pitch = 1.26197
    r_fuel = 0.412275
    r_gap = 0.418987
    r_clad = 0.476121
    n_rings = 1

    # Calculate all the volumes of interest ahead of time
    v_fuel = math.pi * r_fuel**2
    v_gap = math.pi * r_gap**2 - v_fuel
    v_clad = math.pi * r_clad**2 - v_fuel - v_gap
    v_ring = v_fuel / n_rings

    # Form dictionaries for later use.
    volume = OrderedDict()

    # Calculate pin discretization radii
    r_rings = np.zeros(n_rings)

    # Remaining rings
    for i in range(n_rings):
        r_rings[i] = math.sqrt(1.0/(math.pi) * v_ring * (i+1))

    # Form bounding box
    left = openmc.XPlane(x0=-3.0/2.0*pitch, name='left')
    right = openmc.XPlane(x0=3.0/2.0*pitch, name='right')
    bottom = openmc.YPlane(y0=-3.0/2.0*pitch, name='bottom')
    top = openmc.YPlane(y0=3.0/2.0*pitch, name='top')

    left.boundary_type = 'reflective'
    right.boundary_type = 'reflective'
    top.boundary_type = 'reflective'
    bottom.boundary_type = 'reflective'

    # ----------------------------------------------------------------------
    # Fill pin 1 (the one with gadolinium)
    
    gd_fuel_r = [openmc.ZCylinder(x0=0, y0=0, R=r_rings[i])
                 for i in range(n_rings)]
    gd_clad_ir = openmc.ZCylinder(x0=0, y0=0, R=r_gap)
    gd_clad_or = openmc.ZCylinder(x0=0, y0=0, R=r_clad)

    gd_fuel_cell = openmc.Cell(name='fuel_gd')
    gd_fuel_cell.region = -gd_fuel_r[0]
    volume[gd_fuel_cell.id] = v_ring

    # Gap
    gd_fuel_gap = openmc.Cell(name='gap')
    gd_fuel_gap.region = +gd_fuel_r[n_rings-1] & -gd_clad_ir
    volume[gd_fuel_gap.id] = v_gap

    # Clad
    gd_fuel_clad = openmc.Cell(name='clad')
    gd_fuel_clad.region = +gd_clad_ir & -gd_clad_or
    volume[gd_fuel_clad.id] = v_clad

    # ----------------------------------------------------------------------
    # Fill pin 2, 3 and 4 (without gadolinium)
    coords = [[pitch, 0], [pitch, pitch], [0, pitch], [-pitch, pitch], [-pitch, 0], [-pitch, -pitch], [0, -pitch], [pitch, -pitch]]

    fuel_s = []
    clad_ir_s = []
    clad_or_s = []

    fuel_cell = []
    clad_cell = []
    gap_cell = []

    ind = 0

    for x in coords:
        fuel_s.append(openmc.ZCylinder(x0=x[0], y0=x[1], R=r_fuel))
        clad_ir_s.append(openmc.ZCylinder(x0=x[0], y0=x[1], R=r_gap))
        clad_or_s.append(openmc.ZCylinder(x0=x[0], y0=x[1], R=r_clad))

        fs = openmc.Cell(name='fuel')
        cs = openmc.Cell(name='clad')
        gs = openmc.Cell(name='gap')

        fs.region = -fuel_s[ind]
        gs.region = +fuel_s[ind] & -clad_ir_s[ind]
        cs.region = +clad_ir_s[ind] & -clad_or_s[ind]

        volume[fs.id] = v_fuel
        volume[cs.id] = v_clad
        volume[gs.id] = v_gap
        
        fuel_cell.append(fs)
        clad_cell.append(cs)
        gap_cell.append(gs)
        ind += 1

    # ----------------------------------------------------------------------
    # Fill coolant

    cool_cell = openmc.Cell(name='cool')
    cool_cell.region = +clad_or_s[0] & +clad_or_s[1] & +clad_or_s[2] &\
                       +clad_or_s[3] & +clad_or_s[4] & +clad_or_s[5] &\
                       +clad_or_s[6] & +clad_or_s[7] &\
                       +gd_clad_or & +left & -right & +bottom & -top
    volume[cool_cell.id] = (3 * pitch)**2 - 9 * v_fuel - \
        9 * v_gap - 9 * v_clad

    # ----------------------------------------------------------------------
    # Finalize geometry
    root = openmc.Universe(universe_id=0, name='root universe')

    root.add_cells([cool_cell] + clad_cell + gap_cell + fuel_cell +
                   [gd_fuel_cell] + [gd_fuel_clad] + [gd_fuel_gap])

    geometry = openmc.Geometry()
    geometry.root_universe = root
    return geometry, volume
