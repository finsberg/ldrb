"""
In this demo we will show how you can generate fibers using the ldrb algorithm
on a LV mesh. In order to run this demo you also need to install mshr if
you haven't already.
To run the demo in serial do

.. code::

    python demo_lv.py

If you want to run the demo in parallel you should first comment out the lines
that don't work in serial. Say you want to run on 4 cpu's, you run the command:

.. code::

    mpirun -n 4 python demo_lv.py
"""
import dolfin as df
import pulse

import ldrb

# comm = ldrb.utils.mpi_comm_world()

# Here we just create a lv mesh. Here you can use yor own mesh instead.
geometry = ldrb.create_lv_mesh()

# Make fiber field
fiber_params = df.Parameters("Fibers")
fiber_params.add("fiber_space", "CG_1")
# fiber_params.add("fiber_space", "Quadrature_4")
fiber_params.add("include_sheets", False)
fiber_params.add("fiber_angle_epi", -60)
fiber_params.add("fiber_angle_endo", 60)

pulse.geometry_utils.generate_fibers(geometry.mesh, fiber_params)

# The mesh
mesh = geometry.mesh
# The facet function (function with marking for the boundaries)
# ffun = geometry.ffun
# A dictionary with keys and values for the markers
# markers = geometry.markers

# Also if you want to to this demo in parallell you should create the mesh
# in serial and save it to e.g xml
# df.File('lv_mesh.xml') << mesh

# And when you run the code in paralall you should load the mesh from the file.
# mesh = df.Mesh("lv_mesh.xml")

# Since the markers are the default markers and the facet function is
# stored within the mesh itself, you can just set
markers = None
ffun = None


# Decide on the angles you want to use
angles = dict(
    alpha_endo_lv=60,  # Fiber angle on the endocardium
    alpha_epi_lv=-60,  # Fiber angle on the epicardium
    beta_endo_lv=0,  # Sheet angle on the endocardium
    beta_epi_lv=0,
)  # Sheet angle on the epicardium

# Choose space for the fiber fields
# This is a string on the form {family}_{degree}
# fiber_space = 'Quadrature_2'
fiber_space = "Lagrange_1"

# Compte the microstructure
fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
    mesh=mesh, fiber_space=fiber_space, ffun=ffun, markers=markers, **angles
)

# fiber.rename('fiber', 'fibers')
# sheet.rename('sheet', 'fibers')
# sheet_normal.rename('sheet_normal', 'fibers')
# import pulse.geometry_utils as utils


# import numpy as np
# focal = np.sqrt(1.5**2 - 0.5**2)
# sfun = utils.mark_strain_regions(mesh, foc=focal)

# df.File('sfun.pvd') << sfun


# angles['alpha_endo_lv'] = 0
# angles['alpha_epi_lv'] = 0
# circ, _, _ = ldrb.dolfin_ldrb(mesh=mesh,
#                               fiber_space=fiber_space,
#                               ffun=ffun,
#                               markers=markers,
#                               **angles)
# circ.rename("circumferential", "local_basis_function")
# df.File('circ.pvd') << circ

# angles['alpha_endo_lv'] = -90
# angles['alpha_epi_lv'] = -90
# lon, _, _ = ldrb.dolfin_ldrb(mesh=mesh,
#                              fiber_space=fiber_space,
#                              ffun=ffun,
#                              markers=markers,
#                              **angles)
# df.File('long.pvd') << lon
# lon.rename("longitudinal", "local_basis_function")


# def cross(e1, e2):
#     e3 = df.Function(e1.function_space())
#     e1_arr = e1.vector().get_local().reshape((-1, 3))
#     e2_arr = e2.vector().get_local().reshape((-1, 3))

#     crosses = []
#     for c1, c2 in zip(e1_arr, e2_arr):
#         crosses.extend(np.cross(c1, c2.tolist()))

#     e3.vector()[:] = np.array(crosses)[:]
#     return e3


# rad = cross(circ, lon)
# df.File('rad.pvd') << rad
# rad.rename("radial", "local_basis_function")

# mapper = {'lv': 'ENDO', 'epi': 'EPI', 'rv': 'ENDO_RV', 'base': 'BASE'}
# m = {mapper[k]: (v, 2) for k, v in markers.items()}

# utils.save_geometry_to_h5(mesh, 'ellipsoid.h5', markers=m,
#                           fields=[fiber, sheet, sheet_normal],
#                           local_basis=[circ, rad, lon])


# from IPython import embed; embed()
# exit()
# Store the results
df.File("lv_fiber.xml") << fiber
df.File("lv_sheet.xml") << sheet
df.File("lv_sheet_normal.xml") << sheet_normal

# If you run in paralell you should skip the visualization step and do that in
# serial in stead. In that case you can read the the functions from the xml
# Using the following code
V = ldrb.space_from_string(fiber_space, mesh, dim=3)
fiber = df.Function(V, "lv_fiber.xml")
sheet = df.Function(V, "lv_sheet.xml")
sheet_normal = df.Function(V, "lv_sheet_normal.xml")

# Store files in XDMF to be visualized in Paraview
# (These function are not tested in paralell)
ldrb.fiber_to_xdmf(fiber, "lv_fiber")
ldrb.fiber_to_xdmf(sheet, "lv_sheet")
ldrb.fiber_to_xdmf(sheet_normal, "lv_sheet_normal")
