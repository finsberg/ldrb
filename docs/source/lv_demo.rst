LV Geometry
===========

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


.. code:: python

    import dolfin as df
    import ldrb

    # Here we just create a lv mesh. Here you can use yor own mesh instead.
    geometry = ldrb.create_lv_mesh()

    # The mesh
    mesh = geometry.mesh
    # The facet function (function with marking for the boundaries)
    ffun = geometry.ffun
    # A dictionary with keys and values for the markers
    markers = geometry.markers

    # Also if you want to to this demo in parallell you should create the mesh
    # in serial and save it to e.g xml
    # df.File('lv_mesh.xml') << mesh

    # And when you run the code in paralall you should load the mesh from the file.
    # mesh = df.Mesh('lv_mesh.xml')

    # Since the markers are the default markers and the facet function is
    # stored within the mesh itself, you can just set
    # markers = None
    # ffun = None


    # Decide on the angles you want to use
    angles = dict(alpha_endo_lv=60,  # Fiber angle on the endocardium
                  alpha_epi_lv=-60,  # Fiber angle on the epicardium
                  beta_endo_lv=0,    # Sheet angle on the endocardium
                  beta_epi_lv=0)     # Sheet angle on the epicardium

    # Choose space for the fiber fields
     # This is a string on the form {family}_{degree}
    fiber_space = 'Quadrature_2'
    # fiber_space = 'Lagrange_1'

    # Compte the microstructure
    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(mesh=mesh,
                                                  fiber_space=fiber_space,
                                                  ffun=ffun,
                                                  markers=markers,
                                                  **angles)

    # Store the results
    df.File('lv_fiber.xml') << fiber
    df.File('lv_sheet.xml') << sheet
    df.File('lv_sheet_normal.xml') << sheet_normal

    # If you run in paralell you should skip the visualization step and do that in
    # serial in stead. In that case you can read the the functions from the xml
    # Using the following code
    # V = ldrb.space_from_string(fiber_space, mesh, dim=3)
    # fiber = df.Function(V, 'lv_fiber.xml')
    # sheet = df.Function(V, 'lv_sheet.xml')
    # sheet_normal = df.Function(V, 'lv_sheet_normal.xml')

    # Store files in XDMF to be visualized in Paraview
    # (These function are not tested in paralell)
    ldrb.fiber_to_xdmf(fiber, 'lv_fiber')
    ldrb.fiber_to_xdmf(sheet, 'lv_sheet')
    ldrb.fiber_to_xdmf(sheet_normal, 'lv_sheet_normal')

.. figure:: _static/figures/lv_fiber.png

    LV Fiber

.. figure:: _static/figures/lv_sheet.png

    LV Sheet

.. figure:: _static/figures/lv_sheet_normal.png

    LV Sheet-normal
