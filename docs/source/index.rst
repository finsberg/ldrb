|InstallConda| |CircleCI| |Documentation Status| |Platform| |codecov|

Laplace-Dirichlet Rule-Based (LDRB) algorithm for assigning myocardial fiber orientations
=========================================================================================

A software for assigning myocardial fiber orientations based on the
Laplace Dirichlet Ruled-Based algorithm.

   Bayer, J.D., Blake, R.C., Plank, G. and Trayanova, N.A., 2012. A
   novel rule-based algorithm for assigning myocardial fiber orientation
   to computational heart models. Annals of biomedical engineering,
   40(10),
   pp.2243-2254.(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3518842/)

.. code:: python

   # Decide on the angles you want to use
   angles = dict(alpha_endo_lv=30,    # Fiber angle on the LV endocardium
                 alpha_epi_lv=-30,    # Fiber angle on the LV epicardium
                 beta_endo_lv=0,      # Sheet angle on the LV endocardium
                 beta_epi_lv=0,       # Sheet angle on the LV epicardium
                 alpha_endo_sept=60,  # Fiber angle on the Septum endocardium
                 alpha_epi_sept=-60,  # Fiber angle on the Septum epicardium
                 beta_endo_sept=0,   # Sheet angle on the Septum endocardium
                 beta_epi_sept=0,   # Sheet angle on the Septum epicardium
                 alpha_endo_rv=80,    # Fiber angle on the RV endocardium
                 alpha_epi_rv=-80,    # Fiber angle on the RV epicardium
                 beta_endo_rv=0,      # Sheet angle on the RV endocardium
                 beta_epi_rv=0)        # Sheet angle on the RV epicardium

   # Choose space for the fiber fields
   # This is a string on the form {family}_{degree}
   fiber_space = 'Quadrature_2'

   # Compute the microstructure
   fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(mesh=mesh,
                                                 fiber_space=fiber_space,
                                                 ffun=ffun,
                                                 markers=markers,
                                                 **angles)
   # Store files using a built in xdmf viewer that also works for functions
   # defined in quadrature spaces
   ldrb.fiber_to_xdmf(fiber, 'fiber')
   # And visualize it in Paraview

|image1|

Installation
============

Install with pip
----------------

In order to install the software you need to have installed
`FEniCS <https://fenicsproject.org>`__ (versions older than 2016 are not
supprted)

The package can be installed with pip.

::

   python -m pip install ldrb

or if you need the most recent version you can install the source

::

   python -m pip install git+https://github.com/finsberg/ldrb.git

Install with conda
------------------

Alternatively you can install with conda

.. code:: shell

   conda install -c finsberg ldrb

Documetation
============

Documentation is hosted at https://ldrb.readthedocs.io.

Getting started
===============

Check out the `demos <demos>`__ and the
`documentation <https://ldrb.readthedocs.io>`__

Known issues
============

If you encounter the following error:

::

   ImportError: numpy.core.multiarray failed to import

see https://github.com/moble/quaternion/issues/72 for how to
troubleshoot.

License
=======

``ldrb`` is licensed under the GNU LGPL, version 3 or (at your option)
any later version. ``ldrb`` is Copyright (2011-2019) by the authors and
Simula Research Laboratory.

Contributors
============

Henrik Finsberg (henriknf@simula.no)

.. |InstallConda| image:: https://anaconda.org/finsberg/ldrb/badges/installer/conda.svg
   :target: https://anaconda.org/finsberg/ldrb
.. |CircleCI| image:: https://circleci.com/gh/finsberg/ldrb.svg?style=shield
   :target: https://circleci.com/gh/finsberg/ldrb
.. |Documentation Status| image:: https://readthedocs.org/projects/ldrb/badge/?version=latest
   :target: https://ldrb.readthedocs.io/en/latest/?badge=latest
.. |Platform| image:: https://anaconda.org/finsberg/ldrb/badges/platforms.svg
   :target: https://anaconda.org/finsberg/ldrb
.. |codecov| image:: https://codecov.io/gh/finsberg/ldrb/branch/master/graph/badge.svg?token=J69bEFdomc
   :target: https://codecov.io/gh/finsberg/ldrb
.. |image1| image:: _static/figures/biv_fiber.png
