.. ldrb documentation master file, created by
   sphinx-quickstart on Mon Dec 10 20:48:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Laplace-Dirichlet Rule-Based (LDRB) algorithm for assigning myocardial fiber orientations
=========================================================================================

A software for assigning myocardial fiber orientations based on the Laplace Dirichlet Ruled-Based algorithm [1]_

.. [1] Bayer, J.D., Blake, R.C., Plank, G. and Trayanova, N.A., 2012.
           A novel rule-based algorithm for assigning myocardial fiber orientation
           to computational heart models. Annals of biomedical engineering, 40(10), pp.2243-2254.
           (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3518842/)

Installation
------------
In order to install the software you need to have installed `FEniCS <https://fenicsproject.org>`_ verision
2016.x or 2017.x. The `ldrb` package can be installed with `pip`

.. code::

    pip install ldrb

or if you need the most recent version you can install the source

.. code::

    pip install git+https://github.com/finsberg/ldrb.git

You can also install the library using conda

.. code::

   conda install -c finsberg ldrb

However, note that there are some problems with the 2017 version of FEniCS on conda. 
If you want a working conda environment with FEniCS 2017 check out
`this gist <https://gist.github.com/finsberg/96eeb1d564aab4a73f53a46a1588e6a6>`_


.. toctree::
    :maxdepth: 2
    :caption: Demos

    demo

.. toctree::
   :maxdepth: 1
   :caption: Programmers reference:

   ldrb


Source code
-----------
Source code is avaible at GitHub https://github.com/finsberg/ldrb



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======
LGPL version 3 or later

Contributors
============
For questions please contact Henrik Finsberg (henriknf@simula.no)
