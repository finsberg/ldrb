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
2016.x or 2017.x. Next you can install `ldrb` package using

.. code::

    pip install git+https://github.com/finsberg/ldrb.git

Alternatively, you can clone / download the repository at `<https://github.com/finsberg/ldrb>`_
and install the dependencies

.. code::

    pip install -r requirements.txt

and finally you can instll the `ldrb` package using either

.. code::

    pip install .

or

.. code::

    python setup.py install


You can also install the library using conda

.. code::

   conda install -c finsberg ldrb


.. toctree::
    :maxdepth: 2
    :caption: Demos

    demo

.. toctree::
   :maxdepth: 1
   :caption: Programmers reference:

   ldrb



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======
MIT License

Contributors
============
For questions please contact Henrik Finsberg (henriknf@simula.no)
