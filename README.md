[![Anaconda-Server Badge](https://anaconda.org/conda-forge/ldrb/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge)
[![CI](https://github.com/finsberg/ldrb/actions/workflows/main.yml/badge.svg)](https://github.com/finsberg/ldrb/actions/workflows/main.yml)
[![github pages](https://github.com/finsberg/ldrb/actions/workflows/github-pages.yml/badge.svg)](https://github.com/finsberg/ldrb/actions/workflows/github-pages.yml)
[![codecov](https://codecov.io/gh/finsberg/ldrb/branch/master/graph/badge.svg?token=J69bEFdomc)](https://codecov.io/gh/finsberg/ldrb)

# Laplace-Dirichlet Rule-Based (LDRB) algorithm for assigning myocardial fiber orientations


A software for assigning myocardial fiber orientations based on the Laplace Dirichlet Ruled-Based algorithm.

> Bayer, J.D., Blake, R.C., Plank, G. and Trayanova, N.A., 2012.
> A novel rule-based algorithm for assigning myocardial fiber orientation
>to computational heart models. Annals of biomedical engineering, 40(10),
pp.2243-2254.(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3518842/)

```python
# Decide on the angles you want to use
angles = dict(
    alpha_endo_lv=30,  # Fiber angle on the LV endocardium
    alpha_epi_lv=-30,  # Fiber angle on the LV epicardium
    beta_endo_lv=0,  # Sheet angle on the LV endocardium
    beta_epi_lv=0,  # Sheet angle on the LV epicardium
    alpha_endo_sept=60,  # Fiber angle on the Septum endocardium
    alpha_epi_sept=-60,  # Fiber angle on the Septum epicardium
    beta_endo_sept=0,  # Sheet angle on the Septum endocardium
    beta_epi_sept=0,  # Sheet angle on the Septum epicardium
    alpha_endo_rv=80,  # Fiber angle on the RV endocardium
    alpha_epi_rv=-80,  # Fiber angle on the RV epicardium
    beta_endo_rv=0,  # Sheet angle on the RV endocardium
    beta_epi_rv=0,  # Sheet angle on the RV epicardium
)

# Choose space for the fiber fields
# This is a string on the form {family}_{degree}
fiber_space = "P_2"

# Compute the microstructure
fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
    mesh=mesh, fiber_space=fiber_space, ffun=ffun, markers=markers, **angles
)
# Store files using a built in xdmf viewer that also works for functions
# defined in quadrature spaces
ldrb.fiber_to_xdmf(fiber, "fiber")
# And visualize it in Paraview
```

![_](https://github.com/finsberg/ldrb/raw/master/docs/_static/figures/biv_fiber.png)

# Installation

## Install with pip
In order to install the software you need to have
installed [FEniCS](https://fenicsproject.org) (versions older than 2016
are not supprted)

The package can be installed with pip.
```
python -m pip install ldrb
```
or if you need the most recent version you can install the source
```
python -m pip install git+https://github.com/finsberg/ldrb.git
```

## Install with conda
Alternatively you can install with conda

```shell
conda install -c conda-forge ldrb
```
which will also install FEniCS through `conda`.

# Documetation
Documentation is hosted at http://finsberg.github.io/ldrb

# Getting started
Check out the [demos](https://henrikfinsberg.com/ldrb/demo_lv.html)

# License
`ldrb` is licensed under the GNU LGPL, version 3 or (at your option) any later version.
`ldrb` is Copyright (2011-2019) by the authors and Simula Research Laboratory.

# Contributors
Henrik Finsberg (henriknf@simula.no)
