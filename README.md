[![CI](https://github.com/finsberg/ldrb/actions/workflows/main.yml/badge.svg)](https://github.com/finsberg/ldrb/actions/workflows/main.yml)
[![github pages](https://github.com/finsberg/ldrb/actions/workflows/github-pages.yml/badge.svg)](https://github.com/finsberg/ldrb/actions/workflows/github-pages.yml)


# Laplace-Dirichlet Rule-Based (LDRB) algorithm for assigning myocardial fiber orientations


A software for assigning myocardial fiber orientations based on the Laplace Dirichlet Ruled-Based algorithm.

> Bayer, J.D., Blake, R.C., Plank, G. and Trayanova, N.A., 2012.
> A novel rule-based algorithm for assigning myocardial fiber orientation
>to computational heart models. Annals of biomedical engineering, 40(10),
pp.2243-2254.(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3518842/)

```python
# Generate an example geometry using https://github.com/ComputationalPhysiology/cardiac_geometries
import cardiac_geometries  # pip install cardiac-geometries
import ldrb

geo = cardiac_geometries.mesh.create_biv_ellipsoid(char_length=0.2)

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

# Convert markers to correct format
markers = {
    "base": geo.markers["BASE"][0],
    "lv": geo.markers["ENDO_LV"][0],
    "rv": geo.markers["ENDO_RV"][0],
    "epi": geo.markers["EPI"][0],
}

# Choose space for the fiber fields
# This is a string on the form {family}_{degree}
fiber_space = "P_2"

# Compute the microstructure
fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
    mesh=geo.mesh, fiber_space=fiber_space, ffun=geo.ffun, markers=markers, **angles
)
# Store files using a built in xdmf viewer that also works for functions
# defined in quadrature spaces
ldrb.fiber_to_xdmf(fiber, "fiber")
# And visualize it in Paraview
```

![_](https://github.com/finsberg/ldrb/raw/main/docs/_static/figures/biv_fiber.png)

# Installation

## pip
In order to install the software you need to have
installed [FEniCS](https://fenicsproject.org) (versions older than 2016
are not supported)

The package can be installed with pip.
```
python3 -m pip install ldrb
```
or if you need the most recent version you can install the source
```
python3 -m pip install git+https://github.com/finsberg/ldrb.git
```

### Issues with h5py
You might run into issues with incompatible version of h5py. To resolve this you can try to first uninstall the existing version
```
python3 -m pip uninstall h5py
```
and then reinstall h5py from source using the command
```
python3 -m pip install h5py --no-binary=h5py
```

## Conda
`ldrb` is also available on `conda`
```
conda install -c conda-forge ldrb
```

## Docker
If you don't already have FEniCS installed you can use one of the provided [docker images](https://github.com/finsberg/ldrb/pkgs/container/ldrb), e.g
```
docker pull ghcr.io/finsberg/ldrb:latest
```
to pull the image and use the following command to start a container and sharing your current directory
```
docker run --rm -v $PWD:/home/shared -w /home/shared -it ghcr.io/finsberg/ldrb:latest
```

# Documentation
Documentation is hosted at http://finsberg.github.io/ldrb

# Getting started
Check out the [demos](http://finsberg.github.io/ldrb/demo_lv.html)

# License
`ldrb` is licensed under the GNU LGPL, version 3 or (at your option) any later version.
`ldrb` is Copyright (2011-2019) by the authors and Simula Research Laboratory.

# Contributors
Henrik Finsberg (henriknf@simula.no)
