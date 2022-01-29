# Local basis functions

In this demo we will show how to generate local coordinate basis functions in the longitudinal, circumferential and radial direction. These vectors are very important in several applications. For example in ultrasound strain estimation strain in along these directions are important metrics to quantify cardiac function.

Let us first  import the relevant libraries


```python
import ldrb
import dolfin
```


We will show how to to this on an LV and a BiV geometry. The BiV geometry is a bit tricky compared to the LV.


```python
case = "lv"
# case = "biv"
```


And depending on the case we create a geometry


```python
if case == "lv":
    geometry = ldrb.create_lv_mesh()
else:
    geometry = ldrb.create_biv_mesh()
```


And let us also extract the relevant variables


```python
# The mesh
mesh = geometry.mesh
# The facet function (function with marking for the boundaries)
ffun = geometry.ffun
# A dictionary with keys and values for the markers
markers = geometry.markers
# Space for the vector functions
space = "P_2"
```


In the case of a BiV we do a little trick where me mark the RV as LV in order to find the longitudinal vector field. We run the  algorithm again with some different angles to get the circumferential vector field and finally take the cross product to get the radial vector field.
(if you find a better way of doing this, please submit a PR).


```python
if case == "biv":
    lv_ffun = dolfin.MeshFunction("size_t", mesh, 2)
    lv_ffun.array()[:] = ffun.array().copy()
    lv_ffun.array()[ffun.array() == markers["rv"]] = markers["lv"]
    lv_markers = markers.copy()
    lv_markers.pop("rv")

    long, _, _ = ldrb.dolfin_ldrb(
        mesh=mesh,
        fiber_space=space,
        ffun=lv_ffun,
        markers=lv_markers,
        alpha_endo_lv=-90,
        alpha_epi_lv=-90,
        beta_endo_lv=0,
        beta_epi_lv=0,
    )

    circ, _, _ = ldrb.dolfin_ldrb(
        mesh=mesh,
        fiber_space=space,
        ffun=ffun,
        markers=markers,
        alpha_endo_lv=0,
        alpha_epi_lv=0,
        beta_endo_lv=0,
        beta_epi_lv=0,
    )

    rad = dolfin.project(dolfin.cross(circ, long), circ.function_space())
```


For the single ventricle it is much simpler


```python
else:

    long, circ, rad = ldrb.dolfin_ldrb(
        mesh=mesh,
        fiber_space=space,
        ffun=ffun,
        markers=markers,
        alpha_endo_lv=-90,
        alpha_epi_lv=-90,
        beta_endo_lv=0,
        beta_epi_lv=0,
        save_markers=True,
    )
```


Finally we save the vector fields to a file


```python
ldrb.fiber_to_xdmf(long, "long")
ldrb.fiber_to_xdmf(circ, "circ")
ldrb.fiber_to_xdmf(rad, "ran")
```

<!-- #region -->


This resulting BiV and LV vector field are shown in {numref}`Figure {number} <lv_basis_function>` and {numref}`Figure {number} <biv_basis_function>` respectively.

```{figure} _static/figures/lv_basis_functions.png
---
name: lv_basis_function
---

LV longitudinal, circumferential and radial vector fields
```

```{figure} _static/figures/biv_basis_functions.png
---
name: biv_basis_function
---

BiV longitudinal, circumferential and radial vector fields
```
<!-- #endregion -->
