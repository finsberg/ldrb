# Demo


Here you will find the demos for generating fibers using the LDRB algorithm. All demos should work in serial and parallel using `mpirun`. This means that you can run the demos either using e.g

```
python demo_lv.py
```
or using `mpirun`

```
mpirun -n 4 python demo_lv.py
```
which will run the demo using 4 processors.

There are currently three demos that you can find below

- [Simple LV](demo_lv)
- [Simple BiV](demo_biv)
- [Patient specific LV](demo_patient_lv)


Below we also show some example plots of the fiber orientations

## LV
![LV Fiber](_static/figures/lv_fiber.png)

![LV Sheet](_static/figures/lv_sheet.png)

![LV Sheet-normal](_static/figures/lv_sheet_normal.png)


## BiV

![BiV Fiber](_static/figures/biv_fiber.png)

# Patient

![Patient Fiber](_static/figures/patient_fiber_lv.png)
