import dolfin as df
import ldrb


mesh, ffun, markers = ldrb.utils.gmsh2dolfin("slab.msh")

print(markers)

fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
    mesh=mesh,
    fiber_space="CG_1",
    ffun=ffun,
    markers={
        "base": markers["base"][0],
        "apex": markers["apex"][0],
        "epi": markers["epi"][0],
        "lv": markers["lv"][0],
    },
    check_all_boundaries_are_marked=False,
    alpha_endo_lv=60,
    alpha_epi_lv=-60,
    beta_endo_lv=0,
    beta_epi_lv=0,
)

ldrb.save.fiber_to_xdmf(fiber, "SLABfiber")
ldrb.save.fiber_to_xdmf(sheet, "SLABsheet")
ldrb.save.fiber_to_xdmf(sheet_normal, "SLABsheet_normal")
