import os
from textwrap import dedent

import dolfin as df
import h5py
import numpy as np

from .utils import mpi_comm_world
from .utils import value_size

__author__ = "Henrik Finsberg (henriknf@simula.no)"


body = dedent(
    """<?xml version="1.0"?>
    <Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
        <Domain>
            {body}
        </Domain>
    </Xdmf>""",
)

series = dedent(
    """
    <Grid Name="{name}" GridType="Collection" CollectionType="Temporal">
        <Time TimeType="List">
            <DataItem Format="XML" Dimensions="{N}"> {lst}</DataItem>
        </Time>
    {entry}
    </Grid>
    """,
)


entry_single = dedent(
    """
    <Grid Name="time_{iter}" GridType="Uniform">
        {frame}
    </Grid>
    """,
)

entry = dedent(
    """
    <Grid Name="time_{iter}" GridType="Uniform">
        {frame}
    </Grid>
    """,
)


topology = dedent(
    """
    <Topology NumberOfElements="{ncells}" TopologyType="{cell}">
        <DataItem Dimensions="{ncells} {dim}" Format="HDF">{h5name}:/{h5group}</DataItem>
    </Topology>
    """,
)


topology_polyvert = dedent(
    """
    <Topology TopologyType="Polyvertex" NodesPerElement="{nverts}">
    </Topology>
    """,
)

geometry = dedent(
    """
    <Geometry GeometryType="{coords}">
        <DataItem Dimensions="{nverts} {dim}" Format="HDF">{h5name}:/{h5group}</DataItem>
    </Geometry>
    """,
)

vector_attribute = dedent(
    """
    <Attribute Name="{name}" AttributeType="Vector" Center="{center}">
        <DataItem Format="HDF" Dimensions="{nverts} {dim}">{h5name}:/{h5group}</DataItem>
    </Attribute>
    """,
)

scalar_attribute = dedent(
    """
    <Attribute Name="{name}" AttributeType="Scalar" Center="{center}">
        <DataItem Format="HDF" Dimensions="{nverts} {dim}">{h5name}:/{h5group}</DataItem>
    </Attribute>
    """,
)


def dolfin_to_hd5(obj, h5name, time="", comm=mpi_comm_world(), name=None):
    """
    Save object to and HDF file.

    Parameters
    ----------

    obj : dolfin.Mesh or dolfin.Function
        The object you want to save
    name : str
        Name of the object
    h5group : str
        The folder you want to save the object
        withing the HDF file. Default: ''

    """

    name = obj.name() if name is None else name
    df.info("Save {0} to {1}:{0}/{2}".format(name, h5name, time))

    group = name if time == "" else "/".join([name, str(time)])
    file_mode = "a" if os.path.isfile(h5name) else "w"

    if isinstance(obj, df.Function):

        if value_size(obj) == 1:
            return save_scalar_function(comm, obj, h5name, group, file_mode)
        else:
            return save_vector_function(comm, obj, h5name, group, file_mode)

    elif isinstance(obj, df.Mesh):
        with df.HDF5File(comm, h5name, file_mode) as h5file:
            h5file.write(obj, group)

        if obj.geometry().dim() == 3:
            coords = "XYZ"
        elif obj.geometry().dim() == 2:
            coords = "XY"
        else:
            coords = "X"

        return {
            "h5group": group,
            "cell_indices": "/".join([group, "cell_indices"]),
            "coordinates": "/".join([group, "coordinates"]),
            "topology": "/".join([group, "topology"]),
            "ncells": obj.num_cells(),
            "nverts": obj.num_vertices(),
            "coords": coords,
            "type": "mesh",
            "cell": str(obj.ufl_cell()).capitalize(),
            "top_dim": obj.topology().dim() + 1,
            "geo_dim": obj.geometry().dim(),
        }
    else:

        raise ValueError("Unknown type {}".format(type(obj)))


def save_scalar_function(comm, obj, h5name, h5group="", file_mode="w"):

    V = obj.function_space()

    dim = V.mesh().geometry().dim()
    # TODO: gather
    coords_tmp = V.tabulate_dof_coordinates()
    coords = coords_tmp.reshape((-1, dim))
    # TODO: gather
    obj_arr = obj.vector().get_local()
    vecs = np.array(obj_arr).T

    coord_group = "/".join([h5group, "coordinates"])
    vector_group = "/".join([h5group, "vector"])
    if comm.rank == 0:
        with h5py.File(h5name, file_mode) as h5file:
            if h5group in h5file:
                del h5file[h5group]

            h5file.create_dataset(coord_group, data=coords)
            h5file.create_dataset(vector_group, data=vecs)

    element = obj.ufl_element()

    if dim == 3:
        coords = "XYZ"
    elif dim == 2:
        coords = "XY"
    else:
        coords = "X"

    return {
        "h5group": h5group,
        "coordinates": coord_group,
        "vector": vector_group,
        "nverts": obj.vector().size(),
        "dim": 1,
        "family": element.family(),
        "geo_dim": dim,
        "coords": coords,
        "degree": element.degree(),
        "type": "scalar",
    }


def save_vector_function(comm, obj, h5name, h5group="", file_mode="w"):

    V = obj.function_space()
    gs = obj.split(deepcopy=True)

    W = V.sub(0).collapse()
    dim = V.mesh().geometry().dim()
    # TODO: gather
    coords_tmp = W.tabulate_dof_coordinates()
    coords = coords_tmp.reshape((-1, dim))
    # TODO: gather
    us = [g.vector().get_local() for g in gs]
    vecs = np.array(us).T

    coord_group = "/".join([h5group, "coordinates"])
    vector_group = "/".join([h5group, "vector"])
    if comm.rank == 0:
        with h5py.File(h5name, file_mode) as h5file:

            if h5group in h5file:
                del h5file[h5group]
            h5file.create_dataset(coord_group, data=coords)
            h5file.create_dataset(vector_group, data=vecs)

    element = obj.ufl_element()

    if dim == 3:
        coords = "XYZ"
    elif dim == 2:
        coords = "XY"
    else:
        coords = "X"

    return {
        "h5group": h5group,
        "coordinates": coord_group,
        "vector": vector_group,
        "nverts": obj.vector().size() / dim,
        "dim": dim,
        "family": element.family(),
        "geo_dim": dim,
        "coords": coords,
        "degree": element.degree(),
        "type": "vector",
    }


def load_dict_from_h5(fname, h5group="", comm=mpi_comm_world()):
    """
    Load the given h5file into
    a dictionary
    """

    assert os.path.isfile(fname), "File {} does not exist".format(fname)

    with h5py.File(fname, "r") as h5file:

        def h52dict(hdf):
            if isinstance(hdf, h5py._hl.group.Group):
                t = {}

                for key in hdf.keys():
                    t[str(key)] = h52dict(hdf[key])

            elif isinstance(hdf, h5py._hl.dataset.Dataset):
                t = np.array(hdf)

            return t

        if h5group != "" and h5group in h5file:
            d = h52dict(h5file[h5group])
        else:
            d = h52dict(h5file)

    return d


def fun_to_xdmf(fun, fname, name="function"):

    h5name = "{}.h5".format(fname)
    dolfin_to_hd5(fun, h5name, name=name)

    dim = fun.function_space().mesh().geometry().dim()

    if value_size(fun) == 1:
        nverts = len(fun.vector())
        fun_str = scalar_attribute.format(
            name=name,
            nverts=nverts,
            center="Node",
            h5group="/".join([name, "vector"]),
            dim=1,
            h5name=os.path.basename(h5name),
        )
    else:
        nverts = int(len(fun.vector()) / dim)
        fun_str = vector_attribute.format(
            name=name,
            nverts=nverts,
            dim=dim,
            h5group="/".join([name, "vector"]),
            center="Node",
            h5name=os.path.basename(h5name),
        )

    fun_top = topology_polyvert.format(nverts=nverts)
    fun_geo = geometry.format(
        nverts=nverts,
        dim=dim,
        coords="XYZ",
        h5group="/".join([name, "coordinates"]),
        h5name=os.path.basename(h5name),
    )

    fun_entry = entry.format(frame=fun_geo + fun_top + fun_str, iter=0)
    T = body.format(body=fun_entry, name="Visualzation of {}".format(name))

    with open("{}.xdmf".format(fname), "w") as f:
        f.write(T)


def fiber_to_xdmf(fun, fname, comm=mpi_comm_world()):

    h5name = "{}.h5".format(fname)

    if os.path.isfile(h5name):
        if comm.rank == 0:
            os.unlink(h5name)

    dolfin_to_hd5(fun, h5name, name="fiber")

    fx = fun.split(deepcopy=True)[0]
    # TODO: gather
    fx_arr = fx.vector().get_local()
    scalar = np.arcsin(-fx_arr) * 180 / np.pi
    with h5py.File(h5name, "a") as h5file:
        if comm.rank == 0:
            h5file.create_dataset("fiber/scalar", data=scalar)

    dim = fun.function_space().mesh().geometry().dim()
    nverts = int(fun.vector().size() / dim)
    name = "fiber"

    fun_scal = scalar_attribute.format(
        name="angle",
        nverts=nverts,
        center="Node",
        h5group="/".join([name, "scalar"]),
        dim=1,
        h5name=os.path.basename(h5name),
    )

    fun_vec = vector_attribute.format(
        name=name,
        nverts=nverts,
        dim=dim,
        center="Node",
        h5group="/".join([name, "vector"]),
        h5name=os.path.basename(h5name),
    )

    fun_top = topology_polyvert.format(nverts=nverts)
    fun_geo = geometry.format(
        nverts=nverts,
        dim=dim,
        coords="XYZ",
        h5group="/".join([name, "coordinates"]),
        h5name=os.path.basename(h5name),
    )

    fun_entry = entry_single.format(
        frame=fun_geo + fun_top + fun_scal + fun_vec,
        iter=0,
    )
    T = body.format(body=fun_entry, name="Visualzation of {}".format(name))

    with open("{}.xdmf".format(fname), "w") as f:
        f.write(T)
