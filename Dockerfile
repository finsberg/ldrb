FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-08-16

COPY . /tmp/
WORKDIR /tmp/

RUN python3 -m pip install cardiac-geometries
RUN python3 -m pip install --no-cache-dir .
RUN rm -rf /tmp
