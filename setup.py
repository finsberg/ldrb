# System imports
import os
import sys
import platform

from setuptools import setup

# Version number
major = 0
minor = 1

REQUIREMENTS = ['h5py==2.8.0',
                'numba==0.40.1',
                'numpy>=1.7',
                'numpy-quaternion==2018.7.5.21.55.13']

on_rtd = os.environ.get("READTHEDOCS") == "True"
scripts = []

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*\n' % os.path.split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)


setup(
    name="ldrb",
    version="{0}.{1}".format(major, minor),
    description="""
      Laplace-Dirichlet Rule-based algorithm for assigning myocardial
      fiber orientations.
      """,
    author="Henrik Finsberg",
    author_email="henriknf@simula.no",
    license="LGPL version 3 or later",
    install_requires=REQUIREMENTS,
    # dependency_links=dependency_links,
    packages=["ldrb"],
    package_dir={"ldrb": "ldrb"},
)
