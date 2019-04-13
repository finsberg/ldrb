#!/bin/bash

upload=$1

if [ -z "$upload" ]
then
       upload="no"
fi
echo "Upload to anaconda "$upload
anaconda login
conda build purge
rm -rf /Users/henriknf/miniconda3/envs/fenics2017/conda-bld/
conda config --set anaconda_upload $upload
conda build .
if [ $upload == "yes" ]
then
    conda convert --platform linux-64 /Users/henriknf/miniconda3/envs/fenics2017/conda-bld/ma
else
    echo "Done"
fi

anaconda logout
