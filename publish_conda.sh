#!/bin/sh

# set -ex
set -o pipefail

VERSION=2019.1
URL=https://github.com/finsberg/ldrb/archive/v${VERSION}.tar.gz

go_to_build_dir() {
    if [ ! -z $INPUT_SUBDIR ]; then
        cd $INPUT_SUBDIR
    fi
}

get_shasum(){
    # wget -O- $URL | shasum -a 256
    curl $URL -O| openssl sha256
    rm v${VERSION}.tar.gz
}


upload_package(){
    conda config --set anaconda_upload yes
    anaconda login --username $ANACONDA_USERNAME --password $ANACONDA_PASSWORD
    conda build .
    anaconda logout
}

upload_package
# get_shasum