FROM nvidia/cuda:8.0

MAINTAINER Ben Barsdell <benbarsdell@gmail.com>

ARG DEBIAN_FRONTEND=noninteractive

# Get dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        git \
        pkg-config \
	python-dev \
        libfreetype6-dev \
        software-properties-common \
        exuberant-ctags \
        libpng-dev \
        gfortran \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYPY_VERSION 5.8.0
# Get pypy
RUN wget -O pypy.tar.bz2 "https://bitbucket.org/pypy/pypy/downloads/pypy2-v${PYPY_VERSION}-linux64.tar.bz2" && \
    tar -xjC /usr/local --strip-components=1 -f pypy.tar.bz2 && \
    rm pypy.tar.bz2

# Make python link to pypy
RUN export PYTHON=$(which python) && \
    rm $PYTHON && \
    ln -s $(which pypy) $PYTHON

RUN pypy -m ensurepip && \
    pypy -mpip install -U wheel && \
    pypy -mpip install --upgrade pip && \
    pypy -mpip --no-cache-dir install \
        cython \
        numpy \
        setuptools \
        contextlib2 \
        simplejson \
        pint \
        graphviz \ 
        git+https://github.com/davidjamesca/ctypesgen.git@3d2d9803339503d2988382aa861b47a6a4872c32

RUN pypy -m pip --no-cache-dir install \
	matplotlib

ENV TERM xterm

ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}

# IPython
EXPOSE 8888

RUN ["/bin/bash"]
