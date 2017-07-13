FROM ledatelescope/bifrost:gpu-base-pypy

MAINTAINER Ben Barsdell <benbarsdell@gmail.com>

ARG DEBIAN_FRONTEND=noninteractive

ENV TERM xterm

ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}

WORKDIR /tmp

RUN git clone https://github.com/ledatelescope/bifrost.git && \
    cd bifrost && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/bifrost

# IPython
EXPOSE 8888

WORKDIR /workspace

RUN ["/bin/bash"]
