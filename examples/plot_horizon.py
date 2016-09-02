# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import ephem
from bifrost.block import (
        NumpyBlock, TestingBlock, GainSolveBlock, NearestNeighborGriddingBlock,
        IFFT2Block, Pipeline)
from bifrost.addon.leda.blocks import (
    ScalarSkyModelBlock, DELAYS, COORDINATES, DISPERSIONS, UVCoordinateBlock,
    load_telescope, LEDA_SETTINGS_FILE, OVRO_EPHEM, NewDadaReadBlock, CableDelayBlock,
    BAD_STANDS, ImagingBlock, SPEED_OF_LIGHT)

OVRO_EPHEM.date = '2015/04/09 14:34:51'
def horizon_source(x):
    """Say if the source is below the horizon for LEDA"""
    return OVRO_EPHEM.radec_of(np.pi*x, 0)

def slice_away_uv(model_and_uv):
    """Cut off the uv coordinates from the ScalarSkyModelBlock and reshape to GainSolve"""
    number_stands = model_and_uv.shape[1]
    model = np.zeros(shape=[1, number_stands, 2, number_stands, 2]).astype(np.complex64)
    uv_coords = np.zeros(shape=[number_stands, number_stands, 2]).astype(np.float32)
    model[0, :, 0, :, 0] = model_and_uv[0, :, :, 2]+1j*model_and_uv[0, :, :, 3]
    model[0, :, 1, :, 1] = model[0, :, 0, :, 0]
    uv_coords[:, :, 0:2] = model_and_uv[0, :, :, 0:2]
    return model, uv_coords

def reformat_data_for_gridding(visibilities, uv_coordinates):
    """Reshape visibility data for gridding on UV plane"""
    reformatted_data = np.zeros(shape=[256, 256, 4], dtype=np.float32)
    reformatted_data[:, :, 0] = uv_coordinates[:, :, 0]
    reformatted_data[:, :, 1] = uv_coordinates[:, :, 1]
    reformatted_data[:, :, 2] = np.real(visibilities[0, :, 0, :, 0])
    reformatted_data[:, :, 3] = np.imag(visibilities[0, :, 0, :, 0])
    return reformatted_data

cfreq = 36.54e6
bandwidth = 2.616e6
df = bandwidth/109.0
output_channels = np.array([85])
nstand = 256
nchan = 1
npol = 2
#frequencies = cfreq - bandwidth/2 + df*output_channels
frequencies = 60e6
sources = {}

print np.arcsin(0.5)
for x in np.arange(-1, 1, 0.01):
    sources[str(x)] = {
        'ra': str(horizon_source(x)[0]), 'dec': str(horizon_source(x)[1]),
        'flux': 1000.0, 'frequency': 58e6, 'spectral index': -0.7}

blocks = []
blocks.append((
    ScalarSkyModelBlock(OVRO_EPHEM, COORDINATES, frequencies, sources),
    [], ['model+uv']))
blocks.append((
    NumpyBlock(slice_away_uv, outputs=2),
    {'in_1': 'model+uv', 'out_1': 'initial_model',
     'out_2': 'uv_coords'}))
blocks.append((
    NumpyBlock(reformat_data_for_gridding, inputs=2),
    {'in_1': 'initial_model', 'in_2': 'uv_coords', 'out_1': 'gridding'}))
grid_size = 512
blocks.append((
    NearestNeighborGriddingBlock((grid_size, grid_size)),
    ['gridding'], ['grid']))
blocks.append((IFFT2Block(), ['grid'], ['image']))
blocks.append([ImagingBlock('sky.png', np.abs, log=False), {'in': 'image'}])
def brightest_side(array):
    print 2*np.where(np.abs(array) > np.percentile(np.abs(array), 99.9))[0][0]-grid_size
blocks.append((NumpyBlock(brightest_side, outputs=0), {'in_1': 'image'}))
Pipeline(blocks).main()
