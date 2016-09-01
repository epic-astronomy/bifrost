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
from bifrost.block import (
        NumpyBlock, TestingBlock, GainSolveBlock, NearestNeighborGriddingBlock,
        IFFT2Block, Pipeline)
from bifrost.addon.leda.blocks import (
    ScalarSkyModelBlock, DELAYS, COORDINATES, DISPERSIONS, UVCoordinateBlock,
    load_telescope, LEDA_SETTINGS_FILE, OVRO_EPHEM, NewDadaReadBlock, CableDelayBlock,
    BAD_STANDS, ImagingBlock)
import hickle as hkl

def load_sources():
    sources = {}
    with open('/data1/mcranmer/data/real/vlssr_catalog.txt', 'r') as file_in:
        iterator = 0
        for line in file_in:
            iterator += 1
            if iterator == 17:
                break
        iterator = 0
        number_sources = 100000
        total_flux = 1e-10
        for line in file_in:
            iterator += 1
            if iterator > number_sources:
                #break
                pass
            if line[0].isspace():
                continue
            try:
                flux = float(line[29:36].replace(" ",""))
            except:
                break
            if flux > 1:
                source_string = line[:-1]
                ra = line[:2]+':'+line[3:5]+':'+line[6:11]
                ra = ra.replace(" ", "")
                dec = line[12:15]+':'+line[16:18]+':'+line[19:23]
                dec = dec.replace(" ", "")
                total_flux += flux
                sources[str(iterator)] = {
                    'ra': ra, 'dec': dec,
                    'flux': flux, 'frequency': 74e6, 
                    'spectral index': -0.7}
    return sources

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

def transpose_to_gain_solve(data_array):
    """Transpose the DADA data to the gain_solve format"""
    return data_array.transpose((0, 1, 3, 2, 4))

sources = {}
sources['cyg'] = {
    'ra':'19:59:28.4', 'dec':'+40:44:02.1', 'flux': 10571.0, 'frequency': 58e6,
    'spectral index': -0.2046}
sources['cas'] = {
    'ra': '23:23:27.8', 'dec': '+58:48:34',
    'flux': 6052.0, 'frequency': 58e6, 'spectral index':(+0.7581)}

dada_file = '/data2/hg/interfits/lconverter/WholeSkyL64_47.004_d20150203_utc181702_test/2015-04-08-20_15_03_0001133593833216.dada'
OVRO_EPHEM.date = '2015/04/09 14:34:51'
cfreq = 47.004e6
bandwidth = 2.616e6
df = bandwidth/109.0
output_channels = np.array([5])
nstand = 256
nchan = 1
npol = 2
frequencies = cfreq - bandwidth/2 + df*output_channels
flags = 2*np.ones(shape=[1, nstand]).astype(np.int8)
for stand in BAD_STANDS:
    flags[0, stand] = 1
jones = np.ones([nchan, npol, nstand, npol]).astype(np.complex64)
jones[:, 0, :, 1] = 0
jones[:, 1, :, 0] = 0

blocks = []
blocks.append((
    NewDadaReadBlock(dada_file, output_chans=output_channels, time_steps=1),
    {'out': 'raw_visibilities'}))
blocks.append((
    CableDelayBlock(frequencies, DELAYS, DISPERSIONS),
    {'in':'raw_visibilities', 'out':'visibilities'}))
blocks.append((
    NumpyBlock(transpose_to_gain_solve),
    {'in_1': 'visibilities', 'out_1': 'formatted_visibilities'}))
blocks.append((
    ScalarSkyModelBlock(OVRO_EPHEM, COORDINATES, frequencies, sources),
    [], ['model+uv']))
blocks.append((
    NumpyBlock(slice_away_uv, outputs=2),
    {'in_1': 'model+uv', 'out_1': 'model', 'out_2': 'uv_coords'}))
blocks.append((TestingBlock(jones), [], ['jones_in']))
view = 'calibrated_data'
blocks.append((
    NumpyBlock(reformat_data_for_gridding, inputs=2),
    {'in_1': view, 'in_2': 'uv_coords', 'out_1': view+'_data_for_gridding'}))
blocks.append((
    NearestNeighborGriddingBlock((256, 256)),
    [view+'_data_for_gridding'], [view+'_grid']))
blocks.append((IFFT2Block(), [view+'_grid'], [view+'_image']))
blocks.append([ImagingBlock('sky.png', np.abs, log=False), {'in': view+'_image'}])
def print_stats(array):
    print "SNR:", np.max(np.abs(array))/np.average(np.abs(array))
blocks.append([NumpyBlock(print_stats, outputs=0), {'in_1': view+'_image'}])

def baseline_threshold_flagger(allmodel, model, data):
    if np.median(np.abs(model)) == 0: 
        return model, data
    data = data/np.median(np.abs(data[:, :, 0, :, 0]))
    model = model/np.median(np.abs(model[:, :, 0, :, 0]))
    allmodel = allmodel/np.median(np.abs(allmodel[:, :, 0, :, 0]))
    flags = np.abs(data[:, :, 0, :, 0]) > 10#15*np.abs(allmodel[:, :, 0, :, 0])
    print np.sum(flags)
    flagged_model = np.copy(model)
    flagged_data = np.copy(data)
    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        flagged_model[:, :, x, :, y][flags] = 0
        flagged_data[:, :, x, :, y][flags] = 0
    return flagged_model, flagged_data


#Load all sources
"""
allsources = load_sources()
blocks.append((
    ScalarSkyModelBlock(OVRO_EPHEM, COORDINATES, frequencies, allsources),
    [], ['allmodel+uv']))
def dump_to_file(all_vis):
    hkl.dump(all_vis, 'all_sources.hkl', 'w')
blocks.append((NumpyBlock(dump_to_file, outputs=0), {'in_1': 'allmodel'}))
blocks.append((
    NumpyBlock(slice_away_uv, outputs=2),
    {'in_1': 'allmodel+uv', 'out_1': 'allmodel', 'out_2': 'trash1'}))
"""
del sources['cyg']
blocks.append((
    TestingBlock(hkl.load('all_sources.hkl'), complex_numbers=True),
    [], ['allmodel']))
#flag single source baselines based on full model
blocks.append([
    NumpyBlock(baseline_threshold_flagger, inputs=3, outputs=2),
    {'in_1': 'allmodel', 'in_2':'model', 'in_3': 'formatted_visibilities',
    'out_1': 'flagged_model', 'out_2': 'flagged_visibilities'}])
blocks.append([
    GainSolveBlock(flags=flags, eps=0.5, max_iterations=50, l2reg=0.0),
    {'in_data': 'flagged_visibilities', 'in_model': 'flagged_model',
     'in_jones': 'jones_in', 'out_data': 'calibrated_data',
     'out_jones': 'jones_out'}])

Pipeline(blocks).main()
