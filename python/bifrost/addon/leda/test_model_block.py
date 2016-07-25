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

import unittest
import ephem
import json
import numpy as np
from model_block import ScalarSkyModelBlock
from bifrost.block import Pipeline, WriteAsciiBlock

ovro = ephem.Observer()
ovro.lat = '37.239782'
ovro.lon = '-118.281679'
ovro.elevation = 1184.134
ovro.date = ephem.now()

def load_telescope(filename):
    with open(filename, 'r') as telescope_file:
        telescope = json.load(telescope_file)
    coords_local = np.array(telescope['coords']['local']['__data__'], dtype=np.float32)
    # Reshape into ant,column
    coords_local = coords_local.reshape(coords_local.size/4,4)
    ant_coords = coords_local[:,1:]
    inputs = np.array(telescope['inputs']['__data__'], dtype=np.float32)
    # Reshape into ant,pol,column
    inputs      = inputs.reshape(inputs.size/7/2,2,7)
    delays      = inputs[:,:,5]*1e-9
    dispersions = inputs[:,:,6]*1e-9
    return telescope, ant_coords, delays, dispersions

#coords are in meterstelescope.
telescope, coords, delays, dispersions = load_telescope("/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json")

class TestScalarSkyModelBlock(unittest.TestCase):
    """Test various functionality of the sky model block"""
    def setUp(self):
        """Generate simple model based on Cygnus A and the sun at 60 MHz"""
        self.blocks = []
        self.sources = {}
        self.sources['cyg'] = {
            'ra':'19:59:28.4', 'dec':'+40:44:02.1', 
            'flux': 10571.0, 'frequency': 58e6, 
            'spectral index': -0.2046}
        self.sources['sun'] = {
            'ephemeris': ephem.Sun(),
            'flux': 250.0, 'frequency':20e6,
            'spectral index':+1.9920}
        frequencies = [60e6]
        self.blocks.append((ScalarSkyModelBlock(ovro, coords, frequencies, self.sources), [], [0]))
        self.blocks.append((WriteAsciiBlock('.log.txt'), [0], []))
        Pipeline(self.blocks).main()
        self.model = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
    def test_output_size(self):
        """Make sure that the visibility output matches the number of baselines"""
        self.assertEqual(self.model.size, 256*256)
    def test_flux(self):
        """Make sure that the flux of the model is large enough"""
        self.assertGreater(np.abs(self.model).sum(), 10571.0)
    def test_phases(self):
        """Phases should be distributed well about the unit circle
        They should therefore cancel eachother out fairly well"""
        self.assertLess(np.abs(self.model.sum()), 100.0)
    def test_multiple_frequences(self):
        frequencies = [60e6, 70e6]
        self.blocks[0] = (ScalarSkyModelBlock(ovro, coords, frequencies, self.sources), [], [0])
        open('.log.txt', 'w').close()
        Pipeline(self.blocks).main()
        self.model = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        self.assertEqual(self.model.size, 2*256*256)
