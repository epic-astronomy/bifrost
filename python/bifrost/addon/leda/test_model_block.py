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
import ctypes
import numpy as np
from bifrost.libbifrost import _bf, _check
from bifrost.GPUArray import GPUArray
from model_block import ScalarSkyModelBlock
from bifrost.block import Pipeline, WriteAsciiBlock, NearestNeighborGriddingBlock
from bifrost.block import TestingBlock, GainSolveBlock
from bifrost.libbifrost import _bf
from bifrost.GPUArray import GPUArray

ovro = ephem.Observer()
ovro.lat = '37.239782'
ovro.lon = '-118.281679'
ovro.elevation = 1184.134
ovro.date = '2016/07/26 16:17:00.00'

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

#coords are in meters
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
    def test_output_size(self):
        """Make sure that the visibility output matches the number of baselines"""
        Pipeline(self.blocks).main()
        model = np.loadtxt('.log.txt').astype(np.float32)
        self.assertEqual(model.size, 256*256*4)
    def test_flux(self):
        """Make sure that the flux of the model is large enough"""
        Pipeline(self.blocks).main()
        model = np.loadtxt('.log.txt').astype(np.float32)
        self.assertGreater(np.abs(model[2::4]+1j*model[3::4]).sum(), 10571.0)
    def test_phases(self):
        """Phases should be distributed well about the unit circle
        They should therefore cancel eachother out fairly well"""
        Pipeline(self.blocks).main()
        model = np.loadtxt('.log.txt').astype(np.float32)
        self.assertLess(np.abs((model[2::4]+1j*model[3::4]).sum()), 1000.0)
    def test_multiple_frequences(self):
        """Attempt to to create models for multiple frequencies"""
        frequencies = [60e6, 70e6]
        self.blocks[0] = (ScalarSkyModelBlock(ovro, coords, frequencies, self.sources), [], [0])
        open('.log.txt', 'w').close()
        Pipeline(self.blocks).main()
        model = np.loadtxt('.log.txt').astype(np.float32)
        self.assertEqual(model.size, 256*256*4*2)
    def test_grid_visibilities(self):
        """Attempt to grid visibilities to a nearest neighbor approach"""
        gridding_shape = (200, 200)
        self.blocks[1] = (NearestNeighborGriddingBlock(gridding_shape), [0], [1])
        self.blocks.append((WriteAsciiBlock('.log.txt'), [1], []))
        Pipeline(self.blocks).main()
        model = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        model = model.reshape(gridding_shape)
        # Should be the size of the desired grid
        self.assertEqual(model.size, np.product(gridding_shape))
        brightness = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(model))))
        # Should be many nonzero elements in the image
        self.assertGreater(brightness[brightness > 1e-30].size, 100)
        # Should be some bright sources
        self.assertGreater(np.max(brightness)/np.average(brightness), 10)
        from matplotlib.image import imsave
        imsave('model.png', brightness)
    def test_many_sources(self):
        """Load in many sources and test the brightness"""
        from itertools import islice
        with open('/data1/mcranmer/data/real/vlssr_catalog.txt', 'r') as file_in:
            iterator = 0
            for line in file_in:
                iterator += 1
                if iterator == 17:
                    break
            iterator = 0
            self.sources = {}
            number_sources = 1000
            for line in file_in:
                iterator += 1
                if iterator > number_sources:
                    break
                if line[0].isspace():
                    continue
                source_string = line[:-1]
                ra = line[:2]+':'+line[3:5]+':'+line[6:11]
                ra = ra.replace(" ", "")
                dec = line[12:15]+':'+line[16:18]+':'+line[19:23]
                dec = dec.replace(" ", "")
                flux = float(line[29:36].replace(" ",""))
                self.sources[str(iterator)] = {
                    'ra': ra, 'dec': dec,
                    'flux': flux, 'frequency': 58e6, 
                    'spectral index': -0.2046}
        frequencies = [60e6]
        self.blocks[0] = (ScalarSkyModelBlock(ovro, coords, frequencies, self.sources), [], [0])
        gridding_shape = (256, 256)
        self.blocks[1] = (NearestNeighborGriddingBlock(gridding_shape), [0], [1])
        open('.log.txt', 'w').close()
        self.blocks.append((WriteAsciiBlock('.log.txt'), [1], []))
        Pipeline(self.blocks).main()
        model = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        model = model.reshape(gridding_shape)
        # Should be the size of the desired grid
        self.assertEqual(model.size, np.product(gridding_shape))
        brightness = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(model))))
        # Should be many nonzero elements in the image
        self.assertGreater(brightness[brightness > 1e-30].size, 100)
        # Should be some brighter sources
        self.assertGreater(np.max(brightness)/np.average(brightness), 5)

class TestGainSolve(unittest.TestCase):
    """Test various functionality of the mitchcal gain solve"""
    def setUp(self):
        """Set up a random gain solve"""
        self.nchan = 1
        self.nstand = 256
        self.npol = 2
        nchan = self.nchan
        nstand = self.nstand
        npol = self.npol
        host_data = 10*np.random.rand(
            nchan, nstand, npol, nstand, npol).astype(np.complex64)
        host_model = 10*np.random.rand(
            nchan, nstand, npol, nstand, npol).astype(np.complex64)
        self.host_jones = 10*np.random.rand(
            nchan, npol, nstand, npol).astype(np.complex64)
        host_flags = (np.ones(shape=[nchan, nstand])*2).astype(np.int8)
        visibilities = GPUArray([nchan, nstand, npol, nstand, npol], np.complex64)
        visibilities.set(host_data)
        model = GPUArray([nchan, nstand, npol, nstand, npol], np.complex64)
        model.set(host_model)
        self.jones = GPUArray([nchan, npol, nstand, npol], np.complex64)
        self.jones.set(self.host_jones)
        flags = GPUArray([nchan, nstand], np.int8)
        flags.set(host_flags)
        num_unconverged_type = ctypes.POINTER(ctypes.c_int)
        num = ctypes.c_int(1)
        num_unconverged = ctypes.cast(ctypes.addressof(num), num_unconverged_type)
        array_jones = self.jones.as_BFarray(100)
        _bf.SolveGains(
            visibilities.as_BFconstarray(), 
            model.as_BFconstarray(), 
            array_jones,
            flags.as_BFarray(100),
            True, 1.0, 1.0, 20, num_unconverged)
        self.jones.buffer = array_jones.data
    def test_throughput_and_exit(self):
        """Make sure that the MitchCal C library can be called
        (This does not actually test the block's functions themselves)"""
        np.testing.assert_almost_equal(
            self.jones.shape,
            [self.nchan, self.npol, self.nstand, self.npol])
    def test_changes_being_made(self):
        """Check that there are some changes being made to the jones matrix"""
        self.assertGreater(
            np.max(np.abs(
                self.jones.get().reshape(-1)-self.host_jones.reshape(-1))),
            1e-3)
class TestGainSolveBlock(unittest.TestCase):
    """Test the gain solve block, which calls mitchcal gain solve"""
    def test_throughput_size(self):
        """Test input/output sizes are compatible"""
        for nchan in range(1, 5):
            blocks = []
            nstand = 256
            npol = 2
            model = np.zeros(shape=[nchan, nstand, npol, nstand, npol]).astype(np.float32)
            data = np.zeros(shape=[nchan, nstand, npol, nstand, npol]).astype(np.float32)
            jones = np.zeros(shape=[nchan, npol, nstand, npol]).astype(np.float32)
            blocks.append((TestingBlock(model), [], ['model']))
            blocks.append((TestingBlock(data), [], ['data']))
            blocks.append((TestingBlock(jones), [], ['jones_in']))
            blocks.append((GainSolveBlock(), ['data', 'model', 'jones_in'], ['jones_out']))
            blocks.append((WriteAsciiBlock('.log.txt'), ['jones_out'], []))
            Pipeline(blocks).main()
            out_jones = np.loadtxt('.log.txt')
            self.assertEqual(out_jones.size, np.product([nchan, npol, nstand, npol]))
