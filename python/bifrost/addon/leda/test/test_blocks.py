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
import bifrost
import json
import os
import numpy as np
from bifrost.block import WriteAsciiBlock, Pipeline, TestingBlock, NearestNeighborGriddingBlock
from bifrost.addon.leda.blocks import DadaReadBlock, NewDadaReadBlock, CableDelayBlock
from bifrost.addon.leda.blocks import UVCoordinateBlock, BaselineSelectorBlock

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

#LEDA stand flags:
bad_stands = [ 0,56,57,58,59,60,61,62,63,72,74,75,76,77,78,82,83,84,85,86,87,91,92,93,104,120,121,122,123,124,125,126,127,128,145,148,157,161,164,168,184,185,186,187,188,189,190,191,197,220,224,225,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255 ]

class TestDadaBlock(unittest.TestCase):
    """Test the ability of the Dada block to read
        in data that is compatible with other blocks."""
    def setUp(self):
        self.blocks = []
        self.blocks.append(
            (DadaReadBlock(
                "/data1/mcranmer/data/real/leda/2016_xaa.dada"),
            [], [0]))
    def test_read_and_write(self):
        """Reads in a dada file, and logs in ascii
            file."""
        logfile = '.log.txt'
        self.blocks.append((WriteAsciiBlock(logfile), [0], []))
        Pipeline(self.blocks).main() 
        test_bytes = open(logfile, 'r').read(500).split(' ')
        self.assertAlmostEqual(np.float(test_bytes[0]), 3908.5, 3)
    def test_read_copy_write(self):
        """Adds another intermediate block to the
            last step."""
        logfile = '.log.txt'
        self.blocks.append((CopyBlock(), [0], [1, 2, 3]))
        self.blocks.append((WriteAsciiBlock(logfile), [3], []))
        Pipeline(self.blocks).main() 
        test_bytes = open(logfile, 'r').read(500).split(' ')
        self.assertAlmostEqual(np.float(test_bytes[0]), 3908.5, 3)
class TestNewDadaReadBlock(unittest.TestCase):
    """Test the ability of the Dada block to read
        in data that is compatible with other blocks."""
    def setUp(self):
        """Reads in one channel of a dada file, and logs in ascii
            file."""
        self.logfile_visibilities = '.log_vis.txt'
        dadafile = '/data2/hg/interfits/lconverter/WholeSkyL64_47.004_d20150203_utc181702_test/2015-04-08-20_15_03_0001133593833216.dada'
        self.n_stations = 256
        self.n_pol = 2
        self.blocks = []
        self.blocks.append((NewDadaReadBlock(dadafile, output_chans=[100], time_steps=1), 
            {'out': 0}))
        self.blocks.append((WriteAsciiBlock(self.logfile_visibilities ), [0], []))
        Pipeline(self.blocks).main() 
    def test_read_and_write(self):
        """Make sure some data is being written"""
        dumpsize = os.path.getsize(self.logfile_visibilities)
        self.assertGreater(dumpsize, 100)
    def test_output_size(self):
        """Make sure dada read block is putting out full matrix"""
        baseline_visibilities = np.loadtxt(self.logfile_visibilities, dtype=np.float32).view(np.complex64)
        self.assertEqual(baseline_visibilities.size, self.n_pol**2*self.n_stations**2)
    def test_imaging(self):
        """Try to grid and image the data"""
        visibilities = np.loadtxt(self.logfile_visibilities, dtype=np.float32).view(np.complex64)
        visibilities = visibilities.reshape((self.n_stations, self.n_stations, self.n_pol, self.n_pol))[:, :, 0, 0]
        antenna_coordinates = load_telescope("/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json")[1]
        identity_matrix = np.ones((self.n_stations, self.n_stations, 3), dtype=np.float32)
        baselines_xyz = (identity_matrix*antenna_coordinates)-(identity_matrix*antenna_coordinates).transpose((1, 0, 2))
        baselines_u = baselines_xyz[:, :, 0].reshape(-1)
        baselines_v = baselines_xyz[:, :, 1].reshape(-1)
        assert visibilities.dtype == np.complex64
        real_visibilities = visibilities.ravel().view(np.float32)[0::2]
        imaginary_visibilities = visibilities.ravel().view(np.float32)[1::2]
        out_data = np.zeros(shape=[visibilities.size*4]).astype(np.float32)
        out_data[0::4] = baselines_u
        out_data[1::4] = baselines_v
        out_data[2::4] = real_visibilities
        out_data[3::4] = imaginary_visibilities 
        blocks = []
        gridding_shape = (256, 256)
        blocks.append((TestingBlock(out_data), [], [0]))
        blocks.append((NearestNeighborGriddingBlock(gridding_shape), [0], [1]))
        blocks.append((WriteAsciiBlock('.log.txt'), [1], []))
        Pipeline(blocks).main()
        model = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        model = model.reshape(gridding_shape)
        # Should be the size of the desired grid
        self.assertEqual(model.size, np.product(gridding_shape))
        brightness = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(model))))
        # Should be many nonzero elements in the image
        self.assertGreater(brightness[brightness > 1e-30].size, 100)
class TestCableDelayBlock(unittest.TestCase):
    """Test a block which adds cable delays to visibilities"""
    def setUp(self):
        """Set up a file output with and without cable delays"""
        self.logfile_cable_delay = '.log_cables.txt'
        self.logfile_no_cable_delay = '.log_no_cables.txt'
        dadafile = '/data2/hg/interfits/lconverter/WholeSkyL64_47.004_d20150203_utc181702_test/2015-04-08-20_15_03_0001133593833216.dada'
        coordinates, delays, dispersions = load_telescope("/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json")[1:]
        frequencies = 10e5*(47.004-2.616/2) + np.arange(start=0, stop=2.616, step=2.616/109)
        self.n_stations = 256
        self.n_pol = 2
        output_channels = [100]
        self.blocks = []
        self.blocks.append((
            NewDadaReadBlock(dadafile, output_chans=output_channels , time_steps=1),
            {'out': 0}))
        self.blocks.append((
            CableDelayBlock(frequencies[output_channels], delays, dispersions),
            {'in': 0, 'out': 1}))
        self.blocks.append((WriteAsciiBlock(self.logfile_cable_delay), [1], []))
        self.blocks.append((WriteAsciiBlock(self.logfile_no_cable_delay), [0], []))
        Pipeline(self.blocks).main() 
        self.cable_delay_visibilities = np.loadtxt(
            self.logfile_cable_delay,
            dtype=np.float32).view(np.complex64)
        self.no_cable_delay_visibilities = np.loadtxt(
            self.logfile_no_cable_delay,
            dtype=np.float32).view(np.complex64)
    def test_throughput(self):
        """Make sure some data is going through"""
        self.assertEqual(
            self.cable_delay_visibilities.size,
            self.n_pol**2*self.n_stations**2)
    def test_phase_of_data_changing(self):
        """Test that some sort of complex operation is being applied"""
        self.assertGreater(
            np.sum(np.abs(self.cable_delay_visibilities.imag - \
                self.no_cable_delay_visibilities.imag)),
            self.n_stations**2)
class TestUVCoordinateBlock(unittest.TestCase):
    """Test the ability of a block to output UV coordinates"""
    def setUp(self):
        self.logfile_uv_coordinates = '.log_uv.txt'
        self.n_stations = 256
        self.blocks = []
        self.blocks.append((UVCoordinateBlock("/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json"), 
            {'out': 0}))
        self.blocks.append((WriteAsciiBlock(self.logfile_uv_coordinates), [0], []))
        Pipeline(self.blocks).main()
    def test_output_coordinates(self):
        """Make sure dada read block is putting out correct uv coordinates"""
        antenna_coordinates = load_telescope("/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json")[1]
        identity_matrix = np.ones((self.n_stations, self.n_stations, 3), dtype=np.float32)
        baselines_xyz = (identity_matrix*antenna_coordinates)-(identity_matrix*antenna_coordinates).transpose((1, 0, 2))
        baselines_from_file = np.loadtxt(self.logfile_uv_coordinates, dtype=np.float32)
        np.testing.assert_almost_equal(baselines_xyz[:, :, 0:2].ravel(), baselines_from_file)
class TestBaselineSelectorBlock(unittest.TestCase):
    """Test the ability of a block to select only the longest baselines (by setting visibilities to zero)"""
    def test_visibilites_zero(self):
        """Test that the correct baselines are set to zero when the median is selected.
            Perform this with only a minimum baseline"""
        blocks = []
        dadafile = '/data2/hg/interfits/lconverter/WholeSkyL64_47.004_d20150203_utc181702_test/2015-04-08-20_15_03_0001133593833216.dada'
        antenna_coordinates = load_telescope("/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json")[1]
        identity_matrix = np.ones((256, 256, 3), dtype=np.float32)
        baselines_xyz = (identity_matrix*antenna_coordinates)-(identity_matrix*antenna_coordinates).transpose((1, 0, 2))
        median_baseline = np.median(np.abs(baselines_xyz[:, :, 0] + 1j*baselines_xyz[:, :, 1]))
        blocks.append((
            NewDadaReadBlock(dadafile, output_chans=[100], time_steps=1),
            {'out': 'visibilities'}))
        blocks.append((
            UVCoordinateBlock("/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json"), 
            {'out': 'uv_coords'}))
        blocks.append((
            BaselineSelectorBlock(minimum_baseline = median_baseline),
            {'in_vis': 'visibilities', 'in_uv': 'uv_coords', 'out_vis': 'flagged_visibilities'}
            ))
        blocks.append((WriteAsciiBlock('.log.txt'), ['flagged_visibilities'], []))
        Pipeline(blocks).main()
        visibilities = np.loadtxt('.log.txt', dtype=np.float32).view(np.complex64)
        visibilities = visibilities.reshape((1, 256, 256, 2, 2))
        self.assertLess(np.sum([np.abs(visibilities[0, i, i, 0, 0]) for i in range(256)]), 1e-5)
        self.assertGreater(visibilities[np.abs(visibilities)<1e-5].size, visibilities.size*0.45)
        self.assertLess(visibilities[np.abs(visibilities)<1e-5].size, visibilities.size*0.75)
