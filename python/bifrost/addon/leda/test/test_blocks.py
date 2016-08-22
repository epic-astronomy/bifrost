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

"""@module test_blocks
This program tests the LEDA specific blocks of Bifrost"""
import unittest
import json
import os
import numpy as np
import ephem
from bifrost.block import WriteAsciiBlock, Pipeline, TestingBlock, NearestNeighborGriddingBlock
from bifrost.block import IFFT2Block, NumpyBlock
from bifrost.addon.leda.blocks import NewDadaReadBlock, CableDelayBlock
from bifrost.addon.leda.blocks import UVCoordinateBlock, BaselineSelectorBlock
from bifrost.addon.leda.blocks import SlicingBlock, ImagingBlock, load_telescope
from bifrost.addon.leda.blocks import ScalarSkyModelBlock
from bifrost.addon.leda.blocks import OVRO_EPHEM, COORDINATES, DELAYS

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
        self.blocks.append((
            NewDadaReadBlock(dadafile, output_chans=[100], time_steps=1),
            {'out': 0}))
        self.blocks.append((WriteAsciiBlock(self.logfile_visibilities), [0], []))
        Pipeline(self.blocks).main()
    def test_read_and_write(self):
        """Make sure some data is being written"""
        dumpsize = os.path.getsize(self.logfile_visibilities)
        self.assertGreater(dumpsize, 100)
    def test_output_size(self):
        """Make sure dada read block is putting out full matrix"""
        baseline_visibilities = np.loadtxt(
            self.logfile_visibilities,
            dtype=np.float32).view(np.complex64)
        self.assertEqual(baseline_visibilities.size, self.n_pol**2*self.n_stations**2)
    def test_imaging(self):
        """Try to grid and image the data"""
        visibilities = np.loadtxt(self.logfile_visibilities, dtype=np.float32).view(np.complex64)
        visibilities = visibilities.reshape(
            (self.n_stations, self.n_stations, self.n_pol, self.n_pol))[:, :, 0, 0]
        antenna_coordinates = load_telescope(LEDA_SETTINGS_FILE)[1]
        identity_matrix = np.ones((self.n_stations, self.n_stations, 3), dtype=np.float32)
        baselines_xyz = (identity_matrix*antenna_coordinates)-\
            (identity_matrix*antenna_coordinates).transpose((1, 0, 2))
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
        frequencies = 10e5*(47.004-2.616/2) + np.arange(start=0, stop=2.616, step=2.616/109)
        self.n_stations = 256
        self.n_pol = 2
        output_channels = [100]
        self.blocks = []
        self.blocks.append((
            NewDadaReadBlock(dadafile, output_chans=output_channels , time_steps=1),
            {'out': 0}))
        self.blocks.append((
            CableDelayBlock(frequencies[output_channels], DELAYS, DISPERSIONS),
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
    def setUp(self):
        """Create a simple test pipeline for the baseline selector"""
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
            BaselineSelectorBlock(minimum_baseline=0),
            {'in_vis': 'visibilities', 'in_uv': 'uv_coords', 'out_vis': 'unflagged_visibilities'}
            ))
        blocks.append((
            BaselineSelectorBlock(minimum_baseline=median_baseline),
            {'in_vis': 'visibilities', 'in_uv': 'uv_coords', 'out_vis': 'flagged_visibilities'}
            ))
        blocks.append((WriteAsciiBlock('.log_flag.txt'), ['flagged_visibilities'], []))
        blocks.append((WriteAsciiBlock('.log.txt'), ['visibilities'], []))
        blocks.append((WriteAsciiBlock('.log_unflag.txt'), ['unflagged_visibilities'], []))
        Pipeline(blocks).main()
    def test_no_change(self):
        """Test minimum zero baseline produces equal visibilities"""
        visibilities = np.loadtxt('.log.txt', dtype=np.float32).view(np.complex64)
        flagged_visibilities = np.loadtxt('.log_unflag.txt', dtype=np.float32).view(np.complex64)
        np.testing.assert_almost_equal(flagged_visibilities, flagged_visibilities)
    def test_visibilites_zero(self):
        """Test that the correct baselines are set to zero when the median is selected.
            Perform this with only a minimum baseline"""
        visibilities = np.loadtxt('.log_flag.txt', dtype=np.float32).view(np.complex64)
        visibilities = visibilities.reshape((1, 256, 256, 2, 2))
        self.assertLess(np.sum([np.abs(visibilities[0, i, i, 0, 0]) for i in range(256)]), 1e-5)
        self.assertGreater(visibilities[np.abs(visibilities)<1e-5].size, visibilities.size*0.45)
        self.assertLess(visibilities[np.abs(visibilities)<1e-5].size, visibilities.size*0.75)
class TestImagingBlock(unittest.TestCase):
    """Test the ability of a block to produce an image of visibility data"""
    def test_imaging_pipeline(self):
        """Have an entire imaging pipeline into a png file within Bifrost"""
        blocks = []
        dadafile = '/data2/hg/interfits/lconverter/WholeSkyL64_47.004_d20150203_utc181702_test/2015-04-08-20_15_03_0001133593833216.dada'
        blocks.append((
            NewDadaReadBlock(dadafile, output_chans=[100], time_steps=1),
            {'out': 'visibilities'}))
        blocks.append((
            SlicingBlock(np.s_[0, :, :, 0, 0]),
            {'in': 'visibilities', 'out': 'scalar_visibilities'}))
        blocks.append((
            NearestNeighborGriddingBlock((256, 256)),
            ['scalar_visibilities'],
            ['grid']))
        blocks.append((
            IFFT2Block(),
            ['grid'],
            ['ifftd']))
        blocks.append((
            ImagingBlock(filename='my_sky.png', reduction=np.abs, log=True),
            {'in': 'ifftd'}))
        open('my_sky.png', 'w').close()
        Pipeline(blocks).main()
        image_size = os.path.getsize('my_sky.png')
        self.assertGreater(image_size, 1000)
class TestSlicingBlock(unittest.TestCase):
    """Make sure the slicing block performs as it should"""
    def test_simple_slice(self):
        """Put through a known array, and check output matches correctly"""
        blocks = []
        blocks.append((TestingBlock([[1, 2, 3], [4, 5, 6]]), [], [0]))
        blocks.append((
            SlicingBlock(np.s_[:, 2]),
            {'in': 0, 'out': '[3, 6]'}))
        blocks.append((
            SlicingBlock(np.s_[:1, 0::2]),
            {'in': 0, 'out': '[1, 3]'}))
        blocks.append((WriteAsciiBlock('.log36.txt'), ['[3, 6]'], []))
        blocks.append((WriteAsciiBlock('.log13.txt'), ['[1, 3]'], []))
        Pipeline(blocks).main()
        log_36 = np.loadtxt('.log36.txt')
        log_13 = np.loadtxt('.log13.txt')
        np.testing.assert_almost_equal(log_36, [3, 6])
        np.testing.assert_almost_equal(log_13, [1, 3])
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
        self.blocks.append((ScalarSkyModelBlock(OVRO_EPHEM, COORDINATES, frequencies, self.sources), [], [0]))
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
        self.blocks[0] = (ScalarSkyModelBlock(OVRO_EPHEM, COORDINATES, frequencies, self.sources), [], [0])
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
        self.blocks[0] = (ScalarSkyModelBlock(OVRO_EPHEM, COORDINATES, frequencies, self.sources), [], [0])
        gridding_shape = (256, 256)
        self.blocks[1] = (NearestNeighborGriddingBlock(gridding_shape), [0], [1])
        def assert_brightness(model):
            # Should be the size of the desired grid
            self.assertEqual(model.size, np.product(gridding_shape))
            brightness = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(model))))
            # Should be many nonzero elements in the image
            self.assertGreater(brightness[brightness > 1e-30].size, 100)
            # Should be some brighter sources
            self.assertGreater(np.max(brightness)/np.average(brightness), 5)
        self.blocks.append((NumpyBlock(assert_brightness, outputs=0), {'in_1': 1}))
        Pipeline(self.blocks).main()
    def test_source_outside_fov(self):
        """Load in a source on the other side of the planet, and test it is not visible"""
        no_sources = {}
        fake_sources = {}
        fake_sources['my_fake_source'] = {
            'ra': 0, 'dec': '-89:00:00.0',
            'flux': 1000000.0, 'frequency': 58e6, 
            'spectral index': -0.2046}
        frequencies = [40e6]
        self.blocks[0] = (ScalarSkyModelBlock(OVRO_EPHEM, COORDINATES, frequencies, fake_sources), [], [0])
        self.blocks[1] = (ScalarSkyModelBlock(OVRO_EPHEM, COORDINATES, frequencies, no_sources), [], [1])
        gridding_shape = (256, 256)
        self.blocks.append((NearestNeighborGriddingBlock(gridding_shape), [0], ['fake']))
        self.blocks.append((NearestNeighborGriddingBlock(gridding_shape), [1], ['none']))
        def assert_both_zero(model_no_source, model_invisible_source):
            """Make sure that the two models are both zero in brightness"""
            self.assertAlmostEqual(np.max(model_no_source), 0)
            self.assertAlmostEqual(np.max(model_invisible_source), 0)
        self.blocks.append((NumpyBlock(assert_both_zero, inputs=2, outputs=0), {'in_1':'none', 'in_2':'fake'}))
        Pipeline(self.blocks).main()
    def test_source_scaling(self):
        """Make sure that different frequencies lead different brightnesses"""
        negative_spectral_index = -100
        fake_sources = {}
        fake_sources['my_fake_source'] = {
            'ra':'19:59:28.4', 'dec':'+40:44:02.1', 
            'flux': 1000.0, 'frequency': 58e6, 
            'spectral index': negative_spectral_index}
        frequencies = [40e6, 1e9]
        self.blocks[0] = (ScalarSkyModelBlock(OVRO_EPHEM, COORDINATES, frequencies, fake_sources), [], [0])
        gridding_shape = (256, 256)
        self.blocks.append((NearestNeighborGriddingBlock(gridding_shape), [0], ['model']))
        def assert_low_frequency_brighter(models):
            """Make sure that the lower frequency model is much brighter"""
            low_frequency_model = np.abs(models[0])
            high_frequency_model = np.abs(models[1])
            self.assertGreater(np.sum(high_frequency_model), 0)
            self.assertGreater(
                np.sum(low_frequency_model)/np.sum(high_frequency_model),
                10e10)
        self.blocks.append((NumpyBlock(assert_low_frequency_brighter, outputs=0), {'in_1': 'model'}))
        Pipeline(self.blocks).main()
