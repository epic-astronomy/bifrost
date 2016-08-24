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

"""@package test_block
This file tests all aspects of the Bifrost.block module.
"""
import unittest
import numpy as np
from bifrost.fft import fft as bf_fft
from bifrost.ring import Ring
from bifrost.libbifrost import _bf
from bifrost.GPUArray import GPUArray
from bifrost.block import TestingBlock, WriteAsciiBlock, WriteHeaderBlock
from bifrost.block import SigprocReadBlock, CopyBlock, KurtosisBlock, FoldBlock
from bifrost.block import IFFTBlock, FFTBlock, Pipeline, FakeVisBlock
from bifrost.block import NearestNeighborGriddingBlock, IFFT2Block
from bifrost.block import GainSolveBlock, SplitterBlock, MultiAddBlock
from bifrost.block import SplitterBlock, DStackBlock, NumpyBlock
from bifrost.block import ReductionBlock, GPUBlock

class TestIterateRingWrite(unittest.TestCase):
    """Test the iterate_ring_write function of SourceBlocks/TransformBlocks"""
    def test_throughput(self):
        """Read in data with a small throughput size. Expect all to go through."""
        blocks = []
        blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/1chan8bitNoDM.fil', gulp_nframe=4096),
            [], [0]))
        blocks.append((WriteAsciiBlock('.log.txt'), [0], []))
        Pipeline(blocks).main()
        log_data = np.loadtxt('.log.txt')
        self.assertEqual(log_data.size, 12800)
class TestTestingBlock(unittest.TestCase):
    """Test the TestingBlock for basic functionality"""
    def setUp(self):
        """Initiate blocks list with write asciiBlock"""
        self.blocks = []
        self.blocks.append((WriteAsciiBlock('.log.txt', gulp_size=3*4), [0], []))
    def test_simple_dump(self):
        """Input some numbers, and ensure they are written to a file"""
        self.blocks.append((TestingBlock([1, 2, 3]), [], [0]))
        Pipeline(self.blocks).main()
        dumped_numbers = np.loadtxt('.log.txt')
        np.testing.assert_almost_equal(dumped_numbers, [1, 2, 3])
    def test_multi_dimensional_input(self):
        """Input a 2 dimensional list, and have this printed"""
        test_array = [[1, 2], [3, 4]]
        self.blocks[0] = (WriteAsciiBlock('.log.txt', gulp_size=4*4), [0], [])
        self.blocks.append((TestingBlock(test_array), [], [0]))
        self.blocks.append((WriteHeaderBlock('.log2.txt'), [0], []))
        Pipeline(self.blocks).main()
        header = eval(open('.log2.txt').read()) #pylint:disable=eval-used
        dumped_numbers = np.loadtxt('.log.txt').reshape(header['shape'])
        np.testing.assert_almost_equal(dumped_numbers, test_array)
class TestCopyBlock(unittest.TestCase):
    """Performs tests of the Copy Block."""
    def setUp(self):
        """Set up the blocks list, and put in a single
            block which reads in the data from a filterbank
            file."""
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/1chan8bitNoDM.fil'),
            [], [0]))
    def test_simple_copy(self):
        """Test which performs a read of a sigproc file,
            copy to one ring, and then output as text."""
        logfile = '.log.txt'
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((WriteAsciiBlock(logfile), [1], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')
    def test_multi_copy(self):
        """Test which performs a read of a sigproc file,
            copy between many rings, and then output as
            text."""
        logfile = '.log.txt'
        for i in range(10):
            self.blocks.append(
                (CopyBlock(), [i], [i+1]))
        self.blocks.append((WriteAsciiBlock(logfile), [10], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')
    def test_non_linear_multi_copy(self):
        """Test which reads in a sigproc file, and
            loads it between different rings in a
            nonlinear fashion, then outputs to file."""
        logfile = '.log.txt'
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((CopyBlock(), [0], [2]))
        self.blocks.append((CopyBlock(), [2], [5]))
        self.blocks.append((CopyBlock(), [0], [3]))
        self.blocks.append((CopyBlock(), [3], [4]))
        self.blocks.append((CopyBlock(), [5], [6]))
        self.blocks.append((WriteAsciiBlock(logfile), [6], []))
        Pipeline(self.blocks).main()
        log_nums = open(logfile, 'r').read(500).split(' ')
        test_num = np.float(log_nums[8])
        self.assertEqual(test_num, 3)
    def test_single_block_multi_copy(self):
        """Test which forces one block to do multiple
            copies at once, and then dumps to two files,
            checking them both."""
        logfiles = ['.log1.txt', '.log2.txt']
        self.blocks.append((CopyBlock(), [0], [1, 2]))
        self.blocks.append((WriteAsciiBlock(logfiles[0]), [1], []))
        self.blocks.append((WriteAsciiBlock(logfiles[1]), [2], []))
        Pipeline(self.blocks).main()
        test_bytes = int(
            open(logfiles[0], 'r').read(1)) + int(
                open(logfiles[1], 'r').read(1))
        self.assertEqual(test_bytes, 4)
    def test_32bit_copy(self):
        """Perform a simple test to confirm that 32 bit
            copying has no information loss"""
        logfile = '.log.txt'
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/256chan32bitNoDM.fil'),
            [], [0]))
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((WriteAsciiBlock(logfile), [1], []))
        Pipeline(self.blocks).main()
        test_bytes = open(logfile, 'r').read(500).split(' ')
        self.assertAlmostEqual(np.float(test_bytes[0]), 0.72650784254)
class TestFoldBlock(unittest.TestCase):
    """This tests functionality of the FoldBlock."""
    def setUp(self):
        """Set up the blocks list, and put in a single
            block which reads in the data from a filterbank
            file."""
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/pulsar_noisey_NoDM.fil'),
            [], [0]))
    def dump_ring_and_read(self):
        """Dump block to ring, read in as histogram"""
        logfile = ".log.txt"
        self.blocks.append((WriteAsciiBlock(logfile), [1], []))
        Pipeline(self.blocks).main()
        test_bytes = open(logfile, 'r').read().split(' ')
        histogram = np.array([np.float(x) for x in test_bytes])
        return histogram
    def test_simple_pulsar(self):
        """Test whether a pulsar histogram
            shows a large peak and is mostly
            nonzero values"""
        self.blocks.append((
            FoldBlock(bins=100), [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertEqual(histogram.size, 100)
        self.assertTrue(np.min(histogram) > 1e-10)
    def test_different_bin_size(self):
        """Try a different bin size"""
        self.blocks.append((
            FoldBlock(bins=50), [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertEqual(histogram.size, 50)
    def test_show_pulse(self):
        """Test to see if a pulse is visible in the
            histogram from pulsar data"""
        self.blocks[0] = (
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/simple_pulsar_DM0.fil'),
            [], [0])
        self.blocks.append((
            FoldBlock(bins=200), [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertTrue(np.min(histogram) > 1e-10)
        self.assertGreater(
            np.max(histogram)/np.average(histogram), 5)
    def test_many_channels(self):
        """See if many channels work with folding"""
        self.blocks[0] = (
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/simple_pulsar_DM0_128ch.fil'),
            [], [0])
        self.blocks.append((
            FoldBlock(bins=200), [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertTrue(np.min(histogram) > 1e-10)
        self.assertGreater(
            np.max(histogram)/np.min(histogram), 3)
    def test_high_dispersion(self):
        """Test folding on a file with high DM"""
        self.blocks[0] = (
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/simple_pulsar_DM10_128ch.fil'),
            [], [0])
        self.blocks.append((
            FoldBlock(bins=200, dispersion_measure=10, core=0),
            [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertTrue(np.min(histogram) > 1e-10)
        self.assertGreater(
            np.max(histogram)/np.min(histogram), 3)
        #TODO: Test to break bfmemcpy2D for lack of float32 functionality?
class TestKurtosisBlock(unittest.TestCase):
    """This tests functionality of the KurtosisBlock."""
    def test_data_throughput(self):
        """Check that data is being put through the block
        (does this by checking consistency of shape/datatype)"""
        blocks = []
        blocks.append((
            SigprocReadBlock('/data1/mcranmer/data/fake/1chan8bitNoDM.fil'),
            [], [0]))
        blocks.append((
            KurtosisBlock(), [0], [1]))
        blocks.append((
            WriteAsciiBlock('.log.txt'), [1], []))
        Pipeline(blocks).main()
        test_byte = open('.log.txt', 'r').read().split(' ')
        test_nums = np.array([float(x) for x in test_byte])
        self.assertLess(np.max(test_nums), 256)
        self.assertEqual(test_nums.size, 12800)
class TestFFTBlock(unittest.TestCase):
    """This test assures basic functionality of fft block"""
    def setUp(self):
        """Assemble a basic pipeline with the FFT block"""
        self.logfile = '.log.txt'
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/1chan8bitNoDM.fil'),
            [], [0]))
        self.blocks.append((FFTBlock(gulp_size=4096*8*8*8), [0], [1]))
        self.blocks.append((WriteAsciiBlock(self.logfile), [1], []))
    def test_throughput(self):
        """Test that any data is being put through"""
        Pipeline(self.blocks).main()
        test_string = open(self.logfile, 'r').read()
        self.assertGreater(len(test_string), 0)
    def test_throughput_size(self):
        """Number of elements going out should be double that of basic copy"""
        Pipeline(self.blocks).main()
        number_fftd = len(open(self.logfile, 'r').read().split('\n'))
        number_fftd = np.loadtxt(self.logfile).size
        open(self.logfile, 'w').close()
        ## Run pipeline again with simple copy
        self.blocks[1] = (CopyBlock(), [0], [1])
        Pipeline(self.blocks).main()
        number_copied = np.loadtxt(self.logfile).size
        self.assertAlmostEqual(number_fftd, 2*number_copied)
    def test_data_sizes(self):
        """Test that different number of bits give correct throughput size"""
        for iterate in range(5):
            nbit = 2**iterate
            if nbit == 8:
                continue
            self.blocks[0] = (
                SigprocReadBlock(
                    '/data1/mcranmer/data/fake/2chan'+ str(nbit) + 'bitNoDM.fil'),
                [], [0])
            open(self.logfile, 'w').close()
            Pipeline(self.blocks).main()
            number_fftd = np.loadtxt(self.logfile).astype(np.float32).view(np.complex64).size
            # Compare with simple copy
            self.blocks[1] = (CopyBlock(), [0], [1])
            open(self.logfile, 'w').close()
            Pipeline(self.blocks).main()
            number_copied = np.loadtxt(self.logfile).size
            self.assertEqual(number_fftd, number_copied)
            # Go back to FFT
            self.blocks[1] = (FFTBlock(gulp_size=4096*8*8*8), [0], [1])
    def test_fft_result(self):
        """Make sure that fft matches what it should!"""
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        fft_block_result = np.loadtxt(self.logfile).astype(np.float32).view(np.complex64)
        self.blocks[1] = (CopyBlock(), [0], [1])
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        normal_fft_result = np.fft.fft(np.loadtxt(self.logfile))
        np.testing.assert_almost_equal(fft_block_result, normal_fft_result, 2)
class TestIFFTBlock(unittest.TestCase):
    """This test assures basic functionality of the ifft block.
    Requires the FFT block for testing."""
    def test_simple_ifft(self):
        """Put test data through a ring buffer and check correctness"""
        self.logfile = '.log.txt'
        self.blocks = []
        test_array = [1, 2, 3]
        self.blocks.append((TestingBlock(test_array), [], [0]))
        self.blocks.append((IFFTBlock(gulp_size=3*4), [0], [1]))
        self.blocks.append((WriteAsciiBlock(self.logfile), [1], []))
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        true_result = np.fft.ifft(test_array)
        result = np.loadtxt(self.logfile).astype(np.float32).view(np.complex64)
        np.testing.assert_almost_equal(result, true_result, 2)
    def test_equivalent_data_to_copy(self):
        """Test that the data coming out of this pipeline is equivalent
        the initial read data"""
        self.logfile = '.log.txt'
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/1chan8bitNoDM.fil'),
            [], [0]))
        self.blocks.append((FFTBlock(gulp_size=4096*8*8*8*8), [0], [1]))
        self.blocks.append((IFFTBlock(gulp_size=4096*8*8*8*8), [1], [2]))
        self.blocks.append((WriteAsciiBlock(self.logfile), [2], []))
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        unfft_result = np.loadtxt(self.logfile).astype(np.float32).view(np.complex64)
        self.blocks[1] = (CopyBlock(), [0], [1])
        self.blocks[2] = (WriteAsciiBlock(self.logfile), [1], [])
        del self.blocks[3]
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        untouched_result = np.loadtxt(self.logfile).astype(np.float32)
        np.testing.assert_almost_equal(unfft_result, untouched_result, 2)

class TestFakeVisBlock(unittest.TestCase):
    """Performs tests of the fake visibility Block."""
    def setUp(self):
        self.datafile_name = "/data1/mcranmer/data/fake/mona_uvw.dat"
        self.blocks = []
        self.num_stands = 512
        self.blocks.append(
            (FakeVisBlock(self.datafile_name, self.num_stands), [], [0]))
    def tearDown(self):
        """Run the pipeline (which should have the asserts inside it)"""
        Pipeline(self.blocks).main()
    def test_output_size(self):
        """Make sure the outputs are being sized appropriate to the file"""
        def verify_ring_size(array):
            """Check ring output size"""
            self.assertAlmostEqual(
                array.size,
                6*self.num_stands*(self.num_stands+1)//2,
                -2)
        self.blocks.append([
            NumpyBlock(function=verify_ring_size, outputs=0),
            {'in_1': 0}])
    def test_different_size_data(self):
        """Assert that different data sizes are processed properly"""
        self.num_stands = 256
        def verify_ring_size(array):
            """Check ring output size"""
            self.assertAlmostEqual(
                array.size,
                6*self.num_stands*(self.num_stands+1)//2,
                -2)
        datafile_name = "/data1/mcranmer/data/fake/mona_uvw_half.dat"
        self.blocks[0] = (FakeVisBlock(datafile_name, self.num_stands), [], [0])
        self.blocks.append([
            NumpyBlock(function=verify_ring_size, outputs=0),
            {'in_1': 0}])
class TestNearestNeighborGriddingBlock(unittest.TestCase):
    """Test the functionality of the nearest neighbor gridding block"""
    def setUp(self):
        """Run a pipeline on a fake visibility set and grid it"""
        self.datafile_name = "/data1/mcranmer/data/fake/mona_uvw.dat"
        self.blocks = []
        self.blocks.append((FakeVisBlock(self.datafile_name, 512), [], [0]))
        self.blocks.append((NearestNeighborGriddingBlock(shape=(100, 100)), [0], [1]))
        self.blocks.append((WriteAsciiBlock('.log.txt'), [1], []))
    def test_output_size(self):
        """Make sure that 10,000 grid points are created"""
        Pipeline(self.blocks).main()
        grid = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        self.assertEqual(grid.size, 10000)
    def test_same_magnitude(self):
        Pipeline(self.blocks).main()
        grid = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        magnitudes = np.abs(grid)
        self.assertGreater(magnitudes[magnitudes > 0.1].size, 100)
    def test_makes_image(self):
        """Make sure that the grid can be IFFT'd into a non-gaussian image"""
        self.blocks[1] = (NearestNeighborGriddingBlock(shape=(512, 512)), [0], [1])
        Pipeline(self.blocks).main()
        grid = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        grid = grid.reshape((512, 512))
        image = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))))
        #calculate histogram of image
        histogram = np.histogram(image.ravel(), bins=100)[0]
        #check if it is gaussian (and therefore probably just noise)
        from scipy.stats import normaltest
        probability_normal = normaltest(histogram)[1]
        self.assertLess(probability_normal, 1e-2)
class TestIFFT2Block(unittest.TestCase):
    """Test the functionality of the 2D inverse fourier transform block"""
    def setUp(self):
        """Run a pipeline on a fake visibility set and IFFT it after gridding"""
        self.datafile_name = "/data1/mcranmer/data/fake/mona_uvw.dat"
        self.blocks = []
        self.blocks.append((FakeVisBlock(self.datafile_name, 512), [], [0]))
        self.blocks.append((NearestNeighborGriddingBlock(shape=(100, 100)), [0], [1]))
        self.blocks.append((IFFT2Block(), [1], [2]))
        self.blocks.append((WriteAsciiBlock('.log.txt'), [2], []))
    def test_output_size(self):
        """Make sure that the output is the same size as the input
        The input size should be coming from the shape on the nearest neighbor"""
        open('.log.txt', 'w').close()
        Pipeline(self.blocks).main()
        brightness = np.real(np.loadtxt('.log.txt').astype(np.float32).view(np.complex64))
        self.assertEqual(brightness.size, 10000)
    def test_same_magnitude(self):
        """Make sure that many points are nonzero"""
        open('.log.txt', 'w').close()
        Pipeline(self.blocks).main()
        brightness = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        magnitudes = np.abs(brightness)
        self.assertGreater(magnitudes[magnitudes > 0.1].size, 100)
    def test_ifft_correct_values(self):
        """Make sure the IFFT produces values as if we were to do it without the block"""
        open('.log.txt', 'w').close()
        Pipeline(self.blocks).main()
        test_brightness = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        test_brightness = test_brightness.reshape((100, 100))
        self.blocks[2] = (WriteAsciiBlock('.log.txt'), [1], [])
        del self.blocks[3]
        open('.log.txt', 'w').close()
        Pipeline(self.blocks).main()
        grid = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        grid = grid.reshape((100, 100))
        brightness = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid)))
        from matplotlib import pyplot as plt
        plt.imshow(np.real(test_brightness)) #Needs to be in row,col order
        plt.savefig('mona.png')
        np.testing.assert_almost_equal(test_brightness, brightness, 2)
class TestPipeline(unittest.TestCase):
    """Test rigidity and features of the pipeline"""
    def test_naming_rings(self):
        """Name the rings instead of numerating them"""
        blocks = []
        blocks.append((TestingBlock([1, 2, 3]), [], ['ring1']))
        blocks.append((WriteAsciiBlock('.log.txt', gulp_size=3*4), ['ring1'], []))
        open('.log.txt', 'w').close()
        Pipeline(blocks).main()
        result = np.loadtxt('.log.txt').astype(np.float32)
        np.testing.assert_almost_equal(result, [1, 2, 3])
    def test_pass_rings(self):
        """Pass rings entirely instead of naming/numerating them"""
        block_set_one = []
        block_set_two = []
        ring1 = Ring()
        block_set_one.append((TestingBlock([1, 2, 3]), [], [ring1]))
        block_set_two.append((WriteAsciiBlock('.log.txt', gulp_size=3*4), [ring1], []))
        open('.log.txt', 'w').close()
        Pipeline(block_set_one).main() # The ring should communicate between the pipelines
        Pipeline(block_set_two).main()
        result = np.loadtxt('.log.txt').astype(np.float32)
        np.testing.assert_almost_equal(result, [1, 2, 3])
class TestGainSolveBlock(unittest.TestCase):
    """Test the gain solve block, which calls mitchcal gain solve"""
    def setUp(self):
        self.nchan = 1
        self.nstand = 256
        self.npol = 2
        self.jones = np.ones(shape=[
            1, self.npol,
            self.nstand, self.npol]).astype(np.complex64)
    def test_throughput(self):
        """Test shapes are compatible and output is indeed different"""
        def test_jones(out_jones):
            """Make sure the jones matrices are changing"""
            self.assertEqual(out_jones.size, self.jones.size)
            self.assertGreater(np.max(np.abs(out_jones - self.jones)), 1e-3)
        for nchan in range(5, 7):
            blocks = []
            flags = 2*np.ones(shape=[
                nchan, self.nstand]).astype(np.int8)
            model = 10*np.random.rand(
                nchan, self.nstand,
                self.npol, self.nstand,
                self.npol).astype(np.complex64)
            data = 10*np.random.rand(
                nchan, self.nstand,
                self.npol, self.nstand,
                self.npol).astype(np.complex64)
            self.jones = np.ones(shape=[
                nchan, self.npol,
                self.nstand, self.npol]).astype(np.complex64)
            blocks.append((TestingBlock(model), [], ['model']))
            blocks.append((TestingBlock(data), [], ['data']))
            blocks.append((TestingBlock(self.jones), [], ['jones_in']))
            blocks.append([GainSolveBlock(flags=flags), {
                'in_data': 'data', 'in_model': 'model', 'in_jones': 'jones_in',
                'out_data': 'calibrated_data', 'out_jones': 'jones_out'}])
            blocks.append([NumpyBlock(test_jones, outputs=0), {'in_1':'jones_out'}])
            Pipeline(blocks).main()
    def test_solving_to_skymodel(self):
        """Attempt to solve a sky model to itself"""
        #TODO: This relies on LEDA-specific blocks.
        from bifrost.addon.leda.model_block import ScalarSkyModelBlock
        from bifrost.addon.leda.blocks import load_telescope, LEDA_SETTINGS_FILE, OVRO_EPHEM
        coords = load_telescope(LEDA_SETTINGS_FILE)[1]
        sources = {}
        sources['cyg'] = {
            'ra':'19:59:28.4', 'dec':'+40:44:02.1', 'flux': 10571.0, 'frequency': 58e6,
            'spectral index': -0.2046}
        frequencies = [58e6]
        blocks = []
        blocks.append((
            ScalarSkyModelBlock(OVRO_EPHEM, coords, frequencies, sources), [], ['model+uv']))
        def slice_away_uv(model_and_uv):
            """Cut off the uv coordinates from the ScalarSkyModelBlock and reshape to GainSolve"""
            number_stands = model_and_uv.shape[0]
            model = np.zeros(shape=[1, number_stands, 2, number_stands, 2]).astype(np.complex64)
            model[0, :, 0, :, 0] = model_and_uv[:, :, 2]+1j*model_and_uv[:, :, 3]
            model[0, :, 1, :, 1] = model[0, :, 0, :, 0]
            return model
        blocks.append((NumpyBlock(slice_away_uv), {'in_1': 'model+uv', 'out_1': 'model'}))
        blocks.append((NumpyBlock(np.copy), {'in_1': 'model', 'out_1': 'same_model'}))
        flags = 2*np.ones(shape=[
            1, self.nstand]).astype(np.int8)
        blocks.append((TestingBlock(2*self.jones), [], ['jones_in']))
        blocks.append([GainSolveBlock(flags=flags, eps=0.05), {
            'in_data': 'same_model', 'in_model': 'model', 'in_jones': 'jones_in',
            'out_data': 'calibrated_data', 'out_jones': 'jones_out'}])
        def assert_almost_unity(jones_matrices):
            """Make sure that the jones have been calibrated to be identity"""
            identity_jones = np.ones(shape=[
                1, self.npol,
                self.nstand, self.npol]).astype(np.complex64)
            identity_jones[:, 0, :, 1] = 0
            identity_jones[:, 1, :, 0] = 0
            np.testing.assert_almost_equal(jones_matrices, identity_jones, 1)
        blocks.append((NumpyBlock(assert_almost_unity, outputs=0), {'in_1': 'jones_out'}))
        Pipeline(blocks).main()
class TestMultiTransformBlock(unittest.TestCase):
    """Test call syntax and function of a multi transform block"""
    def test_add_block(self):
        """Try some syntax on an addition block."""
        my_ring = Ring()
        blocks = []
        blocks.append([TestingBlock([1, 2]), [], [0]])
        blocks.append([TestingBlock([1, 6]), [], [1]])
        blocks.append([TestingBlock([9, 2]), [], [2]])
        blocks.append([TestingBlock([6, 2]), [], [3]])
        blocks.append([TestingBlock([1, 2]), [], [4]])
        blocks.append([
            MultiAddBlock(),
            {'in_1': 0, 'in_2':1, 'out_sum': 'first_sum'}])
        blocks.append([
            MultiAddBlock(),
            {'in_1': 2, 'in_2':3, 'out_sum': 'second_sum'}])
        blocks.append([
            MultiAddBlock(),
            {'in_1': 'first_sum', 'in_2':'second_sum', 'out_sum': 'third_sum'}])
        blocks.append([
            MultiAddBlock(),
            {'in_1': 'third_sum', 'in_2':4, 'out_sum': my_ring}])
        blocks.append([WriteAsciiBlock('.log.txt'), [my_ring], []])
        Pipeline(blocks).main()
        summed_result = np.loadtxt('.log.txt')
        np.testing.assert_almost_equal(summed_result, [18, 14])
    def test_for_bad_ring_definitions(self):
        """Try to pass bad input and outputs"""
        blocks = []
        blocks.append([TestingBlock([1, 2]), [], [0]])
        blocks.append([
            MultiAddBlock(),
            {'in_2':0, 'out_sum': 1}])
        blocks.append([WriteAsciiBlock('.log.txt'), [1], []])
        with self.assertRaises(AssertionError):
            Pipeline(blocks).main()
        blocks[1] = [
            MultiAddBlock(),
            {'bad_ring_name':0, 'in_2':0, 'out_sum': 1}]
        with self.assertRaises(AssertionError):
            Pipeline(blocks).main()
class TestSplitterBlock(unittest.TestCase):
    """Test a block which splits up incoming data into two rings"""
    def test_simple_half_split(self):
        """Try to split up a single array in half, and dump to file"""
        blocks = []
        blocks.append([TestingBlock([1, 2]), [], [0]])
        blocks.append([SplitterBlock([[0], [1]]), {'in': 0, 'out_1':1, 'out_2':2}])
        blocks.append([WriteAsciiBlock('.log1.txt', gulp_size=4), [1], []])
        blocks.append([WriteAsciiBlock('.log2.txt', gulp_size=4), [2], []])
        Pipeline(blocks).main()
        first_log = np.loadtxt('.log1.txt')
        second_log = np.loadtxt('.log2.txt')
        self.assertEqual(first_log.size, 1)
        self.assertEqual(second_log.size, 1)
        np.testing.assert_almost_equal(first_log+1, second_log)
class TestDStackBlock(unittest.TestCase):
    """Test a block which stacks two incoming streams into one outgoing ring"""
    def test_simple_throughput(self):
        """Send two arrays through, and make sure they come out as one"""
        blocks = []
        blocks.append([TestingBlock([1]), [], [0]])
        blocks.append([TestingBlock([2]), [], [1]])
        blocks.append([
            DStackBlock(),
            {'in_1': 0, 'in_2': 1, 'out': 2}])
        blocks.append([WriteAsciiBlock('.log.txt', gulp_size=8), [2], []])
        Pipeline(blocks).main()
        log_data = np.loadtxt('.log.txt')
        np.testing.assert_almost_equal(log_data, [1, 2])
    def test_multi_array_throughput(self):
        """Send two 2D arrays through, and make sure they come out properly"""
        array_1 = np.arange(10).reshape((2, 5))
        array_2 = 3*np.arange(10).reshape((2, 5))
        desired_output = np.dstack((
            array_1,
            array_2)).ravel()
        blocks = []
        blocks.append([TestingBlock(array_1), [], [0]])
        blocks.append([TestingBlock(array_2), [], [1]])
        blocks.append([
            DStackBlock(),
            {'in_1': 0, 'in_2': 1, 'out': 2}])
        blocks.append([WriteAsciiBlock('.log.txt', gulp_size=80), [2], []])
        Pipeline(blocks).main()
        log_data = np.loadtxt('.log.txt')
        np.testing.assert_almost_equal(log_data, desired_output)
class TestReductionBlock(unittest.TestCase):
    """Test a block which applies a function passed to it to the incoming array"""
    def test_unity_function(self):
        """Apply a function which returns its arguments"""
        def identity(argument):
            return argument
        blocks = []
        blocks.append([TestingBlock([1]), [], [0]])
        blocks.append([
            ReductionBlock(identity),
            {'in': 0, 'out':1}])
        blocks.append([WriteAsciiBlock('.log.txt', gulp_size=4), [1], []])
        Pipeline(blocks).main()
        log_data = np.loadtxt('.log.txt')
        np.testing.assert_almost_equal(log_data, [1])
    def test_complex_to_real(self):
        """Apply a function which outputs only the real components"""
        output_header = {'dtype': str(np.float32), 'nbit': 32}
        blocks = []
        blocks.append([TestingBlock([2j+5]), [], [0]])
        blocks.append([
            ReductionBlock(np.real, output_header=output_header),
            {'in': 0, 'out':1}])
        blocks.append([WriteAsciiBlock('.log.txt', gulp_size=4), [1], []])
        Pipeline(blocks).main()
        log_data = np.loadtxt('.log.txt')
        np.testing.assert_almost_equal(log_data, [5])
    def test_multiply_by_two(self):
        """Apply a function which multiplies by two"""
        def multiply_by_two(argument):
            return 2*argument
        blocks = []
        blocks.append([TestingBlock([1]), [], [0]])
        blocks.append([
            ReductionBlock(multiply_by_two),
            {'in': 0, 'out':1}])
        blocks.append([WriteAsciiBlock('.log.txt', gulp_size=4), [1], []])
        Pipeline(blocks).main()
        log_data = np.loadtxt('.log.txt')
        np.testing.assert_almost_equal(log_data, [2])
class TestNumpyBlock(unittest.TestCase):
    """Tests for a block which can call arbitrary functions that work on numpy arrays.
        This should include the many numpy, scipy and astropy functions.
        Furthermore, this block should automatically move GPU data to CPU,
        call the numpy function, and then put out data on a CPU ring.
        The purpose of this ring is mainly for tests or filling in missing
        functionality."""
    def setUp(self):
        """Set up a pipeline for a numpy operation in the middle"""
        self.blocks = []
        self.test_array = [1, 2, 3, 4]
        self.blocks.append((TestingBlock(self.test_array), [], [0]))
        self.blocks.append((WriteAsciiBlock('.log.txt'), [1], []))
        self.expected_result = []
    def tearDown(self):
        """Run the pipeline and test the output against the expectation"""
        Pipeline(self.blocks).main()
        if np.array(self.expected_result).dtype == 'complex128':
            result = np.loadtxt('.log.txt', dtype=np.float64).view(np.complex128)
        else:
            result = np.loadtxt('.log.txt').astype(np.float32)
        np.testing.assert_almost_equal(result, self.expected_result)
    def test_simple_copy(self):
        """Perform a np.copy on a ring"""
        self.blocks.append([
            NumpyBlock(function=np.copy),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = [1, 2, 3, 4]
    def test_boolean_output(self):
        """Convert a ring into boolean output"""
        def greater_than_two(array):
            """Return a matrix representing whether each element
                is greater than 2"""
            return array > 2
        self.blocks.append([
            NumpyBlock(function=greater_than_two),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = [0, 0, 1, 1]
    def test_different_size_output(self):
        """Test that the output size can be different"""
        def first_half(array):
            """Only return the first half of the input vector"""
            array = np.array(array)
            return array[:int(array.size/2)]
        self.blocks.append([
            NumpyBlock(function=first_half),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = first_half(self.test_array)
    def test_complex_output(self):
        """Test that complex data can be generated"""
        self.blocks.append([
            NumpyBlock(function=np.fft.fft),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = np.fft.fft(self.test_array)
    def test_two_inputs(self):
        """Test that two input rings work"""
        def dstack_handler(array_1, array_2):
            """Stack two matrices along a third dimension"""
            return np.dstack((array_1, array_2))
        self.blocks.append([
            NumpyBlock(function=np.copy),
            {'in_1': 0, 'out_1': 2}])
        self.blocks.append([
            NumpyBlock(function=dstack_handler, inputs=2),
            {'in_1': 0, 'in_2': 2, 'out_1': 1}])
        self.expected_result = np.dstack((self.test_array, self.test_array)).ravel()
    def test_100_inputs(self):
        """Test that 100 input rings work"""
        def dstack_handler(*args):
            """Stack all input arrays"""
            return np.dstack(tuple(args))
        number_inputs = 100
        connections = {'in_1': 0, 'out_1': 1}
        for index in range(number_inputs):
            self.blocks.append([
                NumpyBlock(function=np.copy),
                {'in_1': 0, 'out_1': index+2}])
            connections['in_'+str(index+2)] = index+2
        self.blocks.append([
            NumpyBlock(function=dstack_handler, inputs=len(connections)-1),
            connections])
        self.expected_result = np.dstack((self.test_array,)*(len(connections)-1)).ravel()
    def test_two_outputs(self):
        """Test that two output rings work by copying input data to both"""
        def double(array):
            """Return two of the inputted matrix"""
            return (array, array)
        self.blocks.append([
            NumpyBlock(function=double, outputs=2),
            {'in_1': 0, 'out_1': 2, 'out_2': 1}])
        self.expected_result = [1, 2, 3, 4]
    def test_10_input_10_output(self):
        """Test that 10 input and 10 output rings work"""
        def dstack_handler(*args):
            """Stack all input arrays"""
            return np.dstack(tuple(args))
        def identity(*args):
            """Return all arrays passed"""
            return args
        number_rings = 10
        connections = {}
        for index in range(number_rings):
            #Simple 1 to 1 copy block
            self.blocks.append([
                NumpyBlock(function=np.copy),
                {'in_1': 0, 'out_1': index+2}])
            connections['in_'+str(index+1)] = index+2
            connections['out_'+str(index+1)] = index+2+number_rings
        #Copy all inputs to all outputs
        self.blocks.append([
            NumpyBlock(function=identity, inputs=number_rings, outputs=number_rings),
            dict(connections)])
        second_connections = {}
        for key in connections:
            if key[:3] == 'out':
                second_connections['in'+key[3:]] = int(connections[key])
        second_connections['out_1'] = 1
        #Stack N input rings into 1 output ring
        self.blocks.append([
            NumpyBlock(function=dstack_handler, inputs=number_rings, outputs=1),
            second_connections])
        self.expected_result = np.dstack((self.test_array,)*(len(second_connections)-1)).ravel()
    def test_zero_outputs(self):
        """Test zero outputs on NumpyBlock. Nothing should be sent through self.function at init"""
        def assert_something(array):
            """Assert the array is only 4 numbers, and return nothing"""
            np.testing.assert_almost_equal(array, [1, 2, 3, 4])
        self.blocks.append([
            NumpyBlock(function=assert_something, outputs=0),
            {'in_1': 0}])
        self.blocks.append([
            NumpyBlock(function=np.copy, outputs=1),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = [1, 2, 3, 4]
    def test_global_variable_capture(self):
        """Test that we can pull out a number from a ring using NumpyBlock"""
        self.global_variable = np.array([])
        def create_global_variable(array):
            """Try to append the array to a global variable"""
            self.global_variable = np.copy(array)
        self.blocks.append([
            NumpyBlock(function=create_global_variable, outputs=0),
            {'in_1': 0}])
        self.blocks.append([
            NumpyBlock(function=np.copy),
            {'in_1': 0, 'out_1': 1}])
        Pipeline(self.blocks).main()
        self.expected_result = self.global_variable
class TestGPUBlock(unittest.TestCase):
    """Tests for a block which can call arbitrary functions that work on GPUArrays.
        This block should automatically move CPU data to GPU,
        call the C function, and then put out data on a GPU ring."""
    def test_simple_throughput(self):
        """Apply an identity function to the GPU"""
        self.function_iterations = 0
        def identity(array):
            """Return the GPUArray"""
            if self.function_iterations > 0:
                self.assertTrue(isinstance(array, GPUArray))
                np.testing.assert_almost_equal(array.get(), np.ones(10))
            self.function_iterations += 1
            return array
        blocks = []
        blocks.append([TestingBlock(np.ones(10)), [], [0]])
        blocks.append([GPUBlock(identity), {'in_1':0, 'out_1':1}])
        Pipeline(blocks).main()
        self.assertGreater(self.function_iterations, 0)
    def test_use_pycuda(self):
        """Send a GPU ring through a pycuda function"""
        try:
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule
        except ImportError:
            print "No PyCUDA installation detected. Skipping tests..."
            return
        def double(gpu_array):
            """Double every value of a 4 element gpu vector"""
            device = cuda.Device(0)
            context = device.make_context()
            pycuda_array = gpu_array.as_pycuda(driver=cuda)
            double_kernel = SourceModule("""
              __global__ void double_the_array(float *a)
              {
                int idx = threadIdx.x;
                a[idx] *= 2;
              }
                """)
            function = double_kernel.get_function("double_the_array")
            function(pycuda_array, block=(4, 1, 1))
            gpu_array.set_from_pycuda(pycuda_array, driver=cuda)
            context.pop()
            del pycuda_array
            del context
            return gpu_array
        def assert_twos(array):
            """Test that everything got doubled"""
            np.testing.assert_almost_equal(array, 2*np.ones(4))
        blocks = []
        blocks.append([TestingBlock(np.ones(4)), [], [0]])
        blocks.append([GPUBlock(double), {'in_1':0, 'out_1':1}])
        blocks.append([NumpyBlock(assert_twos, outputs=0), {'in_1':1}])
        Pipeline(blocks).main()
    def test_use_pyclibrary(self):
        """Send a GPU ring through a PyCLibrary-loaded function"""
        try:
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule
        except ImportError:
            print "No PyCUDA installation detected. Skipping tests..."
            return
        def fft(input_array):
            """Perform an fft on the input"""
            bifrost_gpu_array = input_array.as_BFarray()
            bf_fft(bifrost_gpu_array, bifrost_gpu_array)
            input_array.buffer = bifrost_gpu_array.data
            return input_array
        def assert_ffted(array):
            """Test that everything got ffted"""
            print array
            np.testing.assert_almost_equal(
                array,
                np.fft.fft(np.ones(4)+1j*np.ones(4)))
        blocks = []
        blocks.append([TestingBlock(np.ones(4) + 1j*np.ones(4), complex_numbers=True), [], [0]])
        blocks.append([GPUBlock(fft), {'in_1':0, 'out_1':1}])
        blocks.append([NumpyBlock(assert_ffted, outputs=0), {'in_1':1}])
        Pipeline(blocks).main()
