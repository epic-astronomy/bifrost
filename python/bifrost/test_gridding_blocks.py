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

"""@package test_FakeVisBlock
This file tests the fake visibility block.
"""
import unittest
import numpy as np
from bifrost.block import FakeVisBlock, WriteAsciiBlock, NearestNeighborGriddingBlock, Pipeline
class TestFakeVisBlock(unittest.TestCase):
    """Performs tests of the fake visibility Block."""
    def setUp(self):
        self.datafile_name = "/data1/mcranmer/data/fake/mona_uvw.dat"
        self.blocks = []
        self.blocks.append(
            (FakeVisBlock(self.datafile_name), [], [0]))
        self.blocks.append((WriteAsciiBlock('.log.txt'), [0], []))
    def test_output_size(self):
        """Make sure the outputs are being sized appropriate to the file"""
        Pipeline(self.blocks).main()
        # Number of uvw values:
        length_ring_buffer = len(open('.log.txt', 'r').read().split(' '))
        length_data_file = sum(1 for line in open(self.datafile_name, 'r'))
        self.assertAlmostEqual(length_ring_buffer, 4*length_data_file, -2)
    def test_valid_output(self):
        """Make sure that the numbers in the ring match the uvw data"""
        Pipeline(self.blocks).main()
        ring_buffer_10th_u_coord = open('.log.txt', 'r').read().split(' ')[9*4]
        line_count = 0
        for line in open(self.datafile_name, 'r'):
            line_count += 1
            if line_count == 10:
                data_file_10th_line = line
                break
        data_file_10th_u_coord = data_file_10th_line.split(' ')[3]
        self.assertAlmostEqual(
            float(ring_buffer_10th_u_coord),
            float(data_file_10th_u_coord),
            3)
    def test_different_size_data(self):
        """Assert that different data sizes are processed properly"""
        datafile_name = "/data1/mcranmer/data/fake/mona_uvw_half.dat"
        self.blocks[0] = (FakeVisBlock(datafile_name), [], [0])
        Pipeline(self.blocks).main()
        length_ring_buffer = len(open('.log.txt', 'r').read().split(' '))
        length_data_file = sum(1 for line in open(datafile_name, 'r'))
        self.assertAlmostEqual(length_ring_buffer, 4*length_data_file, -2)
class TestNearestNeighborGriddingBlock(unittest.TestCase):
    """Test the functionality of the nearest neighbor gridding block"""
    def setUp(self):
        """Run a pipeline on a fake visibility set and grid it"""
        self.datafile_name = "/data1/mcranmer/data/fake/mona_uvw.dat"
        self.blocks = []
        self.blocks.append((FakeVisBlock(self.datafile_name), [], [0]))
        self.blocks.append((NearestNeighborGriddingBlock(shape=(100, 100)), [0], [1]))
        self.blocks.append((WriteAsciiBlock('.log.txt'), [1], []))
    def test_output_size(self):
        """Make sure that 10,000 grid points are created"""
        Pipeline(self.blocks).main()
        grid = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
        self.assertEqual(grid.size, 10000)
    def test_same_magnitude(self):
        """Make sure that many blocks are nonzero"""
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
