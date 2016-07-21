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
import bifrost
from block import *
class TestFakeVisBlock(unittest.TestCase):
    """Performs tests of the fake visibility Block."""
    def test_output_size(self):
        """Make sure the outputs are being sized appropriate to the file"""
        blocks = []
        blocks.append((FakeVisBlock("/data1/mcranmer/data/fake/mona_uvw.dat"), [], [0]))
        blocks.append((WriteAsciiBlock('.log.txt'), [0], []))
        Pipeline(blocks).main()
        length_ring_buffer = len(open('.log.txt', 'r').read().split(' '))
        length_data_file = len(open('/data1/mcranmer/data/fake/mona_uvw.dat', 'r').read().split('\n'))
        self.assertAlmostEqual(length_ring_buffer, 4*length_data_file, -2)
    def test_valid_output(self):
        """Make sure that the numbers in the ring match the uvw data"""
        blocks = []
        blocks.append((FakeVisBlock("/data1/mcranmer/data/fake/mona_uvw.dat"), [], [0]))
        blocks.append((WriteAsciiBlock('.log.txt'), [0], []))
        Pipeline(blocks).main()
        ring_buffer_10th_u_coord = open('.log.txt', 'r').read().split(' ')[9*4]
        data_file_10th_line = open('/data1/mcranmer/data/fake/mona_uvw.dat', 'r').read().split('\n')[9]
        data_file_10th_u_coord = data_file_10th_line.split(' ')[3]
        self.assertAlmostEqual(
            float(ring_buffer_10th_u_coord), 
            float(data_file_10th_u_coord),
            3)
    def test_different_size_data(self):
        """Assert that different data sizes are processed properly"""
        blocks = []
        blocks.append((FakeVisBlock("/data1/mcranmer/data/fake/mona_uvw_half.dat"), [], [0]))
        blocks.append((WriteAsciiBlock('.log.txt'), [0], []))
        Pipeline(blocks).main()
        length_ring_buffer = len(open('.log.txt', 'r').read().split(' '))
        length_data_file = sum(1 for line in open('/data1/mcranmer/data/fake/mona_uvw_half.dat', 'r'))
        self.assertAlmostEqual(length_ring_buffer, 4*length_data_file, -2)
