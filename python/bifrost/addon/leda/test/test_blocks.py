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
import os
import numpy as np
from bifrost.block import WriteAsciiBlock, Pipeline
from bifrost.addon.leda.blocks import DadaReadBlock, NewDadaReadBlock

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
        self.logfile = '.log.txt'
        dadafile = '/data1/hg/dada_plot/2016-02-03-22_37_50_0001287429875776.dada'
        self.blocks = []
        self.blocks.append((NewDadaReadBlock(dadafile, output_chans=[42]), [], [0]))
        self.blocks.append((WriteAsciiBlock(self.logfile), [0], []))
        Pipeline(self.blocks).main() 
    def test_read_and_write(self):
        """Make sure some data is being written"""
        dumpsize = os.path.getsize(self.logfile)
        self.assertGreater(dumpsize, 100)

