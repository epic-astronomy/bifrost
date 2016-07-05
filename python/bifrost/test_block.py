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
from block import *
from bifrost.ring import Ring

class TestCopyBlock(unittest.TestCase):
    """Performs tests of the Copy Block."""
    def setUp(self):
        """Set up the blocks list, and put in a single
            block which reads in the data from a filterbank 
            file."""
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                ['/data1/mcranmer/data/fake/1chan8bitNoDM.fil']),
            [], [0]))
    def test_simple_ring_copy(self):
        """Test which performs a read of a sigproc file,
            copy to one ring, and then output as text."""
        logfile = 'log.txt'
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((WriteAsciiBlock(logfile), [1], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(1)
        print open(logfile, 'r').read()
        self.assertEqual(test_byte, '2')
    def test_multi_linear_ring_copy(self):
        """Test which performs a read of a sigproc file,
            copy between many rings, and then output as
            text."""
        logfile = 'log2.txt'
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((CopyBlock(), [1], [2]))
        self.blocks.append((CopyBlock(), [2], [3]))
        self.blocks.append((CopyBlock(), [3], [4]))
        self.blocks.append((WriteAsciiBlock(logfile), [4], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')
    def test_nonlinear_ring_copy(self):
        """Test which reads in a sigproc file, and
            loads it between different rings in a 
            nonlinear fashion, then outputs to file."""
        logfile = 'log3.txt'
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((CopyBlock(), [0], [2]))
        self.blocks.append((CopyBlock(), [2], [5]))
        self.blocks.append((CopyBlock(), [0], [3]))
        self.blocks.append((CopyBlock(), [3], [4]))
        self.blocks.append((CopyBlock(), [5], [6]))
        self.blocks.append((WriteAsciiBlock(logfile), [6], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(17)
        self.assertEqual(test_byte[-1:], '3')

if __name__ == "__main__":
    unittest.main()
