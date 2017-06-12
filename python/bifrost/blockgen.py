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
""" 
blockgen.py

This file contains some macros for quick generation of blocks """

import numpy as np 
import time
import bifrost as bf
import bifrost.pipeline as bfp
import bifrost.blocks as blocks

class BlockgenSource(bfp.SourceBlock):
    def __init__(self, generator):
        """ Load the generator into this object

        Args:

            generator: A generator object (a function with yeild statments)
        """
        super(BlockgenSource, self).__init__(["test"], gulp_nframe=1)
        self.generator = generator
        self.first_sample_of_generator = None
        self.on_first_sample = True
    def create_reader(self, sourcename):
        """ Start the generator, use it as the reader for SourceBlock """
        return self.generator()
    def on_sequence(self, reader, sourcename):
        """ Create output settings for the sequence based on the generator """
        self.first_sample_of_generator = next(reader)
        ohdr = {
            'name': 'test',
        }
        ohdr['_tensor'] = {
            'shape': [-1] + list(self.first_sample_of_generator.shape),
            'dtype': bf.DataType(self.first_sample_of_generator.dtype),
            'labels': [
                str(i+1) for i in range(len(self.first_sample_of_generator))],
            'scales': [None, None],
            'units': [None, None]
        }
        ohdr['time_tag'] = 0
        return [ohdr]
    def on_data(self, reader, ospans):
        """ Output the next element of the generator """
        ospan = ospans[0]
        odata = ospan.data
        try:
            if self.on_first_sample:
                odata[0, :] = self.first_sample_of_generator[:]
                self.on_first_sample = False
            else:
                odata[0, :] = next(reader)[:]
            return [1]
        except StopIteration:
            return [0]

def source(generator):
    """ Generate a source block based on a generator object
    
    Args:
        
        generator: A generator object (a function with yeild statments)

    Returns:

        BlockgenSource: A block object which uses the generator to
                        output data.
    """
    return BlockgenSource(generator)
