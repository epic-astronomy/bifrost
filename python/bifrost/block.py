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

"""@package block This file defines a generic block class.

Right now the only possible block type is one
of a simple transform which works on a span by span basis.
"""
import json

import matplotlib
import numpy as np

## Use a graphical backend which supports threading
matplotlib.use('Agg')
import bifrost
from bifrost.sigproc import unpack


def insert_zeros_evenly(input_data, number_zeros):
    """Insert zeros evenly in input_data.
        These zeros are distibuted evenly throughout
        the function, to help for binning of oddly
        shaped arrays.
    @param[in] input_data 1D array to contain zeros.
    @param[out] number_zeros Number of zeros that need
        to be added.
    @returns input_data with extra zeros"""
    insert_index = np.floor(
        np.arange(
            number_zeros,
            step=1.0) * float(input_data.size) / number_zeros)
    output_data = np.insert(
        input_data, insert_index,
        np.zeros(number_zeros))
    return output_data


class TransformBlock(object):
    """Defines the structure for a transform block"""

    def __init__(self, gulp_size=4096):
        super(TransformBlock, self).__init__()
        self.gulp_size = gulp_size
        self.out_gulp_size = None
        self.input_header = {}
        self.output_header = {}
        self.core = -1

    def load_settings(self, input_header):
        """Load in input header and set up block attributes
        @param[in] input_header Header sent from input ring"""
        self.output_header = input_header

    def iterate_ring_read(self, input_ring):
        """Iterate through one input ring"""
        input_ring.resize(self.gulp_size)
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            for span in sequence.read(self.gulp_size):
                yield span

    def iterate_ring_write(self, output_ring, sequence_name="", sequence_time_tag=0, sequence_nringlet=1):
        """Iterate through one output ring"""
        if self.out_gulp_size is None:
            self.out_gulp_size = self.gulp_size
        output_ring.resize(self.out_gulp_size)
        with output_ring.begin_writing() as oring:
            with oring.begin_sequence(
                    sequence_name, sequence_time_tag,
                    header=self.output_header,
                    nringlet=sequence_nringlet) as oseq:
                with oseq.reserve(self.out_gulp_size) as span:
                    yield span

    def ring_transfer(self, input_ring, output_ring):
        """Iterate through two rings span-by-span"""
        input_ring.resize(self.gulp_size)
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            if self.out_gulp_size is None:
                self.out_gulp_size = self.gulp_size
            output_ring.resize(self.out_gulp_size)
            with output_ring.begin_writing() as oring:
                with oring.begin_sequence(
                        sequence.name, sequence.time_tag,
                        header=self.output_header,
                        nringlet=sequence.nringlet) as oseq:
                    for ispan in sequence.read(self.gulp_size):
                        with oseq.reserve(ispan.size * self.out_gulp_size / self.gulp_size) as ospan:
                            yield ispan, ospan


class SourceBlock(object):
    """Defines the structure for a source block"""

    def __init__(self, gulp_size=4096):
        super(SourceBlock, self).__init__()
        self.gulp_size = gulp_size
        self.output_header = {}
        self.core = -1

    def iterate_ring_write(self, output_ring, sequence_name="", sequence_time_tag=0):
        """Iterate over output ring
        @param[in] output_ring Ring to write to
        @param[in] sequence_name Name to label sequence
        @param[in] sequence_time_tag Time tag to label sequence
        """
        output_ring.resize(self.gulp_size)
        with output_ring.begin_writing() as oring:
            with oring.begin_sequence(sequence_name, sequence_time_tag, header=self.output_header, nringlet=1) as oseq:
                while True:
                    with oseq.reserve(self.gulp_size) as span:
                        yield span


class SinkBlock(object):
    """Defines the structure for a sink block"""

    def __init__(self, gulp_size=4096):
        super(SinkBlock, self).__init__()
        self.gulp_size = gulp_size
        self.header = {}
        self.core = -1

    def load_settings(self, input_header):
        """Load in settings from input ring header"""
        self.header = json.loads(input_header.tostring())

    def iterate_ring_read(self, input_ring):
        """Iterate through one input ring
        @param[in] input_ring Ring to read through"""
        input_ring.resize(self.gulp_size)
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            for span in sequence.read(self.gulp_size):
                yield span


class TestingBlock(SourceBlock):
    """Block for debugging purposes.
    Allows you to pass arbitrary N-dimensional arrays in initialization,
    which will be outputted into a ring buffer"""

    def __init__(self, test_array):
        """@param[in] test_array A list or numpy array containing test data"""
        super(TestingBlock, self).__init__()
        self.test_array = np.array(test_array).astype(np.float32)
        self.output_header = json.dumps(
            {'nbit': 32,
             'dtype': str(np.float32),
             'shape': self.test_array.shape})

    def main(self, output_ring):
        """Put the test array onto the output ring
        @param[in] output_ring Holds the flattend test array in a single span"""
        self.gulp_size = self.test_array.nbytes
        for ospan in self.iterate_ring_write(output_ring):
            ospan.data_view(np.float32)[0][:] = self.test_array.ravel()
            break


class WriteHeaderBlock(SinkBlock):
    """Prints the header of a ring to a file"""

    def __init__(self, filename):
        """@param[in] test_array A list or numpy array containing test data"""
        super(WriteHeaderBlock, self).__init__()
        self.filename = filename

    def load_settings(self, input_header):
        """Load the header from json
        @param[in] input_header The header from the ring"""
        write_file = open(self.filename, 'w')
        write_file.write(str(json.loads(input_header.tostring())))

    def main(self, input_ring):
        """Put the header into the file
        @param[in] input_ring Contains the header in question"""
        self.gulp_size = 1
        span_dummy_generator = self.iterate_ring_read(input_ring)
        span_dummy_generator.next()




