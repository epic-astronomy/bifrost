
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

import numpy as np
from bifrost.block import SourceBlock, SinkBlock
from bifrost.pipeline import Pipeline
import json


class NpTxBlock(SourceBlock):
    """Block for debugging purposes.
    Allows you to pass arbitrary N-dimensional arrays in initialization,
    which will be outputted into a ring buffer"""

    def __init__(self, dsize):
        """@param[in] test_array A list or numpy array containing test data"""
        super(NpTxBlock, self).__init__()

        self.seed = 1
        self.test_data = np.arange(dsize, dtype='float32') + self.seed

        self.output_header_dict = {'nbit': 32,
             'dtype': str(np.float32),
             'shape': self.test_data.shape,
             'checksum' : self.seed}

        self.output_header = json.dumps(self.output_header_dict)

    def main(self, output_ring):
        """Put the test array onto the output ring
        @param[in] output_ring Holds the flattend test array in a single span"""
        self.gulp_size = self.test_data.nbytes
        self.initialize_oring(output_ring)

        ospan_generator = self.iterate_ring_write(sequence_name=str(self.seed))

        for ii in range(100):
            #print ii
            self.seed = ii
            self.output_header_dict['checksum'] = str(ii)
            self.output_header = json.dumps(self.output_header_dict)
            self.test_data = np.arange(self.test_data.shape[0], dtype='float32') + self.seed

            ospan = ospan_generator.next()
            ospan.data_view(np.float32)[0][:] = self.test_data.ravel()


class NpRxBlock(SinkBlock):

    """Copies input ring's data into ascii format
        in a text file."""

    def __init__(self, dsize):
        """@param[in] filename Name of file to write ascii to
        @param[out] gulp_size How much of the file to write at once"""
        super(NpRxBlock, self).__init__()
        self.nbit = 8
        self.dtype = np.uint8
        self.gulp_size = dsize * 4

    def load_settings(self, input_header):
        header_dict = json.loads(input_header.tostring())
        self.nbit = header_dict['nbit']
        self.dtype = np.dtype(header_dict['dtype'].split()[1].split(".")[1].split("'")[0]).type

    def main(self, input_ring):
        """Initiate the writing to filename
        @param[in] input_rings First ring in this list will be used for
            data
        @param[out] output_rings This list of rings won't be used."""
        span_generator = self.iterate_ring_read(input_ring)
        for span in span_generator:
            unpacked_data = span.data_view(self.dtype)
            print unpacked_data[0:4]
            print self.header


def test_np():
    blocks = []

    dset = np.arange(8192, dtype='float32')

    np_txblock = NpTxBlock(8192)
    np_rxblock = NpRxBlock(8192)

    blocks.append([np_txblock, [], ['ring1']])
    blocks.append([np_rxblock, ['ring1'], [] ])

    Pipeline(blocks).main()

if __name__ == '__main__':
    test_np()
