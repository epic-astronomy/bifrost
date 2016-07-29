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

import json

import numpy as np
from bifrost.sigproc import unpack

from python.bifrost.block import TransformBlock


class FFTBlock(TransformBlock):
    """Performs complex to complex IFFT on input ring data"""

    def __init__(self, gulp_size):
        super(FFTBlock, self).__init__()
        self.nbit = 8
        self.dtype = np.uint8

    def load_settings(self, input_header):
        header = json.loads(input_header.tostring())
        self.out_gulp_size = self.gulp_size * 64 / header['nbit']
        self.nbit = header['nbit']
        self.dtype = np.dtype(header['dtype'].split()[1].split(".")[1].split("'")[0]).type
        header['nbit'] = 64
        header['dtype'] = str(np.complex64)
        self.output_header = json.dumps(header)

    def main(self, input_rings, output_rings):
        """
        @param[in] input_rings First ring in this list will be used for
            data
        @param[out] output_rings First ring in this list will be used for
            data output."""
        for ispan, ospan in self.ring_transfer(input_rings[0], output_rings[0]):
            if self.nbit < 8:
                unpacked_data = unpack(ispan.data_view(self.dtype), self.nbit)
            else:
                unpacked_data = ispan.data_view(self.dtype)
            result = np.fft.fft(unpacked_data.astype(np.float32))
            ospan.data_view(np.complex64)[0][:] = result[0][:]


class IFFTBlock(TransformBlock):
    """Performs complex to complex IFFT on input ring data"""

    def __init__(self, gulp_size):
        super(IFFTBlock, self).__init__()
        self.gulp_size = gulp_size
        self.nbit = 8
        self.dtype = np.uint8

    def load_settings(self, input_header):
        header = json.loads(input_header.tostring())
        self.out_gulp_size = self.gulp_size * 64 / header['nbit']
        self.nbit = header['nbit']
        self.dtype = np.dtype(header['dtype'].split()[1].split(".")[1].split("'")[0]).type
        header['nbit'] = 64
        header['dtype'] = str(np.complex64)
        self.output_header = json.dumps(header)

    def main(self, input_rings, output_rings):
        """
        @param[in] input_rings First ring in this list will be used for
            data
        @param[out] output_rings First ring in this list will be used for
            data output."""
        for ispan, ospan in self.ring_transfer(input_rings[0], output_rings[0]):
            if self.nbit < 8:
                unpacked_data = unpack(ispan.data_view(self.dtype), self.nbit)
            else:
                unpacked_data = ispan.data_view(self.dtype)
            result = np.fft.ifft(unpacked_data)
            ospan.data_view(np.complex64)[0][:] = result[0][:]