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

from python.bifrost.block import TransformBlock


class KurtosisBlock(TransformBlock):
    """This block performs spectral kurtosis and cleaning
        on sigproc-formatted data in rings"""

    def __init__(self, gulp_size=1048576, core=-1):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] output_ring Ring will contain a 1d
            timeseries that will be cleaned of RFI
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        """
        super(KurtosisBlock, self).__init__()
        self.gulp_size = gulp_size
        self.core = core
        self.output_header = {}
        self.settings = {}
        self.nchan = 1
        self.dtype = np.uint8

    def load_settings(self, input_header):
        self.output_header = input_header
        self.settings = json.loads(input_header.tostring())
        self.nchan = self.settings["frame_shape"][0]
        dtype_str = self.settings["dtype"].split()[1].split(".")[1].split("'")[0]
        self.dtype = np.dtype(dtype_str)

    def main(self, input_rings, output_rings):
        """Calls a kurtosis algorithm and uses the result
            to clean the input data of RFI, and move it to the
            output ring."""
        expected_v2 = 0.5
        for ispan, ospan in self.ring_transfer(input_rings[0], output_rings[0]):
            nsample = ispan.size / self.nchan / (self.settings['nbit'] / 8)
            # Raw data -> power array of the right type
            power = ispan.data.reshape(
                nsample,
                self.nchan * self.settings['nbit'] / 8).view(self.dtype)
            # Following section 3.1 of the Nita paper.
            # the sample is a power value in a frequency bin from an FFT,
            # i.e. the beamformer values in a channel
            number_samples = power.shape[0]
            bad_channels = []
            for chan in range(self.nchan):
                nita_s1 = np.sum(power[:, chan])
                nita_s2 = np.sum(power[:, chan] ** 2)
                # equation 21
                nita_v2 = (number_samples / (number_samples - 1)) * \
                          (number_samples * nita_s2 / (nita_s1 ** 2) - 1)
                if abs(expected_v2 - nita_v2) > 0.1:
                    bad_channels.append(chan)
            flag_power = power.copy()
            for chan in range(self.nchan):
                if chan in bad_channels:
                    flag_power[:, chan] = 0  # set bad channel to zero
            ospan.data[0][:] = flag_power.view(dtype=np.uint8).ravel()