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

import bifrost
import numpy as np

from python.bifrost.block import TransformBlock, insert_zeros_evenly


class FoldBlock(TransformBlock):
    """This block folds a signal into a histogram"""

    def __init__(
            self, bins, period=1e-3,
            gulp_size=4096 * 256, dispersion_measure=0,
            core=-1):
        """
        @param[in] bins The total number of bins to fold into
        @param[in] period Period to fold over (s)
        @param[in] gulp_size How many bytes of the ring to
            read at once.
        @param[in] dispersion_measure DM of the desired
            source (pc cm^-3)
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        """
        super(FoldBlock, self).__init__()
        self.bins = bins
        self.gulp_size = gulp_size
        self.period = period
        self.dispersion_measure = dispersion_measure
        self.core = core
        self.data_settings = {}

    def calculate_bin_indices(
            self, tstart, tsamp, data_size):
        """Calculate the bin that each time sample should be
            added to
        @param[in] tstart Time of the first element (s)
        @param[in] tsamp Difference between the times of
            consecutive elements (s)
        @param[in] data_size Number of elements
        @return Which bin each sample is folded into
        """
        arrival_time = tstart + tsamp * np.arange(data_size)
        phase = np.fmod(arrival_time, self.period)
        return np.floor(phase / self.period * self.bins).astype(int)

    def calculate_delay(self, frequency, reference_frequency):
        """Calculate the time delay because of frequency dispersion
        @param[in] frequency The current channel's frequency(MHz)
        @param[in] reference_frequency The frequency of the
            channel we will hold at zero time delay(MHz)"""
        frequency_factor = \
            np.power(reference_frequency / 1000, -2) - \
            np.power(frequency / 1000, -2)
        return 4.15e-3 * self.dispersion_measure * frequency_factor

    def load_settings(self, input_header):
        self.data_settings = json.loads(
            "".join(
                [chr(item) for item in input_header]))
        self.output_header = json.dumps({'nbit': 32, 'dtype': str(np.float32)})

    def main(self, input_rings, output_rings):
        """Generate a histogram from the input ring data
        @param[in] input_rings List with first ring containing
            data of interest. Must terminate before histogram
            is generated.
        @param[out] output_rings First ring in this list
            will contain the output histogram"""
        histogram = np.reshape(
            np.zeros(self.bins).astype(np.float32),
            (1, self.bins))
        tstart = None
        for span in self.iterate_ring_read(input_rings[0]):
            nchans = self.data_settings['frame_shape'][0]
            if tstart is None:
                tstart = self.data_settings['tstart']
            frequency = self.data_settings['fch1']
            for chan in range(nchans):
                modified_tstart = tstart - self.calculate_delay(
                    frequency,
                    self.data_settings['fch1'])
                frequency -= self.data_settings['foff']
                sort_indices = np.argsort(
                    self.calculate_bin_indices(
                        modified_tstart, self.data_settings['tsamp'],
                        span.data.shape[1] / nchans))
                sorted_data = span.data[0][chan::nchans][sort_indices]
                extra_elements = np.round(self.bins * (1 - np.modf(
                    float(span.data.shape[1] / nchans) / self.bins)[0])).astype(int)
                sorted_data = insert_zeros_evenly(sorted_data, extra_elements)
                histogram += np.sum(
                    sorted_data.reshape(self.bins, -1), 1).astype(np.float32)
            tstart += self.data_settings['tsamp'] * \
                      self.gulp_size * 8 / self.data_settings['nbit'] / nchans
        self.out_gulp_size = self.bins * 4
        out_span_generator = self.iterate_ring_write(output_rings[0])
        out_span = out_span_generator.next()
        bifrost.memory.memcpy(
            out_span.data_view(dtype=np.float32),
            histogram)