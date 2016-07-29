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
from bifrost import affinity
from matplotlib import pyplot as plt


class WaterfallBlock(object):
    """This block creates a waterfall block
        based on the data in a ring, and stores it
        in the headers"""

    def __init__(
            self, ring, imagename,
            core=-1, gulp_nframe=4096):
        """
        @param[in] ring Ring containing a multichannel
            timeseries
        @param[in] imagename Filename to store the
            waterfall image
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        @param[in] gulp_size How many bytes of the ring to
            read at once.
        """
        self.ring = ring
        self.imagename = imagename
        self.core = core
        self.gulp_nframe = gulp_nframe
        self.header = {}

    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        waterfall_matrix = self.generate_waterfall_matrix()
        self.save_waterfall_plot(waterfall_matrix)

    def save_waterfall_plot(self, waterfall_matrix):
        """Save an image of the waterfall plot using
            thread-safe backend for pyplot, and labelling
            the plot using the header information from the
            ring
        @param[in] waterfall_matrix x axis is frequency and
            y axis is time. Values should be power.
            """
        plt.ioff()
        print "Interactive mode off"
        print waterfall_matrix.shape
        fig = pylab.figure()
        ax = fig.gca()
        header = self.header
        ax.set_xticks(
            np.arange(0, 1.33, 0.33) * waterfall_matrix.shape[1])
        ax.set_xticklabels(
            header['fch1'] - np.arange(0, 4) * header['foff'])
        ax.set_xlabel("Frequency [MHz]")
        ax.set_yticks(
            np.arange(0, 1.125, 0.125) * waterfall_matrix.shape[0])
        ax.set_yticklabels(
            header['tstart'] + header['tsamp'] * np.arange(0, 1.125, 0.125) * waterfall_matrix.shape[0])
        ax.set_ylabel("Time (s)")
        plt.pcolormesh(
            waterfall_matrix, axes=ax, figure=fig)
        fig.autofmt_xdate()
        fig.savefig(
            self.imagename, bbox_inches='tight')
        plt.close(fig)

    def generate_waterfall_matrix(self):
        """Create a matrix for a waterfall image
            based on the ring's data"""
        waterfall_matrix = None
        self.ring.resize(self.gulp_nframe)
        # Generate a waterfall matrix:
        for sequence in self.ring.read(guarantee=True):
            ## Get the sequence's header as a dictionary
            self.header = json.loads(
                "".join(
                    [chr(item) for item in sequence.header]))
            tstart = self.header['tstart']
            tsamp = self.header['tsamp']
            nchans = self.header['frame_shape'][0]
            gulp_size = self.gulp_nframe * nchans * self.header['nbit']
            waterfall_matrix = np.zeros(shape=(0, nchans))
            print tstart, tsamp, nchans
            for span in sequence.read(gulp_size):
                array_size = span.data.shape[1] / nchans
                frequency = self.header['fch1']
                try:
                    curr_data = np.reshape(
                        span.data, (-1, nchans))
                    waterfall_matrix = np.concatenate(
                        (waterfall_matrix, curr_data), 0)
                except:
                    print "Bad shape for waterfall"
                    pass
        return waterfall_matrix