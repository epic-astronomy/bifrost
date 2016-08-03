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

from bifrost.sigproc import SigprocFile

from bifrost.block import SourceBlock


class SigprocReadBlock(SourceBlock):
    """This block reads in a sigproc filterbank
    (.fil) file into a ring buffer"""

    def __init__(
            self, filename,
            gulp_nframe=4096, core=-1):
        """
        @param[in] filename filterbank file to read
        @param[in] gulp_nframe Time samples to read
            in at a time
        @param[in] core Which CPU core to bind to (-1) is
            any
        """
        super(SigprocReadBlock, self).__init__()
        self.filename = filename
        self.gulp_nframe = gulp_nframe
        self.core = core

    def main(self, output_ring):
        """Read in the sigproc file to output_ring
        @param[in] output_ring Ring to write to"""
        with SigprocFile().open(self.filename, 'rb') as ifile:
            ifile.read_header()
            ohdr = {}
            ohdr['frame_shape'] = (ifile.nchans, ifile.nifs)
            ohdr['frame_size'] = ifile.nchans * ifile.nifs
            ohdr['frame_nbyte'] = ifile.nchans * ifile.nifs * ifile.nbits / 8
            ohdr['frame_axes'] = ('pol', 'chan')
            ohdr['ringlet_shape'] = (1,)
            ohdr['ringlet_axes'] = ()
            ohdr['dtype'] = str(ifile.dtype)
            ohdr['nbit'] = ifile.nbits
            ohdr['tsamp'] = float(ifile.header['tsamp'])
            ohdr['tstart'] = float(ifile.header['tstart'])
            ohdr['fch1'] = float(ifile.header['fch1'])
            ohdr['foff'] = float(ifile.header['foff'])
            self.output_header = json.dumps(ohdr)
            self.gulp_size = self.gulp_nframe * ifile.nchans * ifile.nifs * ifile.nbits / 8
            out_span_generator = self.iterate_ring_write(output_ring)
            for span in out_span_generator:
                size = ifile.file_object.readinto(span.data.data)
                span.commit(size)
                if size == 0:
                    break
