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
from bifrost.ring import Ring


def test_ring_initialization():

    # Create a ring and confirm it is uninitialized
    r = Ring()
    assert r.initialized is False
    assert r.__repr__() == "<Uninitialized Bifrost Ring: in system space>"

    # Resize ring by initalizing and check that size values have populated
    r.resize(8192)
    assert r.initialized is True

    # Compare against known defaults
    repr = "<Bifrost Ring: in system space>\n"
    repr += "\tTotal span:       32768\n"
    repr += "\tBuffer factor:    4\n"
    repr += "\tContiguous span:  8192\n"
    repr += "\tRinglets:         1"
    assert r.__repr__() == repr
    assert r.total_span == 32768
    assert r.buffer_factor == 4
    assert r.contiguous_span == 8192
    assert r.nringlet == 1

    print "test_ring_initialization() PASSED."


def test_ring_write():
    dsize = 16384

    r = Ring()
    r.resize(dsize)
    d = np.zeros(dsize / 4, dtype='float32')

    oring = r.begin_writing()
    oseq  = oring.begin_sequence('test')

    oseq.write_array(d)
    oseq.write_array(d + 1)
    oseq.write_array(d + 2)
    oseq.write_array(d + 3)

    print "test_ring_write() PASSED."



if __name__ == "__main__":
    test_ring_initialization()
    test_ring_write()