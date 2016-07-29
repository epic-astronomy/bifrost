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

import threading

from bifrost.ring import Ring

from bifrost.block import SourceBlock, SinkBlock


class Pipeline(object):
    """Class which connects blocks linearly, with
        one ring between each block. Does this by creating
        one ring for each input/output 'port' of each block,
        and running data through the rings."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.rings = {}
        for index in self.unique_ring_names():
            if isinstance(index, Ring):
                self.rings[str(index)] = index
            else:
                self.rings[index] = Ring()

    def unique_ring_names(self):
        """Return a list of unique ring indices"""
        all_names = []
        for block in self.blocks:
            for port in block[1:]:
                for index in port:
                    if isinstance(index, Ring):
                        all_names.append(index)
                    else:
                        all_names.append(str(index))
        return set(all_names)

    def main(self):
        """Start the pipeline, and finish when all threads exit"""
        threads = []
        for block in self.blocks:
            input_rings = []
            output_rings = []
            input_rings.extend(
                [self.rings[str(ring_index)] for ring_index in block[1]])
            output_rings.extend(
                [self.rings[str(ring_index)] for ring_index in block[2]])
            if issubclass(type(block[0]), SourceBlock):
                threads.append(threading.Thread(
                    target=block[0].main,
                    args=[output_rings[0]]))
            elif issubclass(type(block[0]), SinkBlock):
                threads.append(threading.Thread(
                    target=block[0].main,
                    args=[input_rings[0]]))
            else:
                threads.append(threading.Thread(
                    target=block[0].main,
                    args=[input_rings, output_rings]))
        for thread in threads:
            thread.daemon = True
            thread.start()
        for thread in threads:
            # Wait for exit
            thread.join()
