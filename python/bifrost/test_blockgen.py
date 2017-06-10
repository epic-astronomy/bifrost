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

import unittest

import numpy as np
import bifrost as bf
from bifrost import pipeline as bfp
from bifrost import blocks as blocks
from bifrost.blocks import CopyBlock
import blockgen

class CallbackBlock(CopyBlock):
    """Testing-only block which calls user-defined
    functions on sequence and on data"""
    def __init__(self, iring, seq_callback, data_callback=None, *args, **kwargs):
        super(CallbackBlock, self).__init__(iring, *args, **kwargs)
        self.seq_callback  = seq_callback
        if data_callback != None:
            self.data_callback = data_callback
    def on_sequence(self, iseq):
        self.seq_callback(iseq)
        return super(CallbackBlock, self).on_sequence(iseq)
    def on_data(self, ispan, ospan):
        self.data_callback(ispan, ospan)
        return super(CallbackBlock, self).on_data(ispan, ospan)
    def data_callback(self, ispan, ospan):
        pass


class BlockgenTest(unittest.TestCase):
    """ Test all aspects of the blockgen.py code """
    def test_simple_gen(self):
        """ Create a simple noise-generating block and check output """
        self.sequence_count = 0
        self.data_count = 0
        def check_sequence(seq):
            tensor = seq.header['_tensor']
            self.assertEqual(tensor['shape'],  [-1,100])
            self.assertEqual(tensor['dtype'],  'f32')
            self.sequence_count += 1
        def check_data(data):
            self.assertGreater(np.stdev(data), 0.5)
            self.data_count += 1
        def generate_data():
            for _ in range(10):
                yield np.random.uniform(size=(100,)).astype(np.float32)

        noise = blockgen.source(generate_data)
        test = CallbackBlock(noise, check_sequence, check_data)
        pipeline = bfp.get_default_pipeline()
        pipeline.run()
        self.assertEqual(self.sequence_count, 1)
        self.assertGreater(self.data_count, 1)
    #TODO: explicit header arg in function makes source pass it in

