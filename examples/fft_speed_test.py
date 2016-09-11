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
import time
from bifrost.block import Pipeline, NumpyBlock, NumpySourceBlock, GPUBlock

N = 1
print "N =", N

print "Times for serial:"
print time.time()

for i in range(10):
    array = np.ones(shape=[1000000]).astype(np.complex64)

    for _ in range(N):
        tmp_array = np.fft.fft(array)
        array = np.fft.ifft(tmp_array)

print time.time()


def generate_100_arrays():
    print time.time()
    for _ in range(10):
        yield np.ones(shape=[1000000]).astype(np.complex64)

blocks = [(NumpySourceBlock(generate_100_arrays, changing=False), {'out_1': 0})]
for i in range(N):
    blocks.append((NumpyBlock(np.fft.fft), {'in_1': 2*i, 'out_1': 2*i+1}))
    blocks.append((NumpyBlock(np.fft.ifft), {'in_1': 2*i+1, 'out_1': 2*i+2}))

print "Times for Bifrost, GPU-disabled:"
Pipeline(blocks).main()
print time.time()

from bifrost.fft import fft as bf_fft
from bifrost.fft import ifft as bf_ifft
def gpu_fft(gpu_array):
    """Perform an fft on the input"""
    bifrost_gpu_array = gpu_array.as_BFarray()
    bf_fft(bifrost_gpu_array, bifrost_gpu_array)
    gpu_array.buffer = bifrost_gpu_array.data
    return gpu_array

def gpu_ifft(gpu_array):
    """Perform an ifft on the input"""
    bifrost_gpu_array = gpu_array.as_BFarray()
    bf_ifft(bifrost_gpu_array, bifrost_gpu_array)
    gpu_array.buffer = bifrost_gpu_array.data
    return gpu_array
print "Times for Bifrost, GPU-enabled:"
blocks = [(NumpySourceBlock(generate_100_arrays, changing=False), {'out_1': 0})]
blocks.append((GPUBlock(gpu_fft), {'in_1':0, 'out_1':1}))
blocks.append((GPUBlock(gpu_ifft), {'in_1':1, 'out_1':2}))
Pipeline(blocks).main()
