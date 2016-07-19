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
import threading
from bifrost import affinity
from bifrost.block import *
from bifrost.ring import Ring

FFT_SIZE = 512
N_BASELINE = 512*512
UV_SPAN_SIZE = N_BASELINE*6*4      # all the baselines then 6 floats - stand numbers, U and V, and Re/Im visibility
GRID_SPAN_SIZE = FFT_SIZE**2

# For debbuging to show the UV data
class PrintBlock(TransformBlock):
  def __init__(self):
    super(PrintBlock, self).__init__(gulp_size=UV_SPAN_SIZE)

  def main(self, input_rings, output_rings):
    i = 0
    # How do i get these
    nbit = 32
    dtype = np.float32
    for span in self.iterate_ring_read(input_rings[0]):
      uv_list = span.data.reshape(N_BASELINE, 6*nbit/8).view(dtype) 
      for k in range(10): print "PrintOp", k, uv_list[k]
      i += 1

# Read visibilities which happen to belong to an image 
# i.e. they are Fourier components from an FFT of an image.
# Some can be missing.
class FakeVisBlock(TransformBlock):

  def __init__(self):
    super(FakeVisBlock, self).__init__(gulp_size=UV_SPAN_SIZE)

  def main(self, input_rings, output_rings):

    ohdr = {}
    ohdr["frame_shape"] = ( N_BASELINE, 1 )
    ohdr["nbit"] = 32
    ohdr["dtype"] = str(np.float32)
    ohdr["tstart"] = 0
    ohdr["tsamp"] = 41.6666666666666e-6


    data = np.zeros((N_BASELINE, 6), dtype=np.float32)
    i = 0
    for line in open("mona_uvw.dat"):
      # Each line in file has: UVW <stand1> <stand2> <U> <V> <Vis Re> <Vis Im>
      uvw = line.split()
      data[i, :] = [ float(uvw[1]), float(uvw[2]), float(uvw[3]), float(uvw[4]), float(uvw[5]), float(uvw[6])  ]
      i += 1
    print "Loaded", i, "values"

    self.output_header = json.dumps(ohdr)
    out_span_generator = self.iterate_ring_write(output_rings[0])
    for span in out_span_generator:
      span.data[0, :] = data.view(dtype=np.uint8).ravel()

class GridBlock(TransformBlock):
    """This block performs gridding of visibilities (1 pol only) onto UV grid"""

    def __init__(self, flags):
      super(GridBlock, self).__init__(gulp_size=UV_SPAN_SIZE)
      self.flags = flags

    def main(self, input_rings, output_rings):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        self.grid(input_rings, output_rings)

    def in_range(self, x, y):
      #if not (-(FFT_SIZE/2) <= x and x < FFT_SIZE/2 and -(FFT_SIZE/2) <= y and y < FFT_SIZE/2): print "something out", x, y
      return -(FFT_SIZE/2) <= x and x < FFT_SIZE/2 and -(FFT_SIZE/2) <= y and y < FFT_SIZE/2

    def gauss_val(self, x_dist, y_dist):
      return np.exp(-(((x_dist)**2)/0.5+((y_dist)**2))/0.5)

    def gauss_here(self, u, v, visibility, data, norm):
      x = int(round(u))
      y = int(round(v))

      for i in range(x-1, x+2):
        for j in range(y-1, y+2):
          if self.in_range(i, j): data[i+data.shape[0]/2, j+data.shape[1]/2] += visibility*self.gauss_val(float(i)-u, float(j)-v)/norm

    def grid(self, input_rings, output_rings):
      data = np.zeros((FFT_SIZE, FFT_SIZE), dtype=np.complex64)

      # Get sum of points that hit the grid from the gaussian for normalization purposes
      # Only approximate.
      g_sum = 0.0
      for i in range(-1, 2):	# 3x3 grid 
        for j in range(-1, 2):
          g_sum += self.gauss_val(i, j)

      # How do i get these
      nbit = 32
      dtype = np.float32

      for span in self.iterate_ring_read(input_rings[0]):
        uv_list = span.data.reshape(N_BASELINE, 6*nbit/8).view(dtype)

        for uv in uv_list:
          st1 = int(uv[0])
          st2 = int(uv[1])
          u = uv[2]
          v = uv[3]
          re = uv[4]
          im = uv[5]
          visibility = complex(re, im)

          if st1 not in self.flags and st2 not in self.flags:
            x = int(round(u))
            y = int(round(v))
 
            quick = False		# if true then nearest neighbour
	    conjugates = False		# Insert them only for telescope data, not for fake
            if quick:	# nearest neighbour
              if self.in_range(x, y): data[x+FFT_SIZE/2, y+FFT_SIZE/2] = visibility
	      if conjugates:
                x = -x
                y = -y
                if self.in_range(x, y): data[x+FFT_SIZE/2, y+FFT_SIZE/2] = visibility
            else:		# Gaussian blur onto the grid
              self.gauss_here(u, v, visibility, data, g_sum)
              if conjugates: self.gauss_here(-u, -v, visibility, data, g_sum)     


	# After gridding, invert the Fourier components and get an image
        image = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data))))
        plt.imshow(image, cmap="gray")
        plt.savefig("x.png")

bad_stands = [ 0,56,57,58,59,60,61,62,63,72,74,75,76,77,78,82,83,84,85,86,87,91,92,93,104,120,121,122,123,124,125,126,127,128,145,148,157,161,164,168,184,185,186,187,188,189,190,191,197,220,224,225,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255 ]
blocks = []
blocks.append((FakeVisBlock(), [], [0]))
blocks.append((GridBlock([]), [0], []))
Pipeline(blocks).main()

