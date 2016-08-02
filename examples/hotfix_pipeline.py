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
import v_p_matrices
import copy
from bifrost import affinity
from bifrost.block import *
from bifrost.ring import Ring

FFT_SIZE = 512
N_STANDS = 250
#N_STANDS = 720
N_BASELINE = N_STANDS*(N_STANDS+1)//2
UV_SPAN_SIZE = N_BASELINE*6*4      # all the baselines then 6 floats - stand numbers, U and V, and Re/Im visibility
GRID_SPAN_SIZE = FFT_SIZE**2

class FakeCalBlock(TransformBlock):
    """This block does simulated calibration by creating visibilities and fitting them to a model."""
    def __init__(self, flags, num_stands):
        super(FakeCalBlock, self).__init__(gulp_size=UV_SPAN_SIZE)
	self.num_stands = num_stands
    def main(self, input_rings, output_rings):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        self.calibrate(input_rings, output_rings)
    def calibrate(self, input_rings, output_rings):
        # How do i get these
        nbit = 32
        dtype = np.float32
        for ispan, ospan in self.ring_transfer(input_rings[0], output_rings[0]):
	    uv_list = ispan.data.reshape(N_BASELINE, 6*nbit/8).view(dtype)
	    if True:
	        # Calibrate. It's a bit round-about but there's a reason.
	        # Take the vis values (FFT components) and generate a model P from them, using some random J matrices.
	        # These V, J, P are called "perfect" because they are an exact solution to P = J V J
		num_stands = 8			# Pretend this many to cut down the work
	        perfect_V = [ [ None for i in range(num_stands) ] for j in range(num_stands) ]
	        i = 0
	        for j in range(num_stands):
  	            for k in range(j+1, num_stands):
		        vis = complex(uv_list[i][4], uv_list[i][5])
		        zero = complex(0, 0)
    		        perfect_V[j][k] = v_p_matrices.Matrix(vis, zero, zero, zero)
		        i += 1
	        perfect_V[4][5].printm()
    	        cal_matrices = v_p_matrices.V_P_J(num_stands)
    	        perfect_P, perfect_J = cal_matrices.create_perfect_P(perfect_V)		# perfect
	        # Now generate an imperfect V by perturbing the orginal ones. They are not perturbed randomly,
	        # but using another hidden set of J
                V_perturb = cal_matrices.perturb_V(perfect_V, perfect_J, perfect_P)
	        # Now using the perfect_P as the model, and the orginal perfect_J as J estimates, see if we can find
	        # a solution for V_perturb. In normal circumstances this would be the end. However we want to compare the solution
	        # against the original visibilities (actually I want to use the solved visibilities). Thus generate a model
		# from the solution. From that model and the perfect J's, generate solution visibilities to match against the originals.
		# These visibilities are V_cal. 
    	        V_cal = cal_matrices.solve(V_perturb, perfect_J, perfect_P)
	        V_cal[4][5].printm()
	        # Unpack
	        new_uv_list = copy.deepcopy(uv_list)		# Return V_cal to see if the image changes
	        i = 0
	        for j in range(num_stands):
  	            for k in range(j+1, num_stands):
		        new_uv_list[i][4] = V_cal[j][k].matrix[0][0].real
		        new_uv_list[i][5] = V_cal[j][k].matrix[0][0].imag
		        i += 1
	        # Send out
	        ospan.data[0][:] = new_uv_list.view(dtype=np.uint8).ravel()
	    else: ospan.data[0][:] = uv_list.view(dtype=np.uint8).ravel()

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

 	#uv_list = np.loadtxt("mona_uvw.dat", dtype=np.float32, usecols={1, 2, 3, 4, 5, 6})

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
              if self.in_range(x, y): data[x+FFT_SIZE/2, y+FFT_SIZE/2] += visibility
	      if conjugates:
                x = -x
                y = -y
                if self.in_range(x, y): data[x+FFT_SIZE/2, y+FFT_SIZE/2] += np.conj(visibility)
            else:		# Gaussian blur onto the grid
              self.gauss_here(u, v, visibility, data, g_sum)
              if conjugates: self.gauss_here(-u, -v, np.conj(visibility), data, g_sum)     


	# After gridding, invert the Fourier components and get an image
        image = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data))))
        plt.imshow(image, cmap="gray")
        plt.savefig("x.png")

class FakeeVisBlock(SourceBlock):
    """Read a formatted file for fake visibility data"""
    def __init__(self, filename, num_stands):
        super(FakeeVisBlock, self).__init__()
        self.filename = filename
        self.num_stands = num_stands
        self.output_header = json.dumps(
            {'dtype':str(np.complex64),
             'nbit':64,
             'shape': [4, 1]})
    def main(self, output_ring):
        """Start the visibility generation.
        @param[out] output_ring Will contain the visibilities in [[stand1, stand2, u,v,re,im],[stand1,..],..]
        """
	n_baseline = self.num_stands*(self.num_stands+1)//2

	# Generate index of baselines -> stand
	baselines = [ None for i in range(n_baseline) ]
	index = 0
	for st1 in range(self.num_stands):
	    for st2 in range(st1, self.num_stands):
  		baselines[index] = ( st1, st2 )
		index += 1

        self.output_header = json.dumps(
            {'dtype':str(np.float32), 
            'nbit':32})
        uvw_data = np.loadtxt(
            self.filename, dtype=np.float32, usecols={1, 2, 3, 4, 5, 6})
        np.random.shuffle(uvw_data)
        # Strip a lot of the incoming values so we only have N_BASELINE visibilties. 
	# Then assign stand numbers.
	inner = np.array([ x for x in uvw_data if abs(x[2]) < 50 and abs(x[3]) < 50 ])
	uvw_data = np.append(inner, uvw_data, axis=0)[:n_baseline]
	print n_baseline, len(uvw_data)
	# Assign stands
	for i in range(len(uvw_data)):
	    uvw_data[i][0] = baselines[i][0]
	    uvw_data[i][1] = baselines[i][1]
        uvw_matrix = np.zeros(shape=[
            1, N_STANDS, 
            2, N_STANDS, 
            2]).astype(np.complex64)
        for row in uvw_data:
            stand1 = int(row[0])
            stand2 = int(row[1])
            uvw_matrix[0, stand1, 0, stand2, 0] += row[4]+1j*row[5]
            uvw_matrix[0, stand1, 1, stand2, 1] += row[4]+1j*row[5]
        print np.max(uvw_matrix)
        self.gulp_size = uvw_matrix.nbytes
        self.header = json.dumps({
            'dtype':str(np.complex64), 
            'nbit':64,
            'shape':uvw_matrix.shape})
        self.output_header = json.dumps({
            'dtype':str(np.complex64), 
            'nbit':64,
            'shape':uvw_matrix.shape})
        span_generator = self.iterate_ring_write(output_ring)
        span = span_generator.next()
        span.data_view(np.complex64)[0][:] = uvw_matrix.ravel()

blocks = []
jones = 1*np.ones(shape=[
    1, 2, N_STANDS, 2]).astype(np.complex64)
#jones[0, 0, :, 1] = np.zeros(N_STANDS)[:].astype(np.complex64)
#jones[0, 1, :, 0] = np.zeros(N_STANDS)[:].astype(np.complex64)
flags = 2*np.ones(shape=[
    1, N_STANDS]).astype(np.int8)
blocks.append((FakeeVisBlock("mona_uvw.dat", N_STANDS), [], ['uncalibrated']))
blocks.append((FakeeVisBlock("mona_uvw.dat", N_STANDS), [], ['perfect']))
blocks.append((TestingBlock(jones), [], ['jones_in']))
blocks.append((GainSolveBlock(flags, max_iterations=20000), ['uncalibrated', 'perfect', 'jones_in'], ['model_out', 'jones_out']))
blocks.append((WriteAsciiBlock('.log.txt'), ['model_out'], []))
Pipeline(blocks).main()
output_visibilities = np.loadtxt('.log.txt', dtype=np.float32).view(np.complex64).reshape((1, N_STANDS, 2, N_STANDS, 2))
"""
bad_stands = [ 0,56,57,58,59,60,61,62,63,72,74,75,76,77,78,82,83,84,85,86,87,91,92,93,104,120,121,122,123,124,125,126,127,128,145,148,157,161,164,168,184,185,186,187,188,189,190,191,197,220,224,225,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255 ]
flags = 2*np.ones(shape=[
    self.nchan, self.nstand]).astype(np.int8)
model = 10*np.random.rand(
    self.nchan, self.nstand, 
    self.npol, self.nstand, 
    self.npol).astype(np.complex64)
data = 10*np.random.rand(
    self.nchan, self.nstand, 
    self.npol, self.nstand, 
    self.npol).astype(np.complex64)
jones = np.ones(shape=[
    self.nchan, self.npol, 
    self.nstand, self.npol]).astype(np.complex64)
out_jones = self.generate_new_jones(model, data, jones, flags)
self.assertGreater(np.max(np.abs(out_jones - jones.ravel())), 1e-3)
blocks = []
blocks.append((FakeeVisBlock("mona_uvw.dat", N_STANDS), [], [0]))
blocks.append((TestingBlock(), [0], []))
Pipeline(blocks).main()
blocks = []
blocks.append((TestingBlock(model), [], ['model']))
blocks.append((TestingBlock(data), [], ['data']))
blocks.append((TestingBlock(jones), [], ['jones_in']))
blocks.append((
    GainSolveBlock(flags=flags), 
    ['data', 'model', 'jones_in'], 
    ['jones_out']))
blocks.append((WriteAsciiBlock('.log.txt'), ['jones_out'], []))
Pipeline(blocks).main()
out_jones = np.loadtxt('.log.txt').astype(np.float32).view(np.complex64)
return out_jones
"""
