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
import ephem
from bifrost import affinity
from bifrost.block import *
from bifrost.ring import Ring
from bifrost.addon.leda.blocks import DadaReadBlock, NewDadaReadBlock, CableDelayBlock
from bifrost.addon.leda.blocks import UVCoordinateBlock, BaselineSelectorBlock
from bifrost.addon.leda.blocks import SlicingBlock, ImagingBlock
from bifrost.addon.leda.model_block import ScalarSkyModelBlock

FFT_SIZE = 512
N_STANDS = 250
#N_STANDS = 720
N_BASELINE = N_STANDS*(N_STANDS+1)//2
UV_SPAN_SIZE = N_BASELINE*6*4      # all the baselines then 6 floats - stand numbers, U and V, and Re/Im visibility
GRID_SPAN_SIZE = FFT_SIZE**2

def load_telescope(filename):
    with open(filename, 'r') as telescope_file:
        telescope = json.load(telescope_file)
    coords_local = np.array(telescope['coords']['local']['__data__'], dtype=np.float32)
    # Reshape into ant,column
    coords_local = coords_local.reshape(coords_local.size/4,4)
    ant_coords = coords_local[:,1:]
    inputs = np.array(telescope['inputs']['__data__'], dtype=np.float32)
    # Reshape into ant,pol,column
    inputs      = inputs.reshape(inputs.size/7/2,2,7)
    delays      = inputs[:,:,5]*1e-9
    dispersions = inputs[:,:,6]*1e-9
    return telescope, ant_coords, delays, dispersions

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

def disturb_visibilities(uvw_matrix, jones):
    """Apply jones to uvw matrix.
        uvw_matrix = np.zeros(shape=[
            1, self.num_stands, 
            2, self.num_stands, 
            2]).astype(np.complex64)
        jones = 2*np.random.rand(
            1, 2, N_STANDS, 2).astype(np.complex64)
            """
    n_stands = uvw_matrix.shape[1]
    for i in range(n_stands):
        for j in range(n_stands):
            uvw_matrix[0, i, :, j, :] = np.dot(
                np.dot(jones[0, :, i, :], uvw_matrix[0, i, :, j, :]),
                np.conj(jones[0, :, j, :]))
    return uvw_matrix
    
class FakeeVisBlock(TransformBlock):
    """Read a formatted file for fake visibility data"""
    def __init__(self, filename, num_stands, disturb = False):
        super(FakeeVisBlock, self).__init__()
        self.filename = filename
        self.num_stands = num_stands
        self.output_header = json.dumps(
            {'dtype':str(np.complex64),
             'nbit':64,
             'shape': [6, 1]})
        self.disturb = disturb
    def main(self, input_rings, output_rings):
        """Start the visibility generation.
        @param[out] output_ring Will contain the visibilities in [[stand1, stand2, u,v,re,im],[stand1,..],..]
        """
        # N_BASELINE is redundant
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
        np.random.seed(10)
        np.random.shuffle(uvw_data)
        # Strip a lot of the incoming values so we only have N_BASELINE visibilties. 
	# Then assign stand numbers.
	inner = np.array([ x for x in uvw_data if abs(x[2]) < 50 and abs(x[3]) < 50 ])
	uvw_data = np.append(inner, uvw_data, axis=0)[:n_baseline]
	# Assign stands
	for i in range(len(uvw_data)):
	    uvw_data[i][0] = baselines[i][0]
	    uvw_data[i][1] = baselines[i][1]

        uvw_matrix = np.zeros(shape=[
            1, self.num_stands, 
            2, self.num_stands, 
            2]).astype(np.complex64)
        #uv are given for each stand
        uv = np.zeros(shape=[
            self.num_stands, self.num_stands, 2])
        for row in uvw_data:
            stand1 = int(row[0])
            stand2 = int(row[1])
            uv[stand1, stand2, :] = [row[2], row[3]]
            uv[stand2, stand1, :] = [-row[2], -row[3]]
            uvw_matrix[0, stand1, 0, stand2, 0] = row[4]+1j*row[5]
            uvw_matrix[0, stand1, 1, stand2, 1] = row[4]+1j*row[5]
            uvw_matrix[0, stand2, 0, stand1, 0] = row[4]-1j*row[5]
            uvw_matrix[0, stand2, 1, stand1, 1] = row[4]-1j*row[5]

        jones = ((0.5+0.5j)*np.random.rand(
            1, 2, N_STANDS, 2)).astype(np.complex64)+0.25+0.25j
        jones[0, 0, :, 1] = np.zeros(N_STANDS).astype(np.complex64)
        jones[0, 1, :, 0] = np.zeros(N_STANDS).astype(np.complex64)
        if self.disturb:
            print "This sentence should occur only once."
            uvw_matrix = disturb_visibilities(uvw_matrix, jones)
        uv = np.array(uv).astype(np.float32)
        self.out_gulp_size = uvw_matrix.nbytes
        self.output_header = json.dumps({
            'dtype':str(np.complex64), 
            'nbit':64,
            'shape':uvw_matrix.shape})
        visibility_span_generator = self.iterate_ring_write(output_rings[0])
        visibility_span = visibility_span_generator.next()
        visibility_span.data_view(np.complex64)[0][:] = uvw_matrix.ravel()
        self.output_header = json.dumps({
            'dtype':str(np.float32), 
            'nbit':32,
            'shape':uv.shape})
        self.out_gulp_size = uv.nbytes
        uv_span_generator = self.iterate_ring_write(output_rings[1])
        uv_span = uv_span_generator.next()
        uv_span.data_view(np.float32)[0][:] = uv.ravel().astype(np.float32)

def generate_image_from_file(data_file, uv_coords_file, image_file):
    output_visibilities = np.loadtxt(data_file, dtype=np.float32).view(np.complex64).reshape((N_STANDS, 2, N_STANDS, 2))
    scalar_visibilities = output_visibilities[:, 0, :, 0]
    uv_points = np.loadtxt(uv_coords_file, dtype=np.float32).reshape((N_STANDS, N_STANDS, 2))
    data_to_grid = np.zeros(shape=[N_STANDS, N_STANDS, 4])
    for i in range(N_STANDS):
        for j in range(N_STANDS):
            data_to_grid[i, j] = np.array([
                uv_points[i, j, 0],
                uv_points[i, j, 1],
                np.real(scalar_visibilities[i, j]), 
                np.imag(scalar_visibilities[i, j])])
    new_blocks = []
    new_blocks.append((TestingBlock(data_to_grid), [], ['viz']))
    new_blocks.append((NearestNeighborGriddingBlock((512,512)), ['viz'], ['gridded']))
    new_blocks.append((IFFT2Block(), ['gridded'], ['ifftd']))
    new_blocks.append((WriteAsciiBlock('ifftd.txt'), ['ifftd'], []))
    Pipeline(new_blocks).main()
    dirty_image = np.abs(np.loadtxt('ifftd.txt', dtype=np.float32).view(np.complex64).reshape((512, 512)))
    from matplotlib import image
    image.imsave(image_file, dirty_image, cmap='gray')

blocks = []
dadafile = '/data2/hg/interfits/lconverter/WholeSkyL64_47.004_d20150203_utc181702_test/2015-04-08-20_15_03_0001133593833216.dada'
telescope, coords, delays, dispersions = load_telescope("/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json")
ovro = ephem.Observer()
ovro.lat = '37.239782'
ovro.lon = '-118.281679'
ovro.elevation = 1184.134
ovro.date = '2015/04/08 20:15:03.0001133593833216'
from itertools import islice
sources = {}
"""
with open('/data1/mcranmer/data/real/vlssr_catalog.txt', 'r') as file_in:
    iterator = 0
    for line in file_in:
        iterator += 1
        if iterator == 17:
            break
    iterator = 0
    number_sources = 100000
    total_flux = 1e-10
    for line in file_in:
        iterator += 1
        if iterator > number_sources:
            #break
            pass
        if line[0].isspace():
            continue
        try:
            flux = float(line[29:36].replace(" ",""))
        except:
            break
        if flux > 1:
            source_string = line[:-1]
            ra = line[:2]+':'+line[3:5]+':'+line[6:11]
            ra = ra.replace(" ", "")
            dec = line[12:15]+':'+line[16:18]+':'+line[19:23]
            dec = dec.replace(" ", "")
            total_flux += flux
            sources[str(iterator)] = {
                'ra': ra, 'dec': dec,
                'flux': flux, 'frequency': 58e6, 
                'spectral index': -0.2046}
                """
"""
sources['cyg'] = {
    'ra':'19:59:28.4', 'dec':'+40:44:02.1', 
    'flux': 10571.0, 'frequency': 58e6, 
    'spectral index': -0.2046}
"""
"""
sources['sun'] = {
    'ephemeris': ephem.Sun(),
    'flux': 250.0, 'frequency':20e6,
    'spectral index':+1.9920}
"""
frequencies = [40e6]
identity_matrix = np.ones((256, 256, 3), dtype=np.float32)
baselines_xyz = (identity_matrix*coords)-(identity_matrix*coords).transpose((1, 0, 2))
median_baseline = np.median(np.abs(baselines_xyz[:, :, 0] + 1j*baselines_xyz[:, :, 1]))
flags = np.ones(shape=[1, 256]).astype(np.int8)*2
jones = np.ones(shape=[1, 2, 256, 2]).astype(np.complex64)+1j*np.ones(shape=[1, 2, 256, 2]).astype(np.complex64)
jones[0, 0, :, 1] = 0.001+0.001j
jones[0, 1, :, 0] = 0.001+0.001j
"""
subblocks = []
subblocks.append((ScalarSkyModelBlock(ovro, coords, frequencies, sources), [], ['test_model']))
subblocks.append((WriteAsciiBlock('.log.txt'), ['test_model'], []))
Pipeline(subblocks).main()
"""
allvis = np.loadtxt('model-log.txt', dtype=np.float32)
test_vis = (allvis[2::4]+1j*allvis[3::4]).astype(np.complex64).reshape((256, 256))
model_visibilities = np.zeros(shape=[1, 256, 2, 256, 2]).astype(np.complex64)
model_visibilities[0, :, 0, :, 0] = test_vis[:, :]
model_visibilities[0, :, 1, :, 1] = test_vis[:, :]
#[nchan,nstand^,npol^,nstand,npol]


blocks.append((TestingBlock(jones), [], ['jones_in']))
blocks.append((TestingBlock(model_visibilities), [], ['model']))
blocks.append((
    NewDadaReadBlock(dadafile, output_chans=[100], time_steps=1),
    {'out': 'visibilities'}))
blocks.append((
    UVCoordinateBlock("/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json"), 
    {'out': 'uv_coords'}))
blocks.append((
    BaselineSelectorBlock(minimum_baseline=median_baseline/100),
    {'in_vis': 'visibilities', 'in_uv': 'uv_coords', 'out_vis': 'flagged_visibilities'}
    ))
blocks.append((
    GainSolveBlock(flags=flags, max_iterations=5000), 
    ['visibilities', 'model', 'jones_in'],
    ['calibrated_data', 'jones_out']))
blocks.append((
    SlicingBlock(np.s_[0, :, 0, :, 0]),
    {'in': 'calibrated_data', 'out': 'sliced_cal'}))
blocks.append((
    SlicingBlock(np.s_[0, :, :, 0, 0]),
    {'in': 'visibilities', 'out': 'sliced_uncal'}))
blocks.append((
    ReductionBlock(np.real, output_header={
        'dtype': str(np.float32),
        'nbit': 32}),
    {'in': 'sliced_cal', 'out': 'real_vis'}))
blocks.append((
   ReductionBlock(np.imag, output_header={
        'dtype': str(np.float32),
        'nbit': 32}),
    {'in': 'sliced_cal', 'out': 'imag_vis'}))
blocks.append((
    DStackBlock(),
    {'in_1': 'real_vis', 'in_2': 'imag_vis', 'out': 'realimag_vis'}))
blocks.append((
    DStackBlock(),
    {'in_1': 'realimag_vis', 'in_2': 'uv_coords', 'out': 'formatted_vis'}))
blocks.append((
    WriteAsciiBlock('log1.txt'), ['realimag_vis'], []))
blocks.append((
    WriteAsciiBlock('log2.txt'), ['uv_coords'], []))
Pipeline(blocks).main()
log_vis = np.loadtxt('log1.txt').reshape((256, 256, 2)).astype(np.float32)
log_uv = np.loadtxt('log2.txt').reshape((256, 256, 2)).astype(np.float32)
my_vis = np.zeros(shape=[256, 256, 4]).astype(np.float32)
my_vis[:, :, 0:2] = log_uv[:, :, :]
my_vis[:, :, 2:4] = log_vis[:, :, :]
#bad_stands = []
bad_stands = [ 0,56,57,58,59,60,61,62,63,72,74,75,76,77,78,82,83,84,85,86,87,91,92,93,104,120,121,122,123,124,125,126,127,128,145,148,157,161,164,168,184,185,186,187,188,189,190,191,197,220,224,225,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255 ]
for bad_stand in bad_stands:
    my_vis[bad_stand, :, 2:4] = 0
    my_vis[:, bad_stand, 2:4] = 0
blocks = []
blocks.append((TestingBlock(my_vis), [], ['test_vis']))
blocks.append((
    NearestNeighborGriddingBlock((256, 256)),
    ['test_vis'],
    ['grid']))
blocks.append((
    IFFT2Block(),
    ['grid'],
    ['ifftd']))
blocks.append((
    ImagingBlock(filename='sky.png', reduction=np.abs, log=True),
    {'in': 'ifftd'}))
Pipeline(blocks).main()

"""
blocks = []

jones = 0.5*((1+1j)*np.ones(shape=[
    1, 2, N_STANDS, 2])).astype(np.complex64)
jones[0, 0, :, 1] = np.zeros(N_STANDS).astype(np.complex64)
jones[0, 1, :, 0] = np.zeros(N_STANDS).astype(np.complex64)
flags = 2*np.ones(shape=[
    1, N_STANDS]).astype(np.int8)
blocks.append((FakeeVisBlock("mona_uvw.dat", N_STANDS), [], ['perfect', 'uv_coords']))
blocks.append((FakeeVisBlock("mona_uvw.dat", N_STANDS, disturb=True), [], ['uncalibrated', 'junk_uv_coords']))
blocks.append((TestingBlock(jones), [], ['jones_in']))
blocks.append((GainSolveBlock(flags, max_iterations=10000), ['uncalibrated', 'perfect', 'jones_in'], ['model_out', 'jones_out']))
blocks.append((WriteAsciiBlock('model.txt'), ['perfect'], []))
blocks.append((WriteAsciiBlock('uncalibrated.txt'), ['uncalibrated'], []))
blocks.append((WriteAsciiBlock('calibrated.txt'), ['model_out'], []))
blocks.append((WriteAsciiBlock('uv_coords.txt'), ['uv_coords'], []))
Pipeline(blocks).main()
#generate_image_from_file('model.txt', 'uv_coords.txt', 'model.png')
generate_image_from_file('uncalibrated.txt', 'uv_coords.txt', 'uncalibrated.png')
generate_image_from_file('calibrated.txt', 'uv_coords.txt', 'calibrated.png')
"""

"""
visibilities = np.array([np.linalg.det(
    np.array([
    [output_visibilities[i, 0, j, 0], output_visibilities[i, 0, j, 1]],
    [output_visibilities[i, 1, j, 0], output_visibilities[i, 1, j, 1]]
    ])) for i in range(N_STANDS) for j in range(N_STANDS)]).reshape(
        (N_STANDS, N_STANDS))
uv_points = np.loadtxt('uv_coords.txt', dtype=np.float32).reshape((N_BASELINE, 4))
#Now have visibilities at each uv coord given...
gridding_feed = []
for row in uv_points:
    stand1 = int(row[0])
    stand2 = int(row[1])
    curr_viz = visibilities[stand1, stand2]
    gridding_feed.append([row[2], row[3], np.real(curr_viz), np.imag(curr_viz)])
gridding_feed = np.array(gridding_feed)


"""


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
