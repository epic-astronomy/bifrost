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

"""@package blocks
This file contains blocks specific to LEDA-OVRO.
"""

import os
import json
import itertools
import numpy as np
import matplotlib
## Use a graphical backend which supports threading
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import bifrost
from bifrost.addon.leda import bandfiles
from bifrost.block import SourceBlock, MultiTransformBlock

DADA_HEADER_SIZE = 4096
LEDA_OUTRIGGERS = [252, 253, 254, 255, 256]
LEDA_NSTATIONS = 256
SPEED_OF_LIGHT = 299792458.

def cast_string_to_number(string):
    """Attempt to convert a string to integer or float"""
    try:
        return int(string)
    except ValueError:
        pass
    try:
        return float(string)
    except ValueError:
        pass
    return string

# Read a collection of DADA files and form an array of time series data over
# many frequencies.
# TODO: Add more to the header.
# Add a list of frequencies present. This could be the full band,
# but there could be gaps. Downstream functionality has to be aware of the gaps.
# Allow specification of span size based on time interval
class DadaReadBlock(SourceBlock):
    # Assemble a group of files in the time direction and the frequency direction
    # time_stamp is of the form "2016-05-24-11:04:38", or a DADA file ending in .dada
    def __init__(self, filename, core=-1, gulp_nframe=4096):
        self.CHANNEL_WIDTH = 0.024
        self.SAMPLING_RATE = 41.66666666667e-6
        self.N_CHAN = 109
        self.N_BEAM = 2
        self.HEADER_SIZE = 4096
        self.OBS_OFFSET = 1255680000
        self.gulp_nframe = gulp_nframe
        self.core = core
        beamformer_scans = []
        beamformer_scans.append(bandfiles.BandFiles(filename))  # Just one file
        # Report what we've got
        print "Num files in time:",  len(beamformer_scans)
        print "File and number:"
        for scan in beamformer_scans:
            print os.path.basename(scan.files[0].name)+":", len(scan.files)
        self.beamformer_scans = beamformer_scans         # List of full-band time steps
    def main(self, output_ring):
        bifrost.affinity.set_core(self.core)
        self.oring = output_ring
        # Calculate some constants for sizes
        length_one_second = int(round(1/self.SAMPLING_RATE))
        ring_span_size = length_one_second*self.N_CHAN*4                                        # 1 second, all the channels (109) and then 4-byte floats
        file_chunk_size = length_one_second*self.N_BEAM*self.N_CHAN*2                               # 1 second, 2 beams, 109 chans, and 2 1-byte ints (real, imag)
        number_of_seconds = 120         # Change this
        ohdr = {}
        ohdr["shape"] = (self.N_CHAN, 1)
        ohdr["frame_shape"] = ( self.N_CHAN, 1 )
        ohdr["nbit"] = 32
        ohdr["dtype"] = str(np.float32)
        ohdr["tstart"] = 0
        ohdr["tsamp"] = self.SAMPLING_RATE
        ohdr['foff'] = self.CHANNEL_WIDTH
        #print length_one_second, ring_span_size, file_chunk_size, number_of_chunks
        with self.oring.begin_writing() as oring:
            # Go through the files by time. 
            for scan in self.beamformer_scans:
                # Go through the frequencies
                for f in scan.files:
                    print "Opening", f.name
                    with open(f.name,'rb') as ifile:
                        ifile.read(self.HEADER_SIZE)
                        ohdr["cfreq"] = f.freq
                        ohdr["fch1"] = f.freq
                        self.oring.resize(ring_span_size)
                        with oring.begin_sequence(f.name, header=json.dumps(ohdr)) as osequence:
                            for i in range(number_of_seconds):
                                # Get a chunk of data from the file. The whole band is used, but only a chunk of time (1 second).
                                # Massage the data so it can go through the ring. That means changng the data type and flattening.
                                try:
                                    data = np.fromfile(ifile, count=file_chunk_size, dtype=np.int8).astype(np.float32)
                                except:
                                    print "Bad read. Stopping read."
                                    return
                                if data.size != length_one_second*self.N_BEAM*self.N_CHAN*2:
                                    print "Bad data shape. Stopping read."
                                    return
                                data = data.reshape(length_one_second, self.N_BEAM, self.N_CHAN, 2)
                                power = (data[...,0]**2 + data[...,1]**2).mean(axis=1)  # Now have time by frequency.
                                # Send the data
                                with osequence.reserve(ring_span_size) as wspan:
                                    wspan.data[0][:] = power.view(dtype=np.uint8).ravel()
class DadaFileRead(object):
    """File object for reading in a dada file
    @param[in] output_chans The frequency channels to output
    @param[in] time_steps The number of time samples to output"""
    def __init__(self, filename, output_chans, time_steps):
        super(DadaFileRead, self).__init__()
        self.filename = filename
        self.file_object = open(filename, 'rb')
        self.dada_header = {}
        self.output_chans = output_chans
        self.time_steps = time_steps
        self.framesize = {'full': 0, 'outrigger': 0}
        self.shape = []
    def parse_dada_header(self):
        """Get settings out of the dada file's header"""
        header_string = self.file_object.read(DADA_HEADER_SIZE)
        self.dada_header = {}
        for line in header_string.split('\n'):
            try:
                key, value = line.split(None, 1)
            except ValueError:
                break
            key = key.strip()
            value = value.strip()
            value = cast_string_to_number(value)
            self.dada_header[key] = value
    def interpret_header(self):
        """Generate settings based on the dada's header"""
        outrig_nbaseline = sum([LEDA_NSTATIONS-x for x in range(len(LEDA_OUTRIGGERS))])
        nchan = self.dada_header['NCHAN']
        npol = self.dada_header['NPOL']
        navg = self.dada_header['NAVG']
        frequency_channel_width = self.dada_header['BW']*1e6 / float(nchan)
        assert self.dada_header['DATA_ORDER'] == "TIME_SUBSET_CHAN_TRIANGULAR_POL_POL_COMPLEX"
        nbaseline = LEDA_NSTATIONS*(LEDA_NSTATIONS+1)//2
        noutrig_per_full = int(navg/frequency_channel_width + 0.5)
        self.framesize['full'] = nchan*nbaseline*npol*npol
        self.framesize['outrigger'] = noutrig_per_full*nchan*outrig_nbaseline*npol*npol
        self.shape = (nchan, nbaseline, npol, npol)
    def dada_read(self):
        """Read in the entirety of the dada file"""
        filesize = os.path.getsize(self.filename) - DADA_HEADER_SIZE
        tot_framesize = self.framesize['full'] + self.framesize['outrigger']
        ntime = float(filesize)/(tot_framesize*8)
        for i in range(self.time_steps):
            if i >= ntime:
                print "Stopping read of Dada file."
                break
            full_data = np.fromfile(
                self.file_object,
                dtype=np.complex64,
                count=self.framesize['full'])
            # Outrigger data is not used:
            np.fromfile(
                self.file_object,
                dtype=np.complex64,
                count=self.framesize['outrigger'])
            try:
                full_data = full_data.reshape(self.shape)
                select_channel_data = full_data[self.output_chans, :, :, :]
                yield select_channel_data
            except ValueError:
                print "Bad data reshape, possibly due to end of file"
                break

def build_baseline_ants(nant):
	nbaseline = nant*(nant+1)/2
	baseline_ants = np.empty((nbaseline,2), dtype=np.int32)
	for i in xrange(nant):
		for j in xrange(i+1):
			b = i*(i+1)/2 + j
			baseline_ants[b,0] = i
			baseline_ants[b,1] = j
	return baseline_ants

class NewDadaReadBlock(DadaFileRead, MultiTransformBlock):
    """Read a dada file in with frequency channels in ringlets."""
    ring_names = {
        'out': """Visibilities outputted as a complex matrix. 
            Shape is [frequencies, nstand, nstand, npol, npol]."""}
    def __init__(self, filename, output_chans, time_steps):
        """@param[in] filename The dada file.
        @param[in] output_chans The frequency channels to output
        @param[in] time_steps The number of time samples to output"""
        super(NewDadaReadBlock, self).__init__(
            output_chans=output_chans,
            time_steps=time_steps,
            filename=filename)
    def main(self):
        """Put dada file data onto output_ring
        @param[out] output_ring Ring to contain outgoing data"""
        self.parse_dada_header()
        self.interpret_header()
        nchan = self.dada_header['NCHAN']
        npol = self.dada_header['NPOL']
        nstand = LEDA_NSTATIONS
        output_shape = [
            len(self.output_chans),
            nstand,
            nstand,
            npol,
            npol]
        sizeofcomplex64 = 8
        self.gulp_size['out'] = np.product(output_shape)*sizeofcomplex64
        self.header['out'] = {
            'nbit':64,
            'dtype':str(np.complex64),
            'shape':output_shape}
        for dadafile_data, vis_span in self.izip(
                self.dada_read(),
                self.write('out')):
            data = np.empty(output_shape, dtype=np.complex64)
            baseline_ants = build_baseline_ants(nstand)
            ants_i = baseline_ants[:,0]
            ants_j = baseline_ants[:,1]
            data[:,ants_i,ants_j,:,:] = dadafile_data
            data[:,ants_j,ants_i,:,:] = dadafile_data.conj()
            vis_span[:] = data.view(np.float32).ravel()
        self.file_object.close()
class CableDelayBlock(MultiTransformBlock):
    """Apply cable delays to a visibility matrix"""
    ring_names = {
        'in': "Visibilities WITHOUT cable delays added",
        'out': "Visibilities WITH cable delays added"}
    def __init__(self, frequencies, delays, dispersions):
        """@param[in] frequencies Frequencies of data in (Hz)
        @param[in] delays Delays for each antenna (s)
        @param[in] dispersions Dispersion for each antenna (?)"""
        super(CableDelayBlock, self).__init__()
        self.cable_delay_matrix = self.calculate_cable_delay_matrix(
            frequencies,
            delays,
            dispersions)
    def calculate_cable_delay_matrix(self, frequencies, delays, dispersions):
        """Calculate the cable delays,
            then build a matrix to apply to the visibilities"""
        cable_delays = (delays+dispersions)/np.sqrt(frequencies)*SPEED_OF_LIGHT*0.82
        cable_delay_weights = np.exp(1j*2*np.pi*cable_delays/SPEED_OF_LIGHT*frequencies)
        return cable_delay_weights.astype(np.complex64)
    def load_settings(self):
        """Gulp data appropriately to the shape of the input"""
        sizeofcomplex64 = 8
        self.gulp_size['in'] = np.product(self.header['in']['shape'])*sizeofcomplex64
        self.header['out'] = self.header['in']
        self.gulp_size['out'] = self.gulp_size['in']
    def main(self):
        """Apply the cable delays to the output matrix"""
        for inspan, outspan in self.izip(self.read('in'), self.write('out')):
            visibilities = np.copy(inspan.view(np.complex64).reshape(self.header['in']['shape']))
            for i in range(self.header['in']['shape'][1]):
                for j in range(self.header['in']['shape'][1]):
                    visibilities[0, i, j, 0, 0] *= self.cable_delay_matrix[i, 0]
                    visibilities[0, i, j, 0, 0] *= self.cable_delay_matrix[j, 0].conj()
                    visibilities[0, i, j, 1, 1] *= self.cable_delay_matrix[i, 1]
                    visibilities[0, i, j, 1, 1] *= self.cable_delay_matrix[j, 1].conj()
            outspan[:] = visibilities.ravel().view(np.float32)[:]
class UVCoordinateBlock(MultiTransformBlock):
    """Read the UV coordinates in from a telescope json file, and put them into a ring"""
    ring_names = {
        'out': "uv coordinates of all of the stands. Shape is [nstand, nstand, 2]"}
    def __init__(self, filename):
        """@param[in] filename The json file containing telescope specifications."""
        super(UVCoordinateBlock, self).__init__()
        self.filename = filename
    def load_telescope_uv(self):
        """Load the json file, and assemble the uv coordinates"""
        with open(self.filename, 'r') as telescope_file:
            telescope = json.load(telescope_file)
        coords_local = np.array(telescope['coords']['local']['__data__'], dtype=np.float32)
        coords_local = coords_local.reshape(coords_local.size/4,4)
        antenna_coordinates = coords_local[:,1:]
        nstand = antenna_coordinates.shape[0]
        identity_matrix = np.ones((nstand, nstand, 3), dtype=np.float32)
        baselines_xyz = (identity_matrix*antenna_coordinates)-(identity_matrix*antenna_coordinates).transpose((1, 0, 2))
        baselines_uv = baselines_xyz[:, :, 0:2]
        return baselines_uv
    def main(self):
        """Assemble the coordinates, and put them out as a single span on a ring"""
        baselines_uv = self.load_telescope_uv()
        nstand = baselines_uv.shape[0]
        self.header['out'] = {
            'nbit':32,
            'dtype':str(np.float32),
            'shape':[nstand, nstand, 2]}
        self.gulp_size['out'] = nstand**2*2*4
        for out_span in self.izip(self.write('out')):
            out_span = out_span[0]
            out_span[:] = baselines_uv.astype(np.float32).ravel()
            break

class BaselineSelectorBlock(MultiTransformBlock):
    """Read in visibilities and UV coordinates, and flag visibilities if
        they do not satisfy the baseline requirements."""
    ring_names = {
        'in_vis': """The input visibilities, in the shape
            [frequencies, nstand, nstand, npol, npol]""",
        'in_uv': """The input uv coords, in the shape
            [nstand, nstand, 2]""",
        'out_vis': """The output visibilities, in the shape
            [frequencies, nstand, nstand, npol, npol]. Flagged
            visibilities are zero."""}
    def __init__(self, minimum_baseline=0):
        """@param[in] minimum_baseline The minimum baseline in uv space"""
        super(BaselineSelectorBlock, self).__init__()
        self.minimum_baseline = minimum_baseline
    def load_settings(self):
        """Update read settings based on inputted header"""
        if 'in_uv' in self.header:
            self.gulp_size['in_uv'] = np.product(self.header['in_uv']['shape'])*8
        if 'in_vis' in self.header:
            self.gulp_size['in_vis'] = np.product(self.header['in_vis']['shape'])*8
            self.gulp_size['out_vis'] = self.gulp_size['in_vis']
            self.header['out_vis'] = self.header['in_vis']
    def calculate_flag_matrix(self, uv_coordinates):
        """Calculate the flags based on the entered minimum baseline"""
        baselines = np.abs(uv_coordinates[:, :, 0]+1j*uv_coordinates[:, :, 1])
        flag_matrix = baselines>self.minimum_baseline
        return flag_matrix.astype(np.int8)
    def main(self):
        """Read in the uv coordinates, and use it to flag the data.
            (i.e., to set the visibilities to zero if they have a badly sized baseline"""
        uv_coordinate_generator = self.read('in_uv')
        uv_coordinates = uv_coordinate_generator.next()[0]
        uv_coordinates = uv_coordinates.reshape(
            self.header['in_uv']['shape'])
        flag_matrix = self.calculate_flag_matrix(uv_coordinates)
        for in_vis, out_vis in self.izip(self.read('in_vis'), self.write('out_vis')):
            visibilities = np.copy(in_vis.view(np.complex64).reshape([1, 256, 256, 2, 2]))
            for i in range(256):
                for j in range(256):
                    visibilities[0, i, j, :, :] *= flag_matrix[i, j]
            out_vis[:] = visibilities.ravel().view(np.float32)

class SlicingBlock(MultiTransformBlock):
    """Slice incoming data arrays with numpy indices"""
    ring_names = {
        'in': """The array to slice. Number of dimensions should be the same 
            as the slice""",
        'out': """The sliced array. The number of dimensions might be different
            from the incoming array."""}
    def __init__(self, indices):
        """@param[in] indices Numpy slices (e.g., produced with np.s_[1, 2]).
            This get used on input data."""
        super(SlicingBlock, self).__init__()
        self.indices = indices
    def load_settings(self):
        """Calculate the outgoing slice size"""
        self.gulp_size['in'] = int(np.product(self.header['in']['shape']))*self.header['in']['nbit']//8
        output_settings = np.array([np.zeros(self.header['in']['shape'], dtype=np.int8)[self.indices]])
        self.gulp_size['out'] = output_settings.nbytes*self.header['in']['nbit']//8
        self.header['out'] = self.header['in'].copy()
        self.dtype = np.dtype(self.header['in']['dtype'].split()[1].split(".")[1].split("'")[0]).type
        if len(output_settings.shape) > 1:
            self.header['out']['shape'] = output_settings.shape[1:]
        else:
            self.header['out']['shape'] = [1, 1]
    def main(self):
        for in_span, out_span in self.izip(self.read('in'), self.write('out')):
            shaped_data = in_span.view(self.dtype).reshape(self.header['in']['shape'])[self.indices]
            out_span.view(np.complex64)[:] = shaped_data.ravel()

class ImagingBlock(MultiTransformBlock):
    ring_names = {
        'in': """Data to be imaged. Each input span will
            generate an image under the same name"""}
    def __init__(self, filename, reduction=None, log=False):
        super(ImagingBlock, self).__init__()
        self.filename = filename
        self.reduction = reduction
        self.log = log
    def load_settings(self):
        self.gulp_size['in'] = int(np.product(self.header['in']['shape']))*self.header['in']['nbit']//8
        self.dtype = np.dtype(self.header['in']['dtype'].split()[1].split(".")[1].split("'")[0]).type
    def main(self):
        for in_span in self.read('in'):
            data_to_plot = np.copy(in_span[0].view(self.dtype)).reshape(self.header['in']['shape'])
            if callable(self.reduction):
                data_to_plot = self.reduction(data_to_plot)
            plt.imshow(data_to_plot, cmap='gray', interpolation='nearest')
            plt.colorbar()
            plt.savefig(self.filename)
