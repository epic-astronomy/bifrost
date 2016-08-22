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

"""@module blocks
This file contains blocks specific to LEDA-OVRO.
"""

import os
import json
import ephem
import numpy as np
import matplotlib
## Use a graphical backend which supports threading
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from bifrost.block import MultiTransformBlock, SourceBlock

def load_telescope(filename):
    """Load in LEDA's settings file (coded as a JSON)
        @param[in] filename The JSON file.
        @returns telescope - The full JSON file as a dictionary
        @returns ant_coords - The coordinates of the LEDA stands
        @returns delays - The delays of each stand [antenna, polarization]
        @returns dispersions- The dispersions of each stand [antenna, polarization]"""
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

DADA_HEADER_SIZE = 4096
LEDA_OUTRIGGERS = [252, 253, 254, 255, 256]
BAD_STANDS = [
    0, 56, 57, 58, 59, 60, 61, 62, 63, 72, 74, 75, 76, 77, 78, 82, 83, 84, 85, 86, 87,
    91, 92, 93, 104, 120, 121, 122, 123, 124, 125, 126, 127, 128, 145, 148, 157, 161,
    164, 168, 184, 185, 186, 187, 188, 189, 190, 191, 197, 220, 224, 225, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
LEDA_NSTATIONS = 256
SPEED_OF_LIGHT = 299792458.
LEDA_SETTINGS_FILE = "/data1/mcranmer/data/real/leda/lwa_ovro.telescope.json"
OVRO_EPHEM = ephem.Observer()
OVRO_EPHEM.lat = '37.239782'
OVRO_EPHEM.lon = '-118.281679'
OVRO_EPHEM.elevation = 1184.134
OVRO_EPHEM.date = '2016/07/26 16:17:00.00'
COORDINATES, DELAYS, DISPERSIONS = load_telescope(LEDA_SETTINGS_FILE)[1:]

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

def triangle_matrix_indices(side_length):
    """Returns the indices of the lower triangle in a 2d square matrix
        @param[in] side_length The number of rows (or columns) in the matrix"""
    indices = []
    for row in range(side_length):
        for column in range(row+1):
            indices.append([row, column])
    return np.array(indices)

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
        """Put dada file data into output ring"""
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
            triangle_indices = triangle_matrix_indices(side_length=nstand)
            row_indices = triangle_indices[:, 0]
            column_indices = triangle_indices[:, 1]
            data[:, row_indices, column_indices, :, :] = dadafile_data
            data[:, column_indices, row_indices, :, :] = dadafile_data.conj()
            vis_span[:] = data.ravel()
        self.file_object.close()

def calculate_cable_delay_matrix(frequencies, delays, dispersions):
    """Calculate the cable delays, then build a matrix to apply to the visibilities
        @param[in] frequencies Frequencies of data in (Hz)
        @param[in] delays Delays for each antenna (s)
        @param[in] dispersions Dispersion for each antenna (?)"""
    cable_delays = (delays+dispersions)/np.sqrt(frequencies)*SPEED_OF_LIGHT*0.82
    cable_delay_weights = np.exp(1j*2*np.pi*cable_delays/SPEED_OF_LIGHT*frequencies)
    return cable_delay_weights.astype(np.complex64)

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
        self.cable_delay_matrix = calculate_cable_delay_matrix(
            frequencies,
            delays,
            dispersions)
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
        coords_local = coords_local.reshape(coords_local.size/4, 4)
        antenna_coordinates = coords_local[:, 1:]
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
        flag_matrix = baselines >= self.minimum_baseline
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
        self.dtype = np.float32
    def load_settings(self):
        """Calculate the outgoing slice size"""
        self.gulp_size['in'] = int(np.product(self.header['in']['shape']))*\
            self.header['in']['nbit']//8
        output_array = np.array(
            [np.zeros(self.header['in']['shape'], dtype=np.int8)[self.indices]])
        self.gulp_size['out'] = output_array.nbytes*self.header['in']['nbit']//8
        self.header['out'] = self.header['in'].copy()
        self.dtype = np.dtype(
            self.header['in']['dtype'].split()[1].split(".")[1].split("'")[0]).type
        if len(output_array.shape) > 1:
            self.header['out']['shape'] = output_array.shape[1:]
        else:
            self.header['out']['shape'] = [1, 1]
    def main(self):
        """Slice the incoming spans into the output span"""
        for in_span, out_span in self.izip(self.read('in'), self.write('out')):
            shaped_data = in_span.view(self.dtype).reshape(self.header['in']['shape'])[self.indices]
            out_span.view(shaped_data.dtype.type)[:] = shaped_data.ravel()

class ImagingBlock(MultiTransformBlock):
    """Image a 2D matrix using matplotlib.pyplot.imsave"""
    ring_names = {
        'in': """Data to be imaged. Each input span will
            generate an image under the same name"""}
    def __init__(self, filename, reduction=None, log=False):
        """@param[in] filename Where to save the image
            @param[in] reduction A function to be applied to the matrix before imaging (e.g., abs)
            @param[in] log Whether or not to take an np.log of the matrix before imaging"""
        super(ImagingBlock, self).__init__()
        self.filename = filename
        self.reduction = reduction
        self.log = log
        self.dtype = np.float32
    def load_settings(self):
        """Update gulp size settings based on inputted header"""
        self.gulp_size['in'] = int(np.product(self.header['in']['shape']))*\
            self.header['in']['nbit']//8
        self.dtype = np.dtype(
            self.header['in']['dtype'].split()[1].split(".")[1].split("'")[0]).type
    def main(self):
        """Load in each span, and save them to the same file as an image (it will update)"""
        for in_span in self.read('in'):
            data_to_plot = np.copy(in_span[0].view(self.dtype)).reshape(self.header['in']['shape'])
            if callable(self.reduction):
                data_to_plot = self.reduction(data_to_plot)
            if self.log:
                plt.imshow(np.log(data_to_plot))
            else:
                plt.imshow(data_to_plot)
            plt.colorbar()
            plt.savefig(self.filename)

def horizontal_to_cartesian(azimuth, altitude, radius=1):
    """transform to the cartesian coordinate system"""
    x_coord = radius*np.cos(altitude)*np.sin(azimuth)
    y_coord = radius*np.cos(altitude)*np.cos(azimuth)
    z_coord = radius*np.sin(altitude)
    return np.array([x_coord, y_coord, z_coord])

class ScalarSkyModelBlock(SourceBlock):
    """Generate a simple scalar model of the sky with input point sources
    Requires the ephem library"""
    def __init__(
            self, observer, antenna_coordinates,
            frequencies, sources):
        """
        @param[in] observer Should be an ephem.Observer()
        @param[in] antenna_coordinates A list of cartesian coordinates in meters,
            relative to the observer center
        @param[in] frequencies All the frequencies in Hz.
        @param[in] sources Dictionary of sources, which are also dictionaries,
            and contain flux, ra, dec and, optionally, frequency and spectral_index"""
        super(ScalarSkyModelBlock, self).__init__()
        self.observer = observer
        self.antenna_coordinates = antenna_coordinates
        self.frequencies = np.array(frequencies).reshape((-1, 1))
        self.sources = sources
    def compute_antenna_phase_delays(self, source_position_ra, source_position_dec):
        """Calculate antenna phase delays based on the source position
        @param[in] source_position_ra Should be a string
        @param[in] source_position_dec Should be a string"""
        source_position = ephem.FixedBody()
        #TODO: PyEphem will apparently change these to ra, dec in future version
        source_position._ra = source_position_ra
        source_position._dec = source_position_dec
        source_position.compute(self.observer)
        azimuth = np.float(repr(source_position.az))
        altitude = np.float(repr(source_position.alt))
        cartesian_direction_to_source = horizontal_to_cartesian(azimuth, altitude)
        antenna_distance_to_observatory = np.sum(
            self.antenna_coordinates*cartesian_direction_to_source, axis=-1)
        antenna_time_delays = antenna_distance_to_observatory/299792458.
        antenna_phase_delays = np.exp(-1j*2*np.pi*antenna_time_delays*self.frequencies)
        below_horizon = (altitude < 0)
        if below_horizon:
            antenna_phase_delays.ravel()[:] = 0
        return antenna_phase_delays
    def extrapolate_flux(self, source):
        """Estimate the source flux at the desired model frequency
            @param[in] source Dictionary containing settings for the source."""
        if 'spectral index' in source:
            spectral_index = float(source['spectral index'])
            frequency_reference = float(source['frequency'])
            flux_reference = float(source['flux'])
            # Using S \propto frequency^(spectral_index)
            log_extrapolated_flux = np.log(flux_reference)
            log_extrapolated_flux += spectral_index*(np.log(self.frequencies/frequency_reference))
            extrapolated_flux = np.exp(log_extrapolated_flux)
            return extrapolated_flux
        else:
            return list((source['flux'],)*len(self.frequencies))
    def generate_model(self):
        """Calculate the total visibilities for the antenna baselines"""
        number_antennas = len(self.antenna_coordinates)
        number_frequencies = len(self.frequencies)
        total_visibilities = np.zeros(
            (number_frequencies, number_antennas, number_antennas)).astype(np.complex64)
        for source in self.sources.itervalues():
            if 'ephemeris' in source:
                source['ephemeris'].compute(self.observer)
                source['ra'] = source['ephemeris'].ra
                source['dec'] = source['ephemeris'].dec
            antenna_phase_delays = self.compute_antenna_phase_delays(
                source['ra'], source['dec'])
            baseline_phase_delays = np.einsum(
                'fi,fj->fij', antenna_phase_delays, antenna_phase_delays.conj())
            extrapolated_flux = self.extrapolate_flux(source)
            for frequency_index in range(number_frequencies):
                visibilities = baseline_phase_delays[frequency_index]*\
                    extrapolated_flux[frequency_index]
                scaled_visibilities = visibilities/number_antennas**2
                total_visibilities[frequency_index] += scaled_visibilities
        return total_visibilities.astype(np.complex64)
    def main(self, output_ring):
        """Generate a model of the sky and put it on a single output span
            @param[in] output_ring The ring to put this visibility model on.
            Is entered as [[u,v,re,im],[u,..],..]"""
        visibilities = self.generate_model()
        number_antennas = self.antenna_coordinates.shape[0]
        baselines_xyz = self.antenna_coordinates[None, :] - self.antenna_coordinates[:, None]
        baselines_u = baselines_xyz[:, :, 0].reshape(-1)
        baselines_v = baselines_xyz[:, :, 1].reshape(-1)
        self.gulp_size = baselines_u.nbytes*4*self.frequencies.size
        model_shape = [self.frequencies.size, number_antennas, number_antennas, 4]
        self.output_header = json.dumps({
            'nbit': 32,
            'dtype': str(np.float32),
            'shape': model_shape})
        out_span_generator = self.iterate_ring_write(output_ring)
        out_span = out_span_generator.next()
        out_span = out_span.data_view(np.float32)[0]
        out_span = out_span.reshape(model_shape)
        for frequency_index in range(self.frequencies.size):
            real_visibilities = np.real(visibilities[frequency_index].ravel())
            imaginary_visibilities = np.imag(visibilities[frequency_index].ravel())
            out_span[frequency_index].ravel()[0::4] = baselines_u
            out_span[frequency_index].ravel()[1::4] = baselines_v
            out_span[frequency_index].ravel()[2::4] = real_visibilities
            out_span[frequency_index].ravel()[3::4] = imaginary_visibilities
