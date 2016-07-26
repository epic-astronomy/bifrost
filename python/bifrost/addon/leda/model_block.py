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

"""@module model_block This module has block definitions for model generation"""

import ephem
import json
import numpy as np
from itertools import izip
from bifrost.block import SourceBlock

def horizontal_to_cartesian(az, alt, radius=1):
    """transform to the cartesian coordiante system"""
    x_coord = radius*np.cos(alt)*np.sin(az)
    y_coord = radius*np.cos(alt)*np.cos(az)
    z_coord = radius*np.sin(alt)
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
        @param[in] frequencies All the frequencies """
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
        source_position._ra = source_position_ra
        source_position._dec = source_position_dec
	source_position.compute(self.observer)
	az = np.float(repr(source_position.az))
        alt = np.float(repr(source_position.alt))
        cartesian_direction_to_source = horizontal_to_cartesian(az, alt)
        antenna_distance_to_observatory = np.sum(
            self.antenna_coordinates*cartesian_direction_to_source, axis=-1)
        antenna_time_delays = antenna_distance_to_observatory/299792458.
        antenna_phase_delays = np.exp(-1j*2*np.pi*antenna_time_delays*self.frequencies)
        return antenna_phase_delays
    def generate_model(self):
        """Calculate the total visibilities on the antenna baselines"""
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
            total_visibilities += (source['flux']/number_antennas**2)*baseline_phase_delays
        return total_visibilities.astype(np.complex64)
    def main(self, output_ring):
        """Generate a model of the sky and put it on a single output span
        @param[in] output_ring The ring to put this visibility model on. 
            Is entered as [[u,v,re,im],[u,..],..]"""
        visibilities = self.generate_model()
        number_antennas = self.antenna_coordinates.shape[0]
        identity_matrix = np.ones((
            len(self.antenna_coordinates),
            len(self.antenna_coordinates), 
            3), dtype=np.float32)
        baselines_xyz = (identity_matrix*self.antenna_coordinates)-(identity_matrix*self.antenna_coordinates).transpose((1, 0, 2))
        baselines_u = baselines_xyz[:, :, 0].reshape(-1)
        baselines_v = baselines_xyz[:, :, 1].reshape(-1)
        self.gulp_size = baselines_u.nbytes*4
        self.output_header = json.dumps({
            'nbit':32,
            'dtype':str(np.float32)})
        out_span_generator = self.iterate_ring_write(output_ring)
        for frequency_index in range(self.frequencies.size):
            out_span = out_span_generator.next()
            real_visibilities = visibilities[frequency_index].reshape(-1).view(np.float32)[0::2]
            imaginary_visibilities = visibilities[frequency_index].reshape(-1).view(np.float32)[1::2]
            out_span.data_view(np.float32)[0][0::4] = baselines_u
            out_span.data_view(np.float32)[0][1::4] = baselines_v
            out_span.data_view(np.float32)[0][2::4] = real_visibilities
            out_span.data_view(np.float32)[0][3::4] = imaginary_visibilities 
