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

"""@package block This file defines a generic block class.

Right now the only possible block type is one
of a simple transform which works on a span by span basis.
"""
import json
import threading
import ctypes
from contextlib import nested
import numpy as np
import matplotlib
## Use a graphical backend which supports threading
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import bifrost
from bifrost import affinity
from bifrost.libbifrost import _bf
from bifrost.ring import Ring
from bifrost.GPUArray import GPUArray
from bifrost.sigproc import SigprocFile, unpack

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
            if issubclass(type(block[0]), MultiTransformBlock):
                # These blocks are allowed dictionaries!
                assert len(block[0].ring_names) == len(block[1])
                for ring_name in block[0].ring_names:
                    assert ring_name in block[1]
                for param_name in block[1]:
                    ring_id = block[1][param_name]
                    if isinstance(ring_id, Ring):
                        all_names.append(ring_id)
                    else:
                        all_names.append(str(ring_id))
            else:
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
            if issubclass(type(block[0]), MultiTransformBlock):
                for param_name in block[1]:
                    block[0].rings[param_name] = self.rings[str(block[1][param_name])]
                threads.append(threading.Thread(
                    target=block[0].main))
            else:
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
def insert_zeros_evenly(input_data, number_zeros):
    """Insert zeros evenly in input_data.
        These zeros are distibuted evenly throughout
        the function, to help for binning of oddly
        shaped arrays.
    @param[in] input_data 1D array to contain zeros.
    @param[out] number_zeros Number of zeros that need
        to be added.
    @returns input_data with extra zeros"""
    insert_index = np.floor(
        np.arange(
            number_zeros,
            step=1.0)*float(input_data.size)/number_zeros)
    output_data = np.insert(
        input_data, insert_index,
        np.zeros(number_zeros))
    return output_data
class TransformBlock(object):
    """Defines the structure for a transform block"""
    def __init__(self, gulp_size=4096):
        super(TransformBlock, self).__init__()
        self.gulp_size = gulp_size
        self.out_gulp_size = None
        self.input_header = {}
        self.output_header = {}
        self.core = -1
    def load_settings(self, input_header):
        """Load in input header and set up block attributes
        @param[in] input_header Header sent from input ring"""
        self.output_header = input_header
    def iterate_ring_read(self, input_ring):
        """Iterate through one input ring"""
        input_ring.resize(self.gulp_size)
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            for span in sequence.read(self.gulp_size):
                yield span
    def iterate_ring_write(
            self, output_ring, sequence_name="",
            sequence_time_tag=0, sequence_nringlet=1):
        """Iterate through one output ring"""
        if self.out_gulp_size is None:
            self.out_gulp_size = self.gulp_size
        output_ring.resize(self.out_gulp_size)
        with output_ring.begin_writing() as oring:
            with oring.begin_sequence(
                sequence_name, sequence_time_tag,
                header=self.output_header,
                nringlet=sequence_nringlet) as oseq:
                with oseq.reserve(self.out_gulp_size) as span:
                    #TODO: Need to continuously spawn extra spans as in SourceBlock
                    yield span
    def ring_transfer(self, input_ring, output_ring):
        """Iterate through two rings span-by-span"""
        input_ring.resize(self.gulp_size)
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            if self.out_gulp_size is None:
                self.out_gulp_size = self.gulp_size
            output_ring.resize(self.out_gulp_size)
            with output_ring.begin_writing() as oring:
                with oring.begin_sequence(
                    sequence.name, sequence.time_tag,
                    header=self.output_header,
                    nringlet=sequence.nringlet) as oseq:
                    for ispan in sequence.read(self.gulp_size):
                        with oseq.reserve(ispan.size*self.out_gulp_size/self.gulp_size) as ospan:
                            yield ispan, ospan

class SourceBlock(object):
    """Defines the structure for a source block"""
    def __init__(self, gulp_size=4096):
        super(SourceBlock, self).__init__()
        self.gulp_size = gulp_size
        self.output_header = {}
        self.core = -1
    def iterate_ring_write(
            self, output_ring, sequence_name="",
            sequence_time_tag=0):
        """Iterate over output ring
        @param[in] output_ring Ring to write to
        @param[in] sequence_name Name to label sequence
        @param[in] sequence_time_tag Time tag to label sequence
        """
        output_ring.resize(self.gulp_size)
        with output_ring.begin_writing() as oring:
            with oring.begin_sequence(
                sequence_name, sequence_time_tag,
                header=self.output_header,
                nringlet=1) as oseq:
                while True:
                    with oseq.reserve(self.gulp_size) as span:
                        yield span
class SinkBlock(object):
    """Defines the structure for a sink block"""
    def __init__(self, gulp_size=4096):
        super(SinkBlock, self).__init__()
        self.gulp_size = gulp_size
        self.header = {}
        self.core = -1
    def load_settings(self, input_header):
        """Load in settings from input ring header"""
        self.header = json.loads(input_header.tostring())
    def iterate_ring_read(self, input_ring):
        """Iterate through one input ring
        @param[in] input_ring Ring to read through"""
        input_ring.resize(self.gulp_size)
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            for span in sequence.read(self.gulp_size):
                yield span
class MultiTransformBlock(object):
    """Defines functions and attributes for a block with multi input/output"""
    def __init__(self):
        """Set up dictionaries for holding ring settings"""
        super(MultiTransformBlock, self).__init__()
        self.rings = {}
        self.header = {}
        self.gulp_size = {}
    def flatten(self, *args):
        """Flatten a nested tuple/list of tuples/lists"""
        flattened_list = []
        for element in args:
            if isinstance(element, (tuple, list)):
                flattened_list.extend(self.flatten(*element))
            else:
                flattened_list.extend([element])
        return flattened_list
    def izip(self, *iterables):
        """Iterate through multpile iterators
        This differs from itertools in that this izip combines list generators
        into a single list generator"""
        iterators = [iter(iterable) for iterable in iterables]
        while True:
            next_set = [iterator.next() for iterator in iterators]
            yield self.flatten(*next_set)
    def load_settings(self):
        """Set by user to interpret input rings"""
        pass
    def read(self, *args):
        """Iterate over selection of input rings"""
        # list of sequences
        for sequences in self.izip(*[self.rings[ring_name].read(guarantee=True) \
                for ring_name in args]):
            # sequences is a tuple of all sequences
            for ring_name, sequence in self.izip(args, sequences):
                self.header[ring_name] = json.loads(sequence.header.tostring())
            self.load_settings()
            # resize all rings
            for ring_name in args:
                self.rings[ring_name].resize(self.gulp_size[ring_name])
            for spans in self.izip(*[sequence.read(self.gulp_size[ring_name]) \
                    for ring_name, sequence in self.izip(args, sequences)]):
                yield [span.data_view(np.float32)[0] for span in spans]
    def write(self, *args):
        """Iterate over selection of output rings"""
        # resize all rings
        for ring_name in args:
            self.rings[ring_name].resize(self.gulp_size[ring_name])
        # list of sequences
        #TODO: Change this code if someone gives a reasonable answer on 
        #http://stackoverflow.com/questions/38834827/multiple-with-statements-in-python-2-7-using-a-list-comprehension
        with nested(*[self.rings[ring_name].begin_writing() \
                for ring_name in args]) as out_rings:
            with nested(*[out_ring.begin_sequence(
                "",
                0,
                header=json.dumps(self.header[ring_name]),
                nringlet=1) \
                    for out_ring, ring_name in self.izip(out_rings, args)]) as out_sequences:
                while True:
                    with nested(*[out_sequence.reserve(self.gulp_size[ring_name]) \
                            for out_sequence, ring_name in self.izip(
                                out_sequences,
                                args)]) as out_spans:
                        #TODO: Other types and sizes
                        yield tuple([out_span.data_view(np.float32)[0] for out_span in out_spans])
class SplitterBlock(MultiTransformBlock):
    """Block which splits up a ring into two"""
    ring_names = {
        'in': 'Input to split. List of floats',
        'out_1': 'Gets first share of the ring. List of floats',
        'out_2': 'Gets second share of the ring. List of floats'}
    def __init__(self, sections):
        """@param[in] sections List of two lists - each list is a
                1D array of integers indicating sections of the ring
                to cut. Like numpy slicing indices."""
        super(SplitterBlock, self).__init__()
        assert len(sections) == 2
        self.sections = sections
        self.header['out_1'] = {}
        self.header['out_1']['dtype'] = str(np.float32)
        self.header['out_1']['nbit'] = 32
        #TODO: These sections should be applied to the incoming shapes
        self.header['out_1']['shape'] = sections[0]
        self.header['out_2'] = self.header['out_1']
        self.header['out_2']['shape'] = sections[1]
    def load_settings(self):
        """Set the gulp sizes appropriate to the input ring"""
        self.gulp_size['in'] = np.product(self.header['in']['shape'])*self.header['in']['nbit']//8
        self.gulp_size['out_1'] = self.gulp_size['in']*np.product(self.header['out_1']['shape']) \
                                    //np.product(self.header['in']['shape'])
        self.gulp_size['out_2'] = self.gulp_size['in']*np.product(self.header['out_2']['shape']) \
                                    //np.product(self.header['in']['shape'])
    def main(self):
        """Split the incoming ring into the outputs rings"""
        for inspan, outspan1, outspan2 in self.izip(
                self.read('in'),
                self.write('out_1', 'out_2')):
            outspan1[:] = inspan[self.sections[0]].ravel()
            outspan2[:] = inspan[self.sections[1]].ravel()
class MultiAddBlock(MultiTransformBlock):
    """Block which adds two input rings"""
    # name all rings with descriptions
    ring_names = {
        'in_1': 'First input to add. List of floats',
        'in_2': 'Second input to add. List of floats',
        'out_sum': 'Result of add. List of floats.'}
    def __init__(self, *args, **kwargs):
        #can hard code these, or calculate them during load_settings
        super(MultiAddBlock, self).__init__()
        self.gulp_size['in_1'] = 2*4
        self.gulp_size['in_2'] = 2*4
        self.gulp_size['out_sum'] = 2*4
        self.header['out_sum'] = {}
        self.header['out_sum']['dtype'] = str(np.float32)
        self.header['out_sum']['nbit'] = '32'
        self.header['out_sum']['shape'] = (2,)
    def main(self):
        """Iterate through the inputs, and add them to the output"""
        for inspan1, inspan2, outspan in self.izip(
                self.read('in_1', 'in_2'),
                self.write('out_sum')):
            outspan[:] = inspan1 + inspan2
class TestingBlock(SourceBlock):
    """Block for debugging purposes.
    Allows you to pass arbitrary N-dimensional arrays in initialization,
    which will be outputted into a ring buffer"""
    def __init__(self, test_array, complex_numbers=False):
        """Figure out data settings from the test array.
        @param[in] test_array A list or numpy array containing test data"""
        super(TestingBlock, self).__init__()
        if isinstance(test_array, np.ndarray):
            if test_array.dtype == np.complex64:
                complex_numbers = True
        if complex_numbers:
            self.test_array = np.array(test_array).astype(np.complex64)
            header = {
                'nbit':64,
                'dtype':str(np.complex64),
                'shape':self.test_array.shape}
            self.dtype = np.complex64
        else:
            self.test_array = np.array(test_array).astype(np.float32)
            header = {
                'nbit':32,
                'dtype':str(np.float32),
                'shape':self.test_array.shape}
            self.dtype = np.float32
        self.output_header = json.dumps(header)
    def main(self, output_ring):
        """Put the test array onto the output ring
        @param[in] output_ring Holds the flattend test array in a single span"""
        self.gulp_size = self.test_array.nbytes
        for ospan in self.iterate_ring_write(output_ring):
            ospan.data_view(self.dtype)[0][:] = self.test_array.ravel()
            break
class WriteHeaderBlock(SinkBlock):
    """Prints the header of a ring to a file"""
    def __init__(self, filename):
        """@param[in] test_array A list or numpy array containing test data"""
        super(WriteHeaderBlock, self).__init__()
        self.filename = filename
    def load_settings(self, input_header):
        """Load the header from json
        @param[in] input_header The header from the ring"""
        write_file = open(self.filename, 'w')
        write_file.write(str(json.loads(input_header.tostring())))
    def main(self, input_ring):
        """Put the header into the file
        @param[in] input_ring Contains the header in question"""
        self.gulp_size = 1
        span_dummy_generator = self.iterate_ring_read(input_ring)
        span_dummy_generator.next()
class FFTBlock(TransformBlock):
    """Performs complex to complex 1D FFT on input ring data"""
    def __init__(self, gulp_size):
        super(FFTBlock, self).__init__()
        self.nbit = 8
        self.dtype = np.uint8
        self.shape = (1, 1)
    def load_settings(self, input_header):
        header = json.loads(input_header.tostring())
        self.nbit = header['nbit']
        self.dtype = np.dtype(header['dtype'].split()[1].split(".")[1].split("'")[0]).type
        if 'frame_shape' in header:
            self.shape = header['frame_shape']
        header['nbit'] = 64
        header['dtype'] = str(np.complex64)
        self.output_header = json.dumps(header)
    def main(self, input_rings, output_rings):
        """
        @param[in] input_rings First ring in this list will be used for
            data
        @param[out] output_rings First ring in this list will be used for
            data output."""
        data_accumulate = None
        for ispan in self.iterate_ring_read(input_rings[0]):
            if self.nbit < 8:
                unpacked_data = unpack(ispan.data_view(self.dtype), self.nbit)
            else:
                unpacked_data = ispan.data_view(self.dtype)
            if data_accumulate is not None:
                data_accumulate = np.concatenate((data_accumulate, unpacked_data[0]))
            else:
                data_accumulate = unpacked_data[0]
        if self.shape != [1, 1]:
            data_accumulate = np.reshape(data_accumulate, (self.shape[0], -1))
        data_accumulate = data_accumulate.astype(np.complex64)
        self.out_gulp_size = data_accumulate.nbytes
        outspan_generator = self.iterate_ring_write(output_rings[0])
        ospan = outspan_generator.next()
        result = np.fft.fft(data_accumulate)
        ospan.data_view(np.complex64)[0] = result.ravel()
class IFFTBlock(TransformBlock):
    """Performs complex to complex 1D IFFT on input ring data"""
    def __init__(self, gulp_size):
        super(IFFTBlock, self).__init__()
        self.gulp_size = gulp_size
        self.nbit = 8
        self.dtype = np.uint8
    def load_settings(self, input_header):
        header = json.loads(input_header.tostring())
        self.nbit = header['nbit']
        self.dtype = np.dtype(header['dtype'].split()[1].split(".")[1].split("'")[0]).type
        header['nbit'] = 64
        header['dtype'] = str(np.complex64)
        self.output_header = json.dumps(header)
    def main(self, input_rings, output_rings):
        """
        @param[in] input_rings First ring in this list will be used for
            data input.
        @param[out] output_rings First ring in this list will be used for 
            data output."""
        data_accumulate = None
        for ispan in self.iterate_ring_read(input_rings[0]):
            if self.nbit < 8:
                unpacked_data = unpack(ispan.data_view(self.dtype), self.nbit)
            else:
                unpacked_data = ispan.data_view(self.dtype)
            if data_accumulate is not None:
                data_accumulate = np.concatenate((data_accumulate, unpacked_data[0]))
            else:
                data_accumulate = unpacked_data[0]
        data_accumulate = data_accumulate.astype(np.complex64)
        self.out_gulp_size = data_accumulate.nbytes
        outspan_generator = self.iterate_ring_write(output_rings[0])
        ospan = outspan_generator.next()
        result = np.fft.ifft(data_accumulate)
        ospan.data_view(np.complex64)[0][:] = result[:]
class IFFT2Block(TransformBlock):
    """Performs complex to complex 2D IFFT on input ring data"""
    def __init__(self, gulp_size=1048576):
        super(IFFT2Block, self).__init__(gulp_size=gulp_size)
    def load_settings(self, input_header):
        header = json.loads(input_header.tostring())
        self.frame_shape = header['frame_shape']
        output_header = {}
        output_header['nbit'] = 64
        output_header['dtype'] = str(np.complex64)
        output_header['shape'] = self.frame_shape
        self.output_header = json.dumps(output_header)
    def main(self, input_rings, output_rings):
        """
        @param[in] input_rings First ring in this list will be used for
            data input.
        @param[out] output_rings First ring in this list will be used for 
            data output."""
        input_data = np.array([])
        for ispan in self.iterate_ring_read(input_rings[0]):
            input_data = np.append(input_data, ispan.data_view(np.complex64)[0][:])
        input_data = input_data.reshape(self.frame_shape)
        self.out_gulp_size = input_data.size*8
        out_span_generator = self.iterate_ring_write(output_rings[0])
        ospan = out_span_generator.next()
        ospan.data_view(np.complex64)[0][:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_data))).ravel()
class WriteAsciiBlock(SinkBlock):
    """Copies input ring's data into ascii format
        in a text file."""
    def __init__(self, filename, gulp_size=1048576, speed_read_factor=1):
        """@param[in] filename Name of file to write ascii to
        @param[out] gulp_size How much of the file to write at once"""
        super(WriteAsciiBlock, self).__init__()
        self.filename = filename
        self.gulp_size = gulp_size
        self.nbit = 8
        self.dtype = np.uint8
        self.speed_read_factor = speed_read_factor
        open(self.filename, "w").close() ## erase file
    def load_settings(self, input_header):
        header_dict = json.loads(input_header.tostring())
        self.nbit = header_dict['nbit']
        if 'shape' in header_dict:
            self.gulp_size = self.nbit*np.product(header_dict['shape'])/8*self.speed_read_factor
        self.dtype = np.dtype(header_dict['dtype'].split()[1].split(".")[1].split("'")[0]).type
    def main(self, input_ring):
        """Initiate the writing to filename
        @param[in] input_rings First ring in this list will be used for
            data
        @param[out] output_rings This list of rings won't be used."""
        span_generator = self.iterate_ring_read(input_ring)
        data_accumulate = None
        for span in span_generator:
            if self.nbit < 8:
                unpacked_data = unpack(span.data_view(self.dtype), self.nbit)
            else:
                if self.dtype == np.complex64:
                    unpacked_data = span.data_view(self.dtype).view(np.float32)
                else:
                    unpacked_data = span.data_view(self.dtype)
            if data_accumulate is not None:
                data_accumulate = np.concatenate((data_accumulate, unpacked_data.ravel()), axis=0).ravel()
            else:
                data_accumulate = unpacked_data.ravel()
        x = data_accumulate.reshape((1, -1))
        text_file = open(self.filename, 'w')
        if self.dtype == np.complex64:
            np.savetxt(text_file, x.view(np.float32))
        else:
            np.savetxt(text_file, x)
class CopyBlock(TransformBlock):
    """Copies input ring's data to the output ring"""
    def __init__(self, gulp_size=1048576):
        super(CopyBlock, self).__init__(gulp_size=gulp_size)
    def main(self, input_rings, output_rings):
        """Iterate through first ring, copying to each output ring"""
        input_ring = input_rings[0]
        for output_ring in output_rings:
            for ispan, ospan in self.ring_transfer(input_ring, output_ring):
                bifrost.memory.memcpy2D(ospan.data, ispan.data)
class SigprocReadBlock(SourceBlock):
    """This block reads in a sigproc filterbank
    (.fil) file into a ring buffer"""
    def __init__(
            self, filename,
            gulp_nframe=4096, core=-1):
        """
        @param[in] filename filterbank file to read
        @param[in] gulp_nframe Time samples to read
            in at a time
        @param[in] core Which CPU core to bind to (-1) is
            any
        """
        super(SigprocReadBlock, self).__init__()
        self.filename = filename
        self.gulp_nframe = gulp_nframe
        self.core = core
    def main(self, output_ring):
        """Read in the sigproc file to output_ring
        @param[in] output_ring Ring to write to"""
        with SigprocFile().open(self.filename, 'rb') as ifile:
            ifile.read_header()
            ohdr = {}
            ohdr['frame_shape'] = (ifile.nchans, ifile.nifs)
            ohdr['frame_size'] = ifile.nchans*ifile.nifs
            ohdr['frame_nbyte'] = ifile.nchans*ifile.nifs*ifile.nbits/8
            ohdr['frame_axes'] = ('pol', 'chan')
            ohdr['ringlet_shape'] = (1,)
            ohdr['ringlet_axes'] = ()
            ohdr['dtype'] = str(ifile.dtype)
            ohdr['nbit'] = ifile.nbits
            ohdr['tsamp'] = float(ifile.header['tsamp'])
            ohdr['tstart'] = float(ifile.header['tstart'])
            ohdr['fch1'] = float(ifile.header['fch1'])
            ohdr['foff'] = float(ifile.header['foff'])
            self.output_header = json.dumps(ohdr)
            self.gulp_size = self.gulp_nframe*ifile.nchans*ifile.nifs*ifile.nbits/8
            out_span_generator = self.iterate_ring_write(output_ring)
            for span in out_span_generator:
                output_size = ifile.file_object.readinto(span.data.data)
                span.commit(output_size)
                if output_size < self.gulp_size:
                    break
class KurtosisBlock(TransformBlock):
    """This block performs spectral kurtosis and cleaning
        on sigproc-formatted data in rings"""
    def __init__(self, gulp_size=1048576, core=-1):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] output_ring Ring will contain a 1d
            timeseries that will be cleaned of RFI
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        """
        super(KurtosisBlock, self).__init__()
        self.gulp_size = gulp_size
        self.core = core
        self.output_header = {}
        self.settings = {}
        self.nchan = 1
        self.dtype = np.uint8
    def load_settings(self, input_header):
        self.output_header = input_header
        self.settings = json.loads(input_header.tostring())
        self.nchan = self.settings["frame_shape"][0]
        dtype_str = self.settings["dtype"].split()[1].split(".")[1].split("'")[0]
        self.dtype = np.dtype(dtype_str)
    def main(self, input_rings, output_rings):
        """Calls a kurtosis algorithm and uses the result
            to clean the input data of RFI, and move it to the
            output ring."""
        expected_v2 = 0.5
        for ispan, ospan in self.ring_transfer(input_rings[0], output_rings[0]):
            nsample = ispan.size/self.nchan/(self.settings['nbit']/8)
            # Raw data -> power array of the right type
            power = ispan.data.reshape(
                nsample,
                self.nchan*self.settings['nbit']/8).view(self.dtype)
            # Following section 3.1 of the Nita paper.
            # the sample is a power value in a frequency bin from an FFT,
            # i.e. the beamformer values in a channel
            number_samples = power.shape[0]
            bad_channels = []
            for chan in range(self.nchan):
                nita_s1 = np.sum(power[:, chan])
                nita_s2 = np.sum(power[:, chan]**2)
                # equation 21
                nita_v2 = (number_samples/(number_samples-1))*\
                    (number_samples*nita_s2/(nita_s1**2)-1)
                if abs(expected_v2-nita_v2) > 0.1:
                    bad_channels.append(chan)
            flag_power = power.copy()
            for chan in range(self.nchan):
                if chan in bad_channels:
                    flag_power[:, chan] = 0    # set bad channel to zero
            ospan.data[0][:] = flag_power.view(dtype=np.uint8).ravel()

class DedisperseBlock(object):
    """This block calculates the dedispersion of
        sigproc-formatted data in a ring, and tags
        it in the headers"""
    def __init__(
            self, ring, core=-1,
            gulp_size=4096):
        """
        @param[in] ring Ring containing a 1d
            timeseries with a source affected
            by DM
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        @param[in] gulp_size How many bytes of the ring to
            read at once.
        """
        self.ring = ring
        self.core = core
        self.gulp_size = gulp_size
    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        self.dedisperse()
    def dedisperse(self):
        """Dedisperse on input ring, tagging the output ring.
        This dedispersion algorithm simply adjusts the start
        time for each channel in the header."""
        pass

class FoldBlock(TransformBlock):
    """This block folds a signal into a histogram"""
    def __init__(
            self, bins, period=1e-3,
            gulp_size=4096*256, dispersion_measure=0,
            core=-1):
        """
        @param[in] bins The total number of bins to fold into
        @param[in] period Period to fold over (s)
        @param[in] gulp_size How many bytes of the ring to
            read at once.
        @param[in] dispersion_measure DM of the desired
            source (pc cm^-3)
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        """
        super(FoldBlock, self).__init__()
        self.bins = bins
        self.gulp_size = gulp_size
        self.period = period
        self.dispersion_measure = dispersion_measure 
        self.core = core
        self.data_settings = {}
    def calculate_bin_indices(
            self, tstart, tsamp, data_size):
        """Calculate the bin that each time sample should be
            added to
        @param[in] tstart Time of the first element (s)
        @param[in] tsamp Difference between the times of
            consecutive elements (s)
        @param[in] data_size Number of elements
        @return Which bin each sample is folded into
        """
        arrival_time = tstart+tsamp*np.arange(data_size)
        phase = np.fmod(arrival_time, self.period)
        return np.floor(phase/self.period*self.bins).astype(int)
    def calculate_delay(self, frequency, reference_frequency):
        """Calculate the time delay because of frequency dispersion
        @param[in] frequency The current channel's frequency(MHz)
        @param[in] reference_frequency The frequency of the 
            channel we will hold at zero time delay(MHz)"""
        frequency_factor = \
            np.power(reference_frequency/1000, -2) -\
            np.power(frequency/1000, -2)
        return 4.15e-3*self.dispersion_measure*frequency_factor
    def load_settings(self, input_header):
        self.data_settings = json.loads(
            "".join(
                [chr(item) for item in input_header]))
        self.output_header = json.dumps({'nbit':32, 'dtype': str(np.float32)})
    def main(self, input_rings, output_rings):
        """Generate a histogram from the input ring data
        @param[in] input_rings List with first ring containing
            data of interest. Must terminate before histogram
            is generated.
        @param[out] output_rings First ring in this list
            will contain the output histogram"""
        histogram = np.reshape(
            np.zeros(self.bins).astype(np.float32),
            (1, self.bins))
        tstart = None
        for span in self.iterate_ring_read(input_rings[0]):
            nchans = self.data_settings['frame_shape'][0]
            if tstart is None:
                tstart = self.data_settings['tstart']
            frequency = self.data_settings['fch1']
            for chan in range(nchans):
                modified_tstart = tstart - self.calculate_delay(
                    frequency,
                    self.data_settings['fch1'])
                frequency -= self.data_settings['foff']
                sort_indices = np.argsort(
                    self.calculate_bin_indices(
                        modified_tstart, self.data_settings['tsamp'],
                        span.data.shape[1]/nchans))
                sorted_data = span.data[0][chan::nchans][sort_indices]
                extra_elements = np.round(self.bins*(1-np.modf(
                    float(span.data.shape[1]/nchans)/self.bins)[0])).astype(int)
                sorted_data = insert_zeros_evenly(sorted_data, extra_elements)
                histogram += np.sum(
                    sorted_data.reshape(self.bins, -1), 1).astype(np.float32)
            tstart += self.data_settings['tsamp']*\
                self.gulp_size*8/self.data_settings['nbit']/nchans
        self.out_gulp_size = self.bins*4
        out_span_generator = self.iterate_ring_write(output_rings[0])
        out_span = out_span_generator.next()
        bifrost.memory.memcpy(
            out_span.data_view(dtype=np.float32),
            histogram)

class WaterfallBlock(object):
    """This block creates a waterfall block
        based on the data in a ring, and stores it
        in the headers"""
    def __init__(
            self, ring, imagename,
            core=-1, gulp_nframe=4096):
        """
        @param[in] ring Ring containing a multichannel
            timeseries
        @param[in] imagename Filename to store the
            waterfall image
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        @param[in] gulp_size How many bytes of the ring to
            read at once.
        """
        self.ring = ring
        self.imagename = imagename
        self.core = core
        self.gulp_nframe = gulp_nframe
        self.header = {}
    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        waterfall_matrix = self.generate_waterfall_matrix()
        self.save_waterfall_plot(waterfall_matrix)
    def save_waterfall_plot(self, waterfall_matrix):
        """Save an image of the waterfall plot using
            thread-safe backend for pyplot, and labelling
            the plot using the header information from the
            ring
        @param[in] waterfall_matrix x axis is frequency and 
            y axis is time. Values should be power.
            """
        plt.ioff()
        print "Interactive mode off"
        print waterfall_matrix.shape
        fig = pylab.figure()
        ax = fig.gca()
        header = self.header
        ax.set_xticks(
            np.arange(0, 1.33, 0.33)*waterfall_matrix.shape[1])
        ax.set_xticklabels(
            header['fch1']-np.arange(0,4)*header['foff'])
        ax.set_xlabel("Frequency [MHz]")
        ax.set_yticks(
            np.arange(0, 1.125, 0.125)*waterfall_matrix.shape[0])
        ax.set_yticklabels(
            header['tstart']+header['tsamp']*np.arange(0, 1.125, 0.125)*waterfall_matrix.shape[0])
        ax.set_ylabel("Time (s)")
        plt.pcolormesh(
            waterfall_matrix, axes=ax, figure=fig)
        fig.autofmt_xdate()
        fig.savefig(
            self.imagename, bbox_inches='tight')
        plt.close(fig)
    def generate_waterfall_matrix(self):
        """Create a matrix for a waterfall image
            based on the ring's data"""
        waterfall_matrix = None
        self.ring.resize(self.gulp_nframe)
        # Generate a waterfall matrix:
        for sequence in self.ring.read(guarantee=True):
            ## Get the sequence's header as a dictionary
            self.header = json.loads(
                "".join(
                    [chr(item) for item in sequence.header]))
            tstart = self.header['tstart']
            tsamp = self.header['tsamp']
            nchans = self.header['frame_shape'][0]
            gulp_size = self.gulp_nframe*nchans*self.header['nbit']
            waterfall_matrix = np.zeros(shape=(0, nchans))
            print tstart, tsamp, nchans
            for span in sequence.read(gulp_size):
                array_size = span.data.shape[1]/nchans
                frequency = self.header['fch1']
                try: 
                    curr_data = np.reshape(
                        span.data,(-1, nchans))
                    waterfall_matrix = np.concatenate(
                        (waterfall_matrix, curr_data), 0)
                except:
                    print "Bad shape for waterfall"
        return waterfall_matrix
class FakeVisBlock(SourceBlock):
    """Read a formatted file for fake visibility data"""

    def __init__(self, filename, num_stands):
        super(FakeVisBlock, self).__init__()
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

        self.gulp_size = uvw_data.nbytes
        for span in self.iterate_ring_write(output_ring):
            span.data_view(np.float32)[0][:] = uvw_data.ravel()
            break

class NearestNeighborGriddingBlock(TransformBlock):
    """Perform a nearest neighbor gridding of visibility data"""
    def __init__(self, shape):
        super(NearestNeighborGriddingBlock, self).__init__()
        self.shape = shape
    def _normalize_coordinates(self, coordinates):
        """Turn float coordinates from another grid into one of self.shape
        @param[in] coordinates The coordinates to normalize (numpy.ndarray)"""
        shifted = coordinates-np.min(coordinates)
        amplitude_unity = shifted/np.max(shifted)
        normalized = amplitude_unity*self.shape[0]-0.5
        return normalized.astype(int)
    def main(self, input_rings, output_rings):
        """Compute a nearest neighbor gridding on the input data
        @param[in] input_rings The first ring will be accumulated into
            a grid. Expected data as floats: [[u,v,re,im],[u,..],..]
        @param[out] output_rings Once accumulated in a local variable,
            will be output on first output ring"""
        grid = np.zeros(self.shape, dtype=np.complex64)
        u_coords = np.array([])
        v_coords = np.array([])
        real_visibilities = np.array([])
        complex_visibilities = np.array([])
        self.gulp_size = 8**6
        for in_span in self.iterate_ring_read(input_rings[0]):
            u_coords = np.append(u_coords, in_span.data_view(np.float32)[0][0::4])
            v_coords = np.append(v_coords, in_span.data_view(np.float32)[0][1::4])
            real_visibilities = np.append(real_visibilities, in_span.data_view(np.float32)[0][2::4])
            complex_visibilities = np.append(
                complex_visibilities,
                in_span.data_view(np.float32)[0][3::4])
        grid_u_coords = self._normalize_coordinates(u_coords)
        grid_v_coords = self._normalize_coordinates(v_coords)
        for index_visibility in range(real_visibilities.size): #pylint: disable=maybe-no-member
            grid[
                grid_u_coords[index_visibility],
                grid_v_coords[index_visibility]] += \
                    real_visibilities[index_visibility] + \
                    1j*complex_visibilities[index_visibility]
        self.out_gulp_size = grid.nbytes
        self.output_header = json.dumps(
            {'dtype':str(np.complex64),
             'nbit':64,
             'frame_shape': self.shape,
             'shape': self.shape})
        out_span_generator = self.iterate_ring_write(output_rings[0])
        out_span = out_span_generator.next()
        out_span.data_view(np.complex64)[0][:] = grid.ravel()

class GainSolveBlock(TransformBlock):
    """Optimize the Jones matrices to produce the sky model."""
    def __init__(self, flags = [], max_iterations = 20):
        super(GainSolveBlock, self).__init__()
        self.flags = np.array(flags)
        self.shapes = []
        self.max_iterations = max_iterations
    def load_settings(self, input_header):
        self.shapes.append(json.loads(input_header.tostring())['shape'])
        self.gulp_size = np.product(self.shapes[-1])*8
        self.output_header = input_header
    def main(self, input_rings, output_rings):
        data_span_generator = self.iterate_ring_read(input_rings[0])
        model_span_generator = self.iterate_ring_read(input_rings[1])
        jones_span_generator = self.iterate_ring_read(input_rings[2])
        data = data_span_generator.next()
        data = data.data_view(np.complex64).reshape(self.shapes[0])
        model = model_span_generator.next()
        model = model.data_view(np.complex64).reshape(self.shapes[1])
        jones = jones_span_generator.next()
        jones = jones.data_view(np.complex64).reshape(self.shapes[2])
        assert model.shape == data.shape
        assert jones.shape[2] == model.shape[1]
        gpu_data = GPUArray(data.shape, np.complex64)
        gpu_model = GPUArray(model.shape, np.complex64)
        gpu_jones = GPUArray(jones.shape, np.complex64)
        gpu_flags = GPUArray(self.flags.shape, np.int8)
        gpu_data.set(data)
        gpu_model.set(model)
        gpu_jones.set(jones)
        gpu_flags.set(self.flags)
        array_jones = gpu_jones.as_BFarray(100)
        num_unconverged_type = ctypes.POINTER(ctypes.c_int)
        num = ctypes.c_int(1)
        num_unconverged = ctypes.cast(ctypes.addressof(num), num_unconverged_type)
        _bf.SolveGains(
            gpu_data.as_BFconstarray(100), 
            gpu_model.as_BFconstarray(100), 
            array_jones,
            gpu_flags.as_BFarray(100),
            True, 7.0, 0.000001, self.max_iterations, num_unconverged)
            #True, 10.0, 0.000001, self.max_iterations, num_unconverged)
            #Best so far^
        gpu_jones.buffer = array_jones.data
        jones = gpu_jones.get()
        for i in range(max(jones.shape)):
            try:
                jones[0, :, i, :] = np.linalg.inv(jones[0, :, i, :])
            except:
                pass
        gpu_jones.set(jones)
        self.out_gulp_size = jones.nbytes
        out_jones_generator = self.iterate_ring_write(output_rings[1])
        out_jones = out_jones_generator.next()
        out_jones.data_view(np.complex64)[0][:] = gpu_jones.get().ravel()

        """ Y = G X G^
BFstatus bfApplyGainsArray(BFconstarray X, // Observed data. [nchan,nstand^,npol^,nstand,npol] cf32
                           BFconstarray G, // Jones matrices. [nchan,pol^,nstand,npol] cf32
                           BFarray      V, // Observed data. [nchan,nstand^,npol^,nstand,npol] cf32
                           BFarray      flags);  // [nchan,nstand] i8
                              """
        gpu_output_image = GPUArray(data.shape, np.complex64)
        array_image = gpu_output_image.as_BFarray(100)
        _bf.ApplyGainsArray(
            gpu_data.as_BFconstarray(100),
            gpu_jones.as_BFconstarray(100),
            array_image,
            gpu_flags.as_BFarray(100))
        gpu_output_image.buffer = array_image.data
        resultant_image = gpu_output_image.get()
        self.out_gulp_size = resultant_image.nbytes
        out_model_generator = self.iterate_ring_write(output_rings[0])
        out_model = out_model_generator.next()
        out_model.data_view(np.complex64)[0][:] = gpu_output_image.get().ravel()
