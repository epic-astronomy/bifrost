""" 
blockgen.py

This file contains some macros for quick generation of blocks """

import numpy as np 
import time
import bifrost as bf
import bifrost.pipeline as bfp
import bifrost.blocks as blocks

class BlockgenSource(bfp.SourceBlock):
    def __init__(self, generator):
        """ Load the generator into this object

        Args:

            generator: A generator object (a function with yeild statments)
        """
        super(BlockgenSource, self).__init__(["test"], gulp_nframe=1)
        self.generator = generator
    def create_reader(self, sourcename):
        """ Start the generator, use it as the reader for SourceBlock """
        return self.generator()
    def on_sequence(self, reader, sourcename):
        """ Create output settings for the sequence based on the generator """
        ohdr = {
                'name': 'test',
                '_tensor': {
                    'shape': [-1, 100],
                    'dtype': 'f32',
                    'labels': ['one', 'two'],
                    'scales': [None, None],
                    'units': [None, None]
                    }
        }
        ohdr['time_tag'] = 0
        return [ohdr]
    def on_data(self, reader, ospans):
        """ Output the next element of the generator """
        ospan = ospans[0]
        odata = ospan.data
        try:
            odata[0, :] = next(reader)[:]
            return [1]
        except StopIteration:
            return [0]

def source(generator):
    """ Generate a source block based on a generator object
    
    Args:
        
        generator: A generator object (a function with yeild statments)

    Returns:

        BlockgenSource: A block object which uses the generator to
                        output data.
    """
    return BlockgenSource(generator)
