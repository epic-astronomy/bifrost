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
        super(BlockgenSource, self).__init__(["test"], gulp_nframe=1)
        self.generator = generator
    def create_reader(self, sourcename):
        return self.generator()
    def on_sequence(self, reader, sourcename):
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
        ospan = ospans[0]
        odata = ospan.data
        try:
            odata[0, :] = next(reader)[:]
            return [1]
        except:
            return [0]

def source(generator):
    return BlockgenSource(generator)
