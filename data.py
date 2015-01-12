import h5py
import numpy as np
import os.path

class Chain:

    def __init__(self, h5file, chunksize=10000):
        self.h5file = h5file

        if not os.path.isfile(self.h5file):
            f = h5py.File(self.h5file, 'x')
            self.chunksize = chunksize
            self.n = 0
        else:
            f = h5py.File(self.h5file, 'r')
            self.chunksize = f['mult'].chunks[0]
            self.n = f['mult'].shape[0]

        f.close()


    def create_column(name, unit):
        h5 = h5py.File(self.h5file, 'r+')

        dset = h5.create_dataset(h5,
                                 shape=(self.n,),
                                 maxshape=(None,),
                                 dtype=np.float64,
                                 chunks=(self.chunksize,),
                                 compression='gzip',
                                 shuffle=True)

        if unit == None:
            unit = ''

        dset.attrs.create('unit', unit.encode('utf8'))

        h5.close()


    def apply(self, d_name, unit=None, func, *args):
        h5 = h5py.File(self.h5file, 'r+')

        if d_name not in h5:
            if unit == None:
                unit = ''
            self.create_column(d_name, unit)

        d = h5[d_name]

        vs = [h5[i] for i in args]

        for i in range(0, self.n, self.s):
            val = func(*[v[i:i+s] for v in vs])
            d[i:i+s] = val

        # Update the column units if necessary.
        if unit != None:
            oldunit = d.attrs.get('unit')
            newunit = unit.encode('utf8')

            if newunit != oldunit:
                d.attrs.set('unit', newunit) 

        h5.close()


    def transform_column(self, column, unit=None, func):
        self.apply(column, unit, func, column)
