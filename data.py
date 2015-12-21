import h5py
import numpy as np
import os.path

class Chain:

    def __init__(self, h5file, chunksize=10000):
        self.h5file = h5file

        if not os.path.isfile(self.h5file):
            f = h5py.File(self.h5file, 'w')
            self.chunksize = chunksize
            self.n = 0

        else:
            f = h5py.File(self.h5file, 'r')
            self.chunksize = f['mult'].chunks[0]
            self.n = f['mult'].shape[0]

        f.close()


    def create_column(self, name, unit):
        h5 = h5py.File(self.h5file, 'r+')

        dset = h5.create_dataset(name,
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


    def apply(self, d_name, unit, func, *args):
        h5 = h5py.File(self.h5file, 'r+')

        if d_name not in h5:
            if unit == None:
                unit = ''
            self.create_column(d_name, unit)

        d = h5[d_name]

        vs = [h5[i] for i in args]

        s = self.chunksize

        for i in range(0, self.n, s):
            val = func(*[v[i:i+s] for v in vs])
            d[i:i+s] = val

        # Update the column units if necessary.
        unit = unit.encode('utf8')
        if d.attrs.get('unit') != unit:
            d.attrs['unit'] = unit

        h5.close()


    def transform_column(self, column, unit, func):
        self.apply(column, unit, func, column)


    def append(self, other):
        h5 = h5py.File(self.h5file, 'r+')

        other_h5 = h5py.File(self.h5file, 'r')

        sym_diff_keys = set(h5.keys()) ^ set(other_h5.keys())
        if len(sym_diff_keys) != 0:
            print('Not compatible datasets. Symmetric difference: %s\n' % sym_diff_keys)
            return

        other_n = other_h5['mult'].shape[0]

        starts = list(range(0, other_n, self.chunksize))
        ends   = starts[1:] + [other_n]

        for s, e in zip(starts, ends):
            for k in other_h5.keys():
                other_nrows = e-s

                nrows = h5[k].shape[0]

                h5[k].resize(nrows+other_nrows, axis=0)

                h5[k][nrows:] = other_h5[k][s:e]

        self.n = h5['mult'].shape[0]

        h5.close()
        other_h5.close()


    def bestfit(self, columns=None):

        h5 = h5py.File(self.h5file, 'r')
        chi2 = h5['-2lnL']
        s = self.chunksize

        minimum, index = (chi2[0], 0)
        for i in range(0, self.n, s):
            chunk_index = np.argmin(chi2)
            chunk_minimum = chi2[chunk_index]

            if chunk_minimum < minimum:
                minimum = chunk_minimum
                index = chunk_index

        if columns == None:
            p = {}
            for k in h5.keys():
                p[k] = h5[k][index]
        else:
            p = []
            for c in columns:
                p.append(h5[c][index])
            p = np.array(p)

        return p


    def get_unit(self, column):
        h5 = h5py.File(self.h5file, 'r')
        unit = h5[column].attrs.get('unit').decode('utf8')
        h5.close()

        return unit


    def log(self, column):
        h5 = h5py.File(self.h5file, 'r+')

        unit = 'log(%s)' % h5[column].attrs.get('unit').decode('utf8')
        self.apply('log(%s)' % column, unit, np.log10, column)

        h5.close()
