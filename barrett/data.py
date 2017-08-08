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
            col = tuple(f.keys())[0]
            self.chunksize = f[col].chunks[0]
            self.n = f[col].shape[0]

        f.close()


    def create_column(self, name):
        h5 = h5py.File(self.h5file, 'r+')

        h5.create_dataset(name,
                          shape=(self.n,),
                          maxshape=(None,),
                          dtype=np.float64,
                          chunks=(self.chunksize,),
                          compression='gzip',
                          shuffle=True)

        h5.close()


    def apply(self, d_name, func, *args):
        h5 = h5py.File(self.h5file, 'r+')

        if d_name not in h5:
            self.create_column(d_name)

        d = h5[d_name]

        vs = [h5[i] for i in args]

        s = self.chunksize

        for i in range(0, self.n, s):
            val = func(*[v[i:i+s] for v in vs])
            d[i:i+s] = val

        h5.close()


    def transform_column(self, column, func):
        self.apply(column, func, column)


    def append(self, other):
        h5 = h5py.File(self.h5file, 'r+')

        other_h5 = h5py.File(self.h5file, 'r')

        sym_diff_keys = set(h5.keys()) ^ set(other_h5.keys())
        if len(sym_diff_keys) != 0:
            print('Not compatible datasets. Symmetric difference: %s\n' % sym_diff_keys)
            return

        col = tuple(other_h5.keys())[0]
        other_n = other_h5[col].shape[0]

        starts = list(range(0, other_n, self.chunksize))
        ends   = starts[1:] + [other_n]

        for s, e in zip(starts, ends):
            for k in other_h5.keys():
                other_nrows = e-s

                nrows = h5[k].shape[0]

                h5[k].resize(nrows+other_nrows, axis=0)

                h5[k][nrows:] = other_h5[k][s:e]

        col = tuple(h5.keys())[0]
        self.n = h5[col].shape[0]

        h5.close()
        other_h5.close()


    def bestfit(self):

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

        p = {}
        for k in h5.keys():
            p[k] = h5[k][index]
        return p


    def posterior_mean(self, post_col='mult', columns=None):
        h5 = h5py.File(self.h5file, 'r')
        s = self.chunksize

        p = {}
        for k in h5.keys():
            p[k] = 0.0

        for i in range(0, self.n, s):
            for k in h5.keys():
                p[k] += np.sum(h5[post_col][i:i+s]*h5[k][i:i+s])
        return p


    def log(self, column):
        h5 = h5py.File(self.h5file, 'r+')

        self.apply('log(%s)' % column, np.log10, column)

        h5.close()
