import h5py
import numpy as np
import scipy as sp
import itertools

def threenum(h5file, var, post_col='mult'):
    """ Calculates the three number summary for a variable.

    The three number summary is the minimum, maximum and the mean
    of the data. Traditionally one would summerise data with the
    five number summary: max, min, 1st, 2nd (median), 3rd quartile.
    But quantiles are hard to calculate without sorting the data
    which hard to do out-of-core.
    """
    f = h5py.File(h5file, 'r')
    d = f[var]
    w = f[post_col]
    s = d.chunks[0]

    n = d.shape[0]
    maxval = -np.abs(d[0])
    minval = np.abs(d[0])
    total = 0
    wsum = 0

    for x in range(0, n, s):

        aN = ~np.logical_or(np.isnan(d[x:x+s]), np.isinf(d[x:x+s]))

        d_c = d[x:x+s][aN]
        w_c = w[x:x+s][aN]

        chunk_max = np.max(d_c)
        chunk_min = np.min(d_c)

        maxval = chunk_max if chunk_max > maxval else maxval
        minval = chunk_min if chunk_min < minval else minval

        total += np.sum(w_c*d_c)
        wsum  += np.sum(w_c)
    f.close()

    mean = total/float(wsum)

    return (minval, maxval, mean)


def filechunk(f, chunksize):
    """Iterator that allow for piecemeal processing of a file."""
    while True:
        chunk = tuple(itertools.islice(f, chunksize))
        if not chunk:
            return
        yield np.loadtxt(iter(chunk), dtype=np.float64)


def convert_chain(
        txtfiles,
        headers,
        h5file,
        chunksize):
    """Converts chain in plain text format into HDF5 format.

    Keyword arguments:
    txtfiles -- list of paths to the plain text chains.
    headers -- name of each column.
    h5file -- where to put the resulting HDF5 file.
    chunksize -- how large the HDF5 chunk, i.e. number of rows.

    Chunking - How to pick a chunksize
    TODO Optimal chunk size unknown, our usage make caching irrelevant, and
    we use all read variable. Larger size should make compression more efficient,
    and less require less IO reads. Measurements needed.
    """

    h5 = h5py.File(h5file, 'w')

    for h in headers:
        h5.create_dataset(h,
                          shape=(0,),
                          maxshape=(None,),
                          dtype=np.float64,
                          chunks=(chunksize,),
                          compression='gzip',
                          shuffle=True)

    for txtfile in txtfiles:

        d = np.loadtxt(txtfile, dtype=np.float64)

        if len(d.shape) == 1:
            d = np.array([d])

        dnrows = d.shape[0]

        for pos, h in enumerate(headers):
            x = h5[h]
            xnrows = x.shape[0]
            x.resize(dnrows+xnrows, axis=0)
            x[xnrows:] = d[:,pos]

    h5.close()
