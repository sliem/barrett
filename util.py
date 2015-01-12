import h5py
import numpy as np
import itertools

def threenum(h5file, var):
    """ Calculates the three number summary for a variable.
    
    The three number summary is the minimum, maximum and the mean 
    of the data. Traditionally one would summerise data with the 
    five number summary: max, min, 1st, 2nd (median), 3rd quartile.
    But quantiles are hard to calculate without sorting the data 
    which hard to do out-of-core.
    """
    f = h5py.File(h5file, 'r')
    d = f[var]
    w = f['mult']
    s = d.chunks[0]

    n = d.shape[0]
    maxval = d[0]
    minval = d[0]
    total = 0
    wsum = 0

    for x in range(0, n, s):

        chunk_max = np.max(d[x:x+s])
        chunk_min = np.min(d[x:x+s])

        maxval = chunk_max if chunk_max > maxval else maxval
        minval = chunk_min if chunk_min < minval else minval

        total += np.sum(w[x:x+s]*d[x:x+s])
        wsum  += np.sum(w[x:x+s])
    f.close()

    mean = total/wsum

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
        units,
        h5file,
        chunksize):
    """Converts chain in plain text format into HDF5 format.

    Keyword arguments:
    txtfiles -- list of paths to the plain text chains.
    headers -- name of each column.
    units -- the unit of each column.
    h5file -- where to put the resulting HDF5 file.
    chunksize -- how large the HDF5 chunk, i.e. number of rows.

    Chunking - How to pick a chunksize
    TODO Optimal chunk size unknown, our usage make caching irrelevant, and 
    we use all read variable. Larger size should make compression more efficient,
    and less require less IO reads. Measurements needed.
    """

    h5 = h5py.File(h5file, 'w')

    for h, u in zip(headers, units):
        dset = h5.create_dataset(h,
                                 shape=(0,),
                                 maxshape=(None,),
                                 dtype=np.float64,
                                 chunks=(chunksize,),
                                 compression='gzip',
                                 shuffle=True)

        dset.attrs.create('unit', u.encode('utf8'))

    for txtfile in txtfiles:
        with open(txtfile, 'r') as f:
            for chunk in filechunk(f, chunksize):

                if chunk.shape == (): # single value, only happens for single column txt files
                    nrows = 1
                else:
                    nrows = chunk.shape[0]

                for pos, h in enumerate(headers):
                    dset = h5[h]
                    old_nrows = dset.shape[0]
                    dset.resize(old_nrows+nrows, axis=0)

                    if chunk.shape == (): # single value
                        dset[old_nrows] = chunk
                    elif len(chunk.shape) == 1: # txt file contains only one column
                        dset[old_nrows:] = chunk
                    else:
                        dset[old_nrows:] = chunk[:,pos]

    h5.close()

