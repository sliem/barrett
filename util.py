import h5py
import numpy as np

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

