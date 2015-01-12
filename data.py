import h5py
import numpy as np

def column_apply(h5file, column, func):
    h5 = h5py.File(h5file, 'r+')
    d = h5[column]
    s = d.chunks[0]
    n = d.shape[0]

    for i in range(0, n, s):
        val = func(d[i:i+s])
        d[i:i+s] = val
    
    h5.close()

