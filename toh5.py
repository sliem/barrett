
import h5py
import numpy as np
import itertools


def filechunk(f, chunksize):
    """Iterator that allow for piecemeal processing of a file."""
    while True:
        chunk = tuple(itertools.islice(f, chunksize))
        if not chunk:
            return
        yield np.loadtxt(iter(chunk), dtype=np.float64)


def parse_infofile(fname):
    """Parse a info file and returns headers, units and other metadata as lists."""

    headers = ['mult', '-2lnL']
    units = ['', '']
    metadata = []

    with open(fname, 'r') as f:
        for l in f:
            if l[0:3] == 'lab':
                v = l.split('=')[1].split('(')
                headers.append(v[0].strip())

                if len(v) > 1:
                    units.append(v[1].strip()[:-1])
                else:
                    units.append('')

    return (headers, units, metadata)


def convert_chain(
        txtfile,
        infofile,
        h5file,
        chunksize):
    """Converts a chain in plain text format into HDF5 format.

    Keyword arguments:
    txtfile -- path to the chain in plain text.
    infofile -- path to associated info file with metadata.
    h5file -- path to where to put the resulting HDF5 file.
    chunksize -- how large the HDF5 chunk, i.e. number of rows.

    Chunking - Howto pick a chunksize
    TODO Optimal chunk size unknown, our usage make caching irrelevant, and 
    we use all read variable. Larger size should make compression more efficient,
    and less require less IO reads. Measurements needed.
    """
    headers, units, metadata = parse_infofile(infofile)

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


