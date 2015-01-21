
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import h5py
import barrett.util as util

class oneD:
    """ Calculate and plot the one dimensional marginalised posteriors.
    """
    def __init__(self, h5file, var, nbins=None):

        self.h5file = h5file
        self.var = var

        with h5py.File(h5file, 'r') as f:
            self.n = f[self.var].shape[0]
            self.name = f[self.var].name
            self.unit = f[self.var].attrs['unit'].decode('utf8')
        
        self.min, self.max, self.mean = util.threenum(self.h5file, self.var)

        if nbins != None:
            self.nbins = nbins
        else:
            self.nbins = np.floor(self.n**1/2)

        self.bin_edges  = np.linspace(self.min, self.max, num=self.nbins+1)
        self.bin_widths = np.diff(self.bin_edges)
        self.bins = None


    def marginalise(self):

        f = h5py.File(self.h5file, 'r')
        x = f[self.var]
        w = f['mult']
        s = x.chunks[0]

        bins = np.zeros(self.nbins)

        for i in range(0, self.n, s):
            ret = np.histogram(x[i:i+s], weights=w[i:i+s], bins=self.bin_edges)
            bins = bins + ret[0]

        self.bins = bins / bins.sum()

        # Calculate the smoothed bins.
        # self.bins_smoothed = ndimage.gaussian_filter(
        #            self.bins,
        #            sigma = self.bin_widths[0],
        #            order = 0)

        #self.bins_smoothed = self.bins_smoothed / self.bins_smoothed.sum()

        f.close()


    def plot(self, path, smoothed=False):

        if smoothed:
            pdf = self.bins_smoothed
        else:
            pdf = self.bins

        fig = plt.figure()

        plt.hist(self.bin_edges[:-1],
                 bins=self.bin_edges,
                 weights=pdf,
                 histtype='stepfilled',
                 alpha=0.5,
                 color='red')

        plt.ylim(0, pdf.max()*1.1)
        plt.xlabel('%s [%s]' % (self.name, self.unit))

        fig.savefig(path)


class twoD:
    """ Calculate and plot the two dimensional marginalised posteriors.
    """

    def __init__(self, h5file, xvar, yvar, nbins=None):

        self.h5file = h5file
        self.xvar = xvar
        self.yvar = yvar

        with h5py.File(h5file, 'r') as f:
            self.n = f[self.xvar].shape[0]
            self.xname = f[self.xvar].name
            self.xunit = f[self.xvar].attrs['unit'].decode('utf8')
            self.yname = f[self.yvar].name
            self.yunit = f[self.yvar].attrs['unit'].decode('utf8')
        
        self.xmin, self.xmax, self.xmean = util.threenum(self.h5file, self.xvar)
        self.ymin, self.ymax, self.ymean = util.threenum(self.h5file, self.yvar)

        if nbins != None:
            self.nbins = nbins
        else:
            self.nbins = np.floor(self.n**1/2)

        self.xbin_edges  = np.linspace(self.xmin, self.xmax, num=self.nbins+1)
        self.ybin_edges  = np.linspace(self.ymin, self.ymax, num=self.nbins+1)

        self.xbin_widths = np.diff(self.xbin_edges)
        self.ybin_widths = np.diff(self.ybin_edges)

        self.bins = None


    def marginalise(self):

        f = h5py.File(self.h5file, 'r')
        x = f[self.xvar]
        y = f[self.yvar]
        w = f['mult']
        s = x.chunks[0]

        bins = np.zeros((self.nbins, self.nbins))

        for i in range(0, self.n, s):
            ret = np.histogram2d(x[i:i+s], y[i:i+s], weights=w[i:i+s], 
                    bins=(self.xbin_edges, self.ybin_edges))

            bins = bins + ret[0]

        # Normalise so the sum of the bins is one, i.e. we have a pdf.
        self.bins = bins.T / bins.sum()


        # Calculate the smoothed bins.
        #self.bins_smoothed = ndimage.gaussian_filter(
        #            self.bins,
        #            sigma=(self.xbin_widths[0], self.ybin_widths[0]),
        #            order = 0)

        #self.bins_smoothed = self.bins_smoothed / self.bins_smoothed.sum()

        f.close()


    def plot(self, path, smoothing=False):

        if smoothing:
            pdf = self.bins_smoothed
        else:
            pdf = self.bins

        fig = plt.figure()

        xcenter = self.xbin_edges[:-1] + self.xbin_widths/2
        ycenter = self.ybin_edges[:-1] + self.ybin_widths/2

        X, Y = np.meshgrid(xcenter, ycenter)
        
        cmap = matplotlib.cm.gist_heat_r
        levels = np.linspace(0.0, pdf.max(), 100)
       
        cred_levels = self.credibleregions([0.95, 0.68], smoothing)

        plt.contourf(X, Y, pdf, levels=levels, cmap=cmap)
        plt.contour(X, Y, pdf, levels=cred_levels, colors='k')

        plt.xlabel('%s [%s]' % (self.xname, self.xunit))
        plt.ylabel('%s [%s]' % (self.yname, self.yunit))

        fig.savefig(path)


    def credibleregions(self, probs, smoothing=False):
        """ Calculates the one and two sigma credible regions.

        The algorithm is extremely simple: 
        
        1. start with z = 0
        2. calculate s = sum(pdf(x)) where pdf(x) >= z
        3. if s >= p, return z
        4. else z += step and goto 2.

        z defines the regions in conjunction with contour/contourf
        """

        if smoothing:
            pdf = self.bins_smoothed
        else:
            pdf = self.bins

        levels = []

        z = 0.0
        step = pdf.max()*1e-4
        
        for p in sorted(probs):
            while True:
                s = np.ma.masked_where(pdf > z, pdf).sum()

                if s > p:
                    levels.append(z)
                    break

                z += step
        return levels

