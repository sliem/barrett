
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import h5py
import barrett.util as util

class oneD:
    """ Calculate and plot the one dimensional marginalised posteriors.
    """
    def __init__(self, h5file, var, limits=None, nbins=None):

        self.h5file = h5file
        self.var = var

        with h5py.File(h5file, 'r') as f:
            self.n = f[self.var].shape[0]
            self.name = f[self.var].name
            self.unit = f[self.var].attrs['unit'].decode('utf8')

        self.min, self.max, self.mean = util.threenum(self.h5file, self.var)

        if limits is None:
            self.limits = (0.99*self.min, 1.01*self.max)
        else:
            self.limits = limits

        if nbins != None:
            self.nbins = nbins
        else:
            self.nbins = np.floor(self.n**1/2)

        self.bin_edges  = np.linspace(self.limits[0], self.limits[1], num=self.nbins+1)
        self.bin_widths = np.diff(self.bin_edges)
        self.bins = None


    def marginalise(self):

        f = h5py.File(self.h5file, 'r')
        x = f[self.var]
        w = f['mult']
        s = x.chunks[0]

        bins = np.zeros(self.nbins)

        for i in range(0, self.n, s):
            aN = ~np.logical_or(np.isnan(x[i:i+s]), np.isinf(x[i:i+s]))
            ret = np.histogram(x[i:i+s][aN], weights=w[i:i+s][aN], bins=self.bin_edges)
            bins = bins + ret[0]

        self.bins = bins

        # Calculate the smoothed bins.
        self.bins_smoothed = ndimage.gaussian_filter(
                    self.bins,
                    sigma = self.bin_widths[0],
                    order = 0)

        self.bins_smoothed = self.bins_smoothed

        f.close()


    def plot(self, ax, smoothed=False):

        if smoothed:
            pdf = self.bins_smoothed
        else:
            pdf = self.bins

        ax.hist(self.bin_edges[:-1],
                 bins=self.bin_edges,
                 weights=pdf,
                 histtype='stepfilled',
                 alpha=0.5,
                 color='red')

        ax.set_ylim(0, pdf.max()*1.1)
        ax.set_xlabel('%s [%s]' % (self.name, self.unit))


class twoD:
    """ Calculate and plot the two dimensional marginalised posteriors.
    """

    def __init__(self, h5file, xvar, yvar, xlimits=None, ylimits=None, nbins=None):

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

        if xlimits is None:
            self.xlimits = (0.99*self.xmin, 1.01*self.xmax)
        else:
            self.xlimits = xlimits

        if ylimits is None:
            self.ylimits = (0.99*self.ymin, 1.01*self.ymax)
        else:
            self.ylimits = ylimits

        if nbins != None:
            self.nbins = nbins
        else:
            self.nbins = np.floor(self.n**1/2)

        self.xbin_edges  = np.linspace(self.xlimits[0], self.xlimits[1], num=self.nbins+1)
        self.ybin_edges  = np.linspace(self.ylimits[0], self.ylimits[1], num=self.nbins+1)

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
        self.bins = bins.T


        # Calculate the smoothed bins.
        self.bins_smoothed = ndimage.gaussian_filter(
                    self.bins,
                    sigma=(self.xbin_widths[0], self.ybin_widths[0]),
                    order = 0)

        f.close()


    def plot(self, ax, levels=[0.95, 0.68], smoothing=False):

        if smoothing:
            pdf = self.bins_smoothed
        else:
            pdf = self.bins

        xcenter = self.xbin_edges[:-1] + self.xbin_widths/2
        ycenter = self.ybin_edges[:-1] + self.ybin_widths/2

        X, Y = np.meshgrid(xcenter, ycenter)

        cmap = matplotlib.cm.gist_heat_r

        levels = list(self.credibleregions(crlevels, smoothing=smoothing)) + [pdf.max()]

        ax.contourf(X, Y, pdf, levels=levels, cmap=cmap)

        ax.set_xlabel('%s [%s]' % (self.xname, self.xunit))
        ax.set_ylabel('%s [%s]' % (self.yname, self.yunit))


    def credibleregions(self, probs, smoothing=False):
        """ Calculates the credible regions.
        """

        if smoothing:
            pdf = self.bins_smoothed
        else:
            pdf = self.bins

        step = 0.0001

        probs = np.array(probs)
        levels = np.zeros(probs.size)

        for i, p in enumerate(probs):
            while np.ma.masked_where(pdf < levels[i], pdf).sum() > p:
                levels[i] += step

        return levels
