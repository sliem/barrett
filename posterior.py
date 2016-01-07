
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import h5py
import barrett.util as util

class oneD:
    """ Calculate and plot the one dimensional marginalised posteriors.
    """
    def __init__(self, h5file, var, limits=None, nbins=None):

        self.h5file = h5file
        self.var = var

        h5 = h5py.File(h5file, 'r')

        self.n = h5[self.var].shape[0]
        self.name = h5[self.var].name
        self.unit = h5[self.var].attrs['unit'].decode('utf8')
        self.chunksize = h5[self.var].chunks[0]

        self.min, self.max, self.mean = util.threenum(self.h5file, self.var)
        self.nbins = np.floor(self.n**1/2) if nbins is None else nbins
        self.limits = (self.min, self.max) if limits is None else limits

        self.bins = np.zeros(self.nbins)
        s = self.chunksize
        for i in range(0, self.n, s):
            r = stats.binned_statistic(h5[self.var][i:i+s],
                                       h5['mult'][i:i+s],
                                       'sum',
                                       bins=self.nbins,
                                       range=self.limits)
            self.bins += r.statistic

        self.bin_edges  = r.bin_edges

        h5.close()


    def plot(self, ax):

        ax.hist(self.bin_edges[:-1],
                 bins=self.bin_edges,
                 weights=self.bins,
                 histtype='stepfilled',
                 alpha=0.5,
                 color='red')

        ax.set_ylim(0, self.bins.max()*1.1)
        ax.set_xlabel('%s [%s]' % (self.name, self.unit))


class twoD:
    """ Calculate and plot the two dimensional marginalised posteriors.
    """

    def __init__(self, h5file, xvar, yvar, xlimits=None, ylimits=None, nbins=None):

        self.h5file = h5file
        self.xvar = xvar
        self.yvar = yvar

        h5 = h5py.File(h5file, 'r')

        self.n = h5[self.xvar].shape[0]
        self.chunksize = h5[self.xvar].chunks[0]
        self.xname = h5[self.xvar].name
        self.xunit = h5[self.xvar].attrs['unit'].decode('utf8')
        self.yname = h5[self.yvar].name
        self.yunit = h5[self.yvar].attrs['unit'].decode('utf8')

        self.xmin, self.xmax, self.xmean = util.threenum(self.h5file, self.xvar)
        self.ymin, self.ymax, self.ymean = util.threenum(self.h5file, self.yvar)

        self.nbins = np.floor(self.n**1/2) if nbins is None else nbins
        self.xlimits = (self.xmin, self.xmax) if xlimits is None else xlimits
        self.ylimits = (self.ymin, self.ymax) if ylimits is None else ylimits

        self.bins = np.zeros((self.nbins, self.nbins))
        s = self.chunksize
        for i in range(0, self.n, s):
            r = stats.binned_statistic_2d(h5[self.xvar][i:i+s],
                                          h5[self.yvar][i:i+s],
                                          h5['mult'][i:i+s],
                                          'sum',
                                          bins=self.nbins,
                                          range=[self.xlimits, self.ylimits])
            self.bins += r.statistic
        self.bins = self.bins.T

        self.xbin_edges  = r.x_edge
        self.ybin_edges  = r.y_edge
        self.xcenters = self.xbin_edges[:-1] + np.diff(self.xbin_edges)/2
        self.ycenters = self.ybin_edges[:-1] + np.diff(self.ybin_edges)/2

        h5.close()


    def plot(self, ax, levels=[0.95, 0.68]):

        X, Y = np.meshgrid(self.xcenters, self.ycenters)

        if levels == None:
            levels = np.linspace(0, self.bins.max(), 10)[1:]
        else:
            levels = np.append(self.credibleregions(levels), self.bins.max())

        cmap = matplotlib.cm.gist_heat_r
        colors = [cmap(i) for i in np.linspace(0.2,0.8,len(levels))][1:]
        ax.contourf(X, Y, self.bins, levels=levels, colors=colors)

        ax.set_xlabel('%s [%s]' % (self.xname, self.xunit))
        ax.set_ylabel('%s [%s]' % (self.yname, self.yunit))


    def credibleregions(self, probs):
        """ Calculates the credible regions.
        """

        step = 0.0001

        probs = np.array(probs)
        levels = np.zeros(probs.size)

        for i, p in enumerate(probs):
            while np.ma.masked_where(self.bins < levels[i], self.bins).sum() > p:
                levels[i] += step

        return levels
