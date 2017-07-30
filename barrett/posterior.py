
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import brentq
import h5py
import barrett.util as util

class oneD:
    """ Calculate and plot the one dimensional marginalised posteriors.
    """
    def __init__(self, h5file, var, limits=None, bins=None, post_col='mult'):

        self.h5file = h5file
        self.var = var

        h5 = h5py.File(h5file, 'r')

        self.n = h5[self.var].shape[0]
        self.name = h5[self.var].name
        self.chunksize = h5[self.var].chunks[0]

        self.min, self.max, self.mean = util.threenum(self.h5file, self.var)
        self.bins = np.floor(self.n**0.5) if bins is None else bins
        self.nbins = self.bins if np.isscalar(self.bins) else self.bins.shape[0]
        self.limits = (self.min, self.max) if limits is None else limits

        self.pdf = np.zeros(self.nbins)
        s = self.chunksize
        for i in range(0, self.n, s):
            r = stats.binned_statistic(h5[self.var][i:i+s],
                                       h5[post_col][i:i+s],
                                       'sum',
                                       bins=self.bins,
                                       range=self.limits)
            self.pdf += r.statistic

        self.bin_edges  = r.bin_edges

        h5.close()


    def plot(self, ax, **hist_kwargs):

        defaults = {
                'color'    : 'red',
                'alpha'    : 0.5,
                'histtype' : 'stepfilled'
        }
        defaults.update(hist_kwargs)

        ax.hist(self.bin_edges[:-1],
                 bins=self.bin_edges,
                 weights=self.pdf,
                 **defaults)

        ax.set_ylim(0, self.pdf.max()*1.1)
        ax.set_xlabel('%s' % (self.name))


class twoD:
    """ Calculate and plot the two dimensional marginalised posteriors.
    """

    def __init__(self, h5file, xvar, yvar, xlimits=None, ylimits=None, xbins=None, ybins=None, post_col='mult'):

        self.h5file = h5file
        self.xvar = xvar
        self.yvar = yvar

        h5 = h5py.File(h5file, 'r')

        self.n = h5[self.xvar].shape[0]
        self.chunksize = h5[self.xvar].chunks[0]
        self.xname = h5[self.xvar].name
        self.yname = h5[self.yvar].name

        self.xmin, self.xmax, self.xmean = util.threenum(self.h5file, self.xvar)
        self.ymin, self.ymax, self.ymean = util.threenum(self.h5file, self.yvar)

        self.xbins = np.floor(self.n**0.5) if xbins is None else xbins
        self.xnbins = self.xbins if np.isscalar(self.xbins) else self.xbins.shape[0] - 1
        self.ybins = np.floor(self.n**0.5) if ybins is None else ybins
        self.ynbins = self.ybins if np.isscalar(self.ybins) else self.ybins.shape[0] - 1
        self.xlimits = (self.xmin, self.xmax) if xlimits is None else xlimits
        self.ylimits = (self.ymin, self.ymax) if ylimits is None else ylimits

        self.pdf = np.zeros((self.xnbins, self.ynbins))
        s = self.chunksize
        for i in range(0, self.n, s):
            r = stats.binned_statistic_2d(h5[self.xvar][i:i+s],
                                          h5[self.yvar][i:i+s],
                                          h5[post_col][i:i+s],
                                          'sum',
                                          bins=(self.xbins, self.ybins),
                                          range=[self.xlimits, self.ylimits])
            self.pdf += r.statistic
        self.pdf = self.pdf.T

        self.xbin_edges  = r.x_edge
        self.ybin_edges  = r.y_edge
        self.xcenters = self.xbin_edges[:-1] + np.diff(self.xbin_edges)/2.0
        self.ycenters = self.ybin_edges[:-1] + np.diff(self.ybin_edges)/2.0

        h5.close()


    def plot(self, ax, levels=[0.95, 0.68], cmap=None, **contourf_kwargs):

        X, Y = np.meshgrid(self.xcenters, self.ycenters)

        if levels is None:
            levels = np.linspace(0, self.pdf.max(), 10)[1:]
        else:
            levels = np.append(self.credibleregions(levels), self.pdf.max())

        if cmap is None:
            cmap = matplotlib.cm.gist_heat_r

        colors = [cmap(i) for i in np.linspace(0.2,0.8,len(levels))][1:]

        defaults = {
                'colors'    : colors,
        }
        defaults.update(contourf_kwargs)

        ax.contourf(X, Y, self.pdf, levels=levels, **defaults)

        ax.set_xlabel('%s' % (self.xname))
        ax.set_ylabel('%s' % (self.yname))


    def credibleregions(self, probs):
        """ Calculates the credible regions.
        """

        return [brentq(lambda l:  self.pdf[self.pdf > l].sum() - p, 0.0, 1.0) for p in probs]
