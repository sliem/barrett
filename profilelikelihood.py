
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.stats as stats
import h5py
import barrett.util as util

class oneD:
    """ Calculate and plot the one dimensional profile likelihood.
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


    def profile(self):

        f = h5py.File(self.h5file, 'r')
        x = f[self.var]
        l = f['-2lnL']
        s = x.chunks[0]

        bins = 1e100*np.ones(self.nbins)

        for i in range(0, self.n, s):
            for j in range(i, s):
                bin_index = np.floor((x[j]-self.limits[0])/self.bin_widths[0])

                if l[j] < bins[bin_index]:
                    bins[bin_index] = l[j]

        self.profchisq = bins - bins.min()
        self.proflike = np.exp(-bins/2)

        f.close()


    def plot(self, ax):

        ax.hist(self.bin_edges[:-1],
                 bins=self.bin_edges,
                 weights=self.proflike,
                 histtype='stepfilled',
                 alpha=0.5,
                 color='red')

        ax.set_ylim(0, self.proflike.max()*1.1)
        ax.set_xlabel('%s [%s]' % (self.name, self.unit))


class twoD:
    """ Calculate and plot the two dimensional profile likelihood.
    """

    def __init__(self, h5file, xvar, yvar, xlimits=None, ylimits=None, zero=False, nbins=None):

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

            if zero:
                self.xlimits = (0.0, self.xlimits[1])

        else:
            self.xlimits = xlimits

        if ylimits is None:
            self.ylimits = (0.99*self.ymin, 1.01*self.ymax)

            if zero:
                self.ylimits = (0.0, self.ylimits[1])

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


    def profile(self):

        f = h5py.File(self.h5file, 'r')
        x = f[self.xvar]
        y = f[self.yvar]
        l = f['-2lnL']
        s = x.chunks[0]

        bins = 1e100*np.ones((self.nbins, self.nbins))

        for i in range(0, self.n, s):
            for j in range(i,s):
                xbin_index = np.floor((x[j]-self.xlimits[0])/self.xbin_widths[0])
                ybin_index = np.floor((y[j]-self.ylimits[0])/self.ybin_widths[0])

                if xbin_index > bins.shape[0] or ybin_index > bins.shape[1]:
                    pass

                if l[j] < bins[ybin_index, xbin_index]:
                    bins[ybin_index, xbin_index] = l[j]

        self.profchisq = bins - bins.min() 
        self.proflike = np.exp(-self.profchisq/2)

        f.close()


    def plot(self, ax, smoothing=False):

        xcenter = self.xbin_edges[:-1] + self.xbin_widths/2
        ycenter = self.ybin_edges[:-1] + self.ybin_widths/2

        X, Y = np.meshgrid(xcenter, ycenter)
        
        cmap = matplotlib.cm.gist_heat_r
        levels = np.linspace(0, self.proflike.max(), 10)

        conf_levels = self.confidenceregions([0.95, 0.68])      
 
        ax.contourf(X, Y, self.proflike, levels=levels, cmap=cmap)
        ax.contour(X, Y, self.proflike, levels=conf_levels, colors='k')

        ax.set_xlabel('%s [%s]' % (self.xname, self.xunit))
        ax.set_ylabel('%s [%s]' % (self.yname, self.yunit))


    def confidenceregions(self, probs):
       
        deltachi2 = stats.chi2.ppf(probs, 2)

        return np.exp(-stats.chi2.ppf(probs, 2)/2)

