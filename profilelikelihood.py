
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import h5py
import barrett.util as util

class oneD:
    """ Calculate and plot the one dimensional profile likelihood.
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


    def profile(self):

        f = h5py.File(self.h5file, 'r')
        x = f[self.var]
        l = f['-2lnL']
        s = x.chunks[0]

        bins = 1e100*np.ones(self.nbins)

        for i in range(0, self.n, s):
            for j in range(i, s):
                bin_index = np.floor((x[j]-self.min)/self.bin_widths[0])

                if l[j] < bins[bin_index]:
                    bins[bin_index] = l[j]

        self.profchisq = bins - bins.min()
        self.proflike = np.exp(-bins/2)

        f.close()


    def plot(self, path):

        fig = plt.figure()

        plt.hist(self.bin_edges[:-1],
                 bins=self.bin_edges,
                 weights=self.proflike,
                 histtype='stepfilled',
                 alpha=0.5,
                 color='red')

        plt.ylim(0, self.proflike.max()*1.1)
        plt.xlabel('%s [%s]' % (self.name, self.unit))

        fig.savefig(path)


class twoD:
    """ Calculate and plot the two dimensional profile likelihood.
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


    def profile(self):

        f = h5py.File(self.h5file, 'r')
        x = f[self.xvar]
        y = f[self.yvar]
        l = f['-2lnL']
        s = x.chunks[0]

        bins = np.zeros((self.nbins, self.nbins))

        for i in range(0, self.n, s):
            for j in range(i,s):
                xbin_index = np.floor((x[j]-self.xmin)/self.xbin_widths[0])
                ybin_index = np.floor((y[j]-self.ymin)/self.ybin_widths[0])

                if -l[j] > bins[xbin_index, ybin_index]:
                    bins[xbin_index, ybin_index] = -l[j]

        # Normalise so the sum of the bins is one, i.e. we have a pdf.
        self.bins = bins / bins.sum()

        # Calculate the smoothed bins.
        # self.bins_smoothed = ndimage.gaussian_filter(
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
