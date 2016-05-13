#!/usr/bin/env python

import barrett.posterior as posterior
import barrett.data as data

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def main():
    """ We here demostrate the basic functionality of barrett. We use a global scan 
    of scalar dark matter as an example. The details aren't really important.
    """
    dataset = 'RD'

    observables = ['log(<\sigma v>)', '\Omega_{\chi}h^2', 'log(\sigma_p^{SI})']
    var  = ['log(m_{\chi})']
    var += ['log(C_1)', 'log(C_2)', 'log(C_3)', 'log(C_4)', 'log(C_5)', 'log(C_6)']
    var += observables

    plot_vs_mass(dataset, observables, 'mass_vs_observables.png')
    plot_oneD(dataset, var, 'oneD.png')


def plot_vs_mass(dataset, vars, filename, bins=60):
    """ Plot 2D marginalised posteriors of the 'vars' vs the dark matter mass.
    We plot the one sigma, and two sigma filled contours. More contours can be plotted 
    which produces something more akin to a heatmap.

    If one require more complicated plotting, it is recommended to write a custom 
    plotting function by extending the default plot() method.
    """

    n = len(vars)

    fig, axes = plt.subplots(nrows=n,
                             ncols=1,
                             sharex='col',
                             sharey=False)

    plt.subplots_adjust(wspace=0, hspace=0)

    m = 'log(m_{\chi})'

    for i, y in enumerate(vars):
        ax = axes[i]
        P = posterior.twoD(dataset+'.h5', m, y, 
                           xlimits=limits(m), ylimits=limits(y), xbins=bins, ybins=bins)

        # apply some gaussian smoothing to make the contours slightly smoother
        sigmas = (np.diff(P.ycenters)[0], np.diff(P.xcenters)[0])
        P.pdf = gaussian_filter(P.pdf, sigmas, mode='nearest')

        P.plot(ax, levels=np.linspace(0.9, 0.1, 9))
        ax.set_xlabel(labels('log(m_{\chi})'))
        ax.set_ylabel(labels(y))

    fig.set_size_inches(4,n*3)
    fig.savefig(filename, dpi=200, bbox_inches='tight')

    plt.close(fig)


def plot_oneD(dataset, vars, filename, bins=60):
    """ Plot 1D marginalised posteriors for the 'vars' of interest."""
    n = len(vars)

    fig, axes = plt.subplots(nrows=n,
                             ncols=1,
                             sharex=False,
                             sharey=False)

    for i, x in enumerate(vars):
        ax = axes[i]

        P = posterior.oneD(dataset+'.h5', x, limits=limits(x), bins=bins)
        P.plot(ax)
        ax.set_xlabel(labels(x))

        ax.set_yticklabels([])

    fig.set_size_inches(4, 4*n)
    fig.savefig(filename, dpi=200, bbox_inches='tight')

    plt.close(fig)


# Define labels and the limits, or ranges, for the variables we're plotting.

limits_dict = {
    'log(m_{\chi})' : (0,3),
    '\Omega_{\chi}h^2' : (0, 0.2),
    'log(<\sigma v>)' : (-45, -20),
    'log(\sigma_p^{SI})' : (-30, 10)
}

labels_dict = {
    'log(m_{\chi})' : 'log10(mDM / 1 GeV)',
    '\Omega_{\chi}h^2' : 'oh2',
    'log(<\sigma v>)' : 'log10(<sv> / 1 cm^3 s^-1)',
    'log(\sigma_p^{SI})' : 'log10(sigma_SI / 1 pb)'
}

for n in range(1,7):
    limits_dict['log(C_%i)' % n] = (-30, 30)
    labels_dict['log(C_%i)' % n] = 'log10(C%i / 1 GeV^-2)' % n

def limits(x):
    return None if x not in limits_dict else limits_dict[x]

def labels(x):
    return None if x not in labels_dict else labels_dict[x]

if __name__ == '__main__':
	main()
