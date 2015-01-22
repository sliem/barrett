#!/usr/bin/env python

import os
import barrett.posterior as posterior
import barrett.data as data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():

    dataset = 'RD'

    #logObservables(dataset)
    #calculateC(dataset)
    #logC(dataset)

    mass = ['log(m_{\chi})']
    C = ['log(C_1)', 'log(C_2)', 'log(C_3)', 'log(C_4)', 'log(C_5)', 'log(C_6)']
    observables = ['log(<\sigma v>)', 'log(\Omega_{\chi}h^2)', 'log(\sigma_p^{SI})']

    plot_histograms(dataset, mass + C + observables, 'histograms')
    plot_coefficients(dataset, C, 'coefficients')
    plot_vs_mass(dataset, C, 'mass_vs_coefficients')
    plot_vs_mass(dataset, observables, 'mass_vs_observables')


def logObservables(dataset):

	d = data.Chain(dataset+'.h5')

	for v in ['<\sigma v>', '\Omega_{\chi}h^2', '\sigma_p^{SI}']:
		d.log(v)


def calculateC(dataset):
    
    def DtoC(D, logm, k):
        return np.sign(D) * k * np.power(10, -np.abs(D) - 2 * logm)

    def DtoC1234(D, logm):
        k = 39.4784176044
        return DtoC(D, logm, k)

    def DtoC56(D, logm):
        k = 1.16856116108
        return DtoC(D, logm, k)

    d = data.Chain(dataset+'.h5')

    for n in ['1', '2', '3', '4']:
        d.apply('C_'+n, 'GeV^{-2}', DtoC1234, 'D_'+n, 'log(m_{\chi})')

    for n in ['5', '6']:
        d.apply('C_'+n, 'GeV^{-2}', DtoC56, 'D_'+n, 'log(m_{\chi})')


def logC(dataset):
    def log(x):

        if np.abs(x) > 1:
            return 1e10
        
        return np.sign(x) * np.log10(np.abs(x))

    d = data.Chain(dataset+'.h5')

    for n in ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6']:
        d.apply('log(%s)' % n, 'log(GeV^{-2})', np.vectorize(log, otypes=[np.float64]), n) 


def plot_coefficients(dataset, vars, filename, nbins=60):

    if not os.path.isdir(dataset):
         os.mkdir(dataset)
    
    n = len(vars)

    fig, axes = plt.subplots(nrows=n,
                             ncols=n,
                             sharex='col',
                             sharey=False) # The histograms otherwise become terrible.

    plt.subplots_adjust(wspace=0, hspace=0)

    for i, y in enumerate(vars):
        for j, x in enumerate(vars):

            ax = axes[i,j]

            if j > i:
                # No need to do the analysis twice. (a[i,j] == a[j,i].T)
                ax.set_axis_off()

            elif i == j:
                P = posterior.oneD(dataset+'.h5', x, limits=(-30,30), nbins=nbins)
                P.marginalise()
                P.plot(ax)
                ax.set_title(P.var)

                if i == 0:
                	ax.set_yticklabels([])
                	ax.set_ylabel(ax.get_xlabel())

            else:
                P = posterior.twoD(dataset+'.h5', x, y, xlimits=(-30,30), ylimits=(-30,30), nbins=nbins)
                P.marginalise()
               	P.plot(ax)

            if j != 0: 
                # remove labels from inner 
            	ax.set_yticklabels([])
            	ax.set_ylabel('')

    fig.set_size_inches(n*3+1,n*3+1)
    fig.savefig('%s/%s.png' % (dataset, filename), dpi=100)

    plt.close(fig)


def plot_vs_mass(dataset, vars, filename, nbins=60):

    if not os.path.isdir(dataset):
         os.mkdir(dataset)
    
    n = len(vars)

    fig, axes = plt.subplots(nrows=n,
                             ncols=1,
                             sharex='col',
                             sharey=False)

    plt.subplots_adjust(wspace=0, hspace=0)

    for i, y in enumerate(vars):
        ax = axes[i]

        if y[0:6] == 'log(C_':
            ylimits = (-30, 30)
        else:
            ylimits = None	

        P = posterior.twoD(dataset+'.h5', 'log(m_{\chi})', y, ylimits=ylimits, nbins=nbins)
        P.marginalise()
        P.plot(ax)

    fig.set_size_inches(8,n*6)
    fig.savefig('%s/%s.png' % (dataset, filename), dpi=100)

    plt.close(fig)


def plot_histograms(dataset, vars, filename, nbins=60):

    if not os.path.isdir(dataset):
         os.mkdir(dataset)
    
    n = len(vars)

    fig, axes = plt.subplots(nrows=n,
                             ncols=1,
                             sharex=False,
                             sharey=False)

    plt.subplots_adjust(wspace=0, hspace=1)

    for i, x in enumerate(vars):
        ax = axes[i]

        if x[0:6] == 'log(C_':
            limits = (-30, 30)
        else:
            limits = None

        P = posterior.oneD(dataset+'.h5', x, limits=limits, nbins=nbins)
        P.marginalise()
        P.plot(ax)

        ax.set_yticklabels([])

    fig.set_size_inches(4,n*2)
    fig.savefig('%s/%s.png' % (dataset, filename), dpi=100)

    plt.close(fig)


if __name__ == '__main__':
	main()

