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

    mass = ['log(m_{\chi})']
    coefficients = ['D_1', 'D_2', 'D_3', 'D_4', 'D_5', 'D_6']
    observables = ['log(<\sigma v>)', 'log(\Omega_{\chi}h^2)', 'log(\sigma_p^{SI})']

    plot1D(dataset, mass + coefficients + observables, 'histograms')
    plot_coefficients(dataset, coefficients)
    plot_vs_mass(dataset, coefficients, 'mass_vs_coefficients')
    plot_vs_mass(dataset, observables, 'mass_vs_observables')


def logObservables(dataset):

	d = data.Chain(dataset+'.h5')

	for v in ['<\sigma v>', '\Omega_{\chi}h^2', '\sigma_p^{SI}']:
		d.log(v)


def plot_coefficients(dataset, vars, nbins=60):

    if not os.path.isdir(dataset):
         os.mkdir(dataset)
    
    n = len(vars)

    fig, axes = plt.subplots(nrows=n,
                             ncols=n,
                             sharex='col',
                             sharey=False) # The histograms otherwise become terrible.

    plt.subplots_adjust(wspace=0, hspace=0)

    ticks = [-15, -10, -5, 0, 5, 10, 15]

    for i, y in enumerate(vars):
        for j, x in enumerate(vars):

            ax = axes[i,j]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            if j > i:
                # No need to do the analysis twice. (a[i,j] == a[j,i].T)
                ax.set_axis_off()

            elif i == j:
                P = posterior.oneD(dataset+'.h5', x, nbins)
                P.marginalise()
                P.plot(ax)
                ax.set_title(P.var)

                if i == 0:
                	ax.set_yticklabels([])
                	ax.set_ylabel(ax.get_xlabel())

            else:
                P = posterior.twoD(dataset+'.h5', x, y, nbins)
                P.marginalise()
               	P.plot(ax)

            if j != 0: 
                # remove labels from inner 
            	ax.set_yticklabels([])
            	ax.set_ylabel('')

    fig.set_size_inches(n*3+1,n*3+1)
    fig.savefig('%s/coefficients.png' % dataset, dpi=100)

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
		
        P = posterior.twoD(dataset+'.h5', 'log(m_{\chi})', y, nbins)
        P.marginalise()
        P.plot(ax)

    fig.set_size_inches(8,n*6)
    fig.savefig('%s/%s.png' % (dataset, filename), dpi=100)

    plt.close(fig)


def plot1D(dataset, vars, filename, nbins=60):

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

        P = posterior.oneD(dataset+'.h5', x, nbins)
        P.marginalise()
        P.plot(ax)

        ax.set_yticklabels([])

    fig.set_size_inches(4,n*2)
    fig.savefig('%s/%s.png' % (dataset, filename), dpi=100)

    plt.close(fig)


if __name__ == '__main__':
	main()