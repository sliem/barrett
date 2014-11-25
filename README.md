# py-getplots

py-getplots is meant to replace getplots in the [SuperBayeS](http://superbayes.org/) 
package. Main differential features are parallel and out-of-core processing so the 
user can handle huge datasets.

Specific technologies: MPI, HDF5, and Python (h5py, mpi4py, numpy, matplotlib). 

# TODO

- Calculating the posterior (1D and 2D)
- Calculating the profile likelihood (1D and 2D)
- Credibility intervals (1D and 2D)
- Confidence intervals (1D and 2D)
- Plotting.
- Transform values, e.g. change units, log, exponentiate.
- Calculate dependent variables, e.g. M_squark = mean of light squark masses.
- Smoothing distributions.
- Implement logic for estimating the appropiate binwidth.
- Simple configuration language to control the package (cf. ini files for getplots).
- Parallize!