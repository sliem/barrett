# barrett

barrett is a software package meant to replace getplots in the [SuperBayeS](http://superbayes.org/) 
package. Main differential features are parallel and out-of-core processing so the 
user can handle very large datasets.

It is named after George Barrett (1752-1821) who took the supposed portrait of
Rev. Thomas Bayes. Some [history](http://www.york.ac.uk/depts/maths/histstat/bayespic.htm).

Specific technologies: MPI, HDF5, and Python (h5py, mpi4py, numpy, matplotlib). 

# TODO

- Calculating the profile likelihood (1D and 2D)
- Confidence intervals (1D and 2D)
- Plotting.
- Transform values, e.g. change units, log, exponentiate.
- Calculate dependent variables, e.g. M_squark = mean of light squark masses.
- Smoothing distributions.
- Implement logic for estimating the appropiate binwidth.
- Simple configuration language to control the package (cf. ini files for getplots).
- Parallize!
