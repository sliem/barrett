=======
barrett
=======

barrett is a software package meant to replace getplots in the SuperBayeS
package or other MultiNest applicationsj. Main differential feature is out-of-core processing 
so the code can handle very large datasets.

Specific technologies: HDF5, and Python (h5py, numpy, matplotlib).

Usage
-----

barrett is split into four submodules:

 + barrett.data implements methods for modifying data (e.g. log, change units) or calculate
   depended variables (e.g. mean squark mass)

 + barrett.posterior is for calculating and plot the one or two dimensional marginal
   posterior distribution.

 + barrett.profilelikelihood is for calculating and plot the one or two dimensional profile
   likelihood.

 + barrett.util contain various utility functions most notable convert_chain() which converts
   the plain text MultiNest output to the HDF5 format used by barrett.

As for parallelisation; writing to the same hdf5 file is strongly discouraged. Reading the file
is however perfectly fine. So posterior/profilelikelihood module is perfectly parallelisable.

The code is not parallelised, instead I recommend using Python's multiprocessing module to
producing several plots asynchronously. In most system tested the analysis is CPU bound, your
mileage may vary.

Example
-----

Please check the example directory for plot.py for an, you guessed it, example.

