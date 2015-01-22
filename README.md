# barrett

barrett is a software package meant to replace getplots in the [SuperBayeS](http://superbayes.org/) 
package. Main differential features are parallel and out-of-core processing so the 
user can handle very large datasets.

It is named after George Barrett (1752-1821) who took the supposed portrait of
Rev. Thomas Bayes. Some [history](http://www.york.ac.uk/depts/maths/histstat/bayespic.htm).

Specific technologies: HDF5, and Python (h5py, numpy, matplotlib). 


## Usage

barrett is split into three main modules:

 + barrett.data implements methods for modifying data (e.g. log, change units) or calculate 
   depended variables (e.g. mean squark mass)

 + barrett.posterior is for calculating and plot the one or two dimensional marginal 
   posterior distribution.

 + barrett.profilelikelihood is for calculating and plot the one or two dimensional profile 
   likelihood.

As for parallelisation; writing to the same hdf5 file is strongly discouraged. Reading the file 
is however perfectly fine. So posterior/profilelikelihood module is perfectly parallelisable. 

The code is not parallelised, instead I recommend using Python's multiprocessing module to 
producing several plots asynchronously. In most system tested the analysis is CPU bound, your
mileage may vary. 


## Example

Please check the example directory for plot.py for an, you guessed it, example.

