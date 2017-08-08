from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='barrett',
      version='0.2.1',
      description='out-of-core processing and plotting of MultiNest output',
      long_description=long_description,
      url='https://github.com/sliem/barrett',
      author='Sebastian Liem',
      author_email='sebastian@liem.se',
      license='ISC',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Programming Language :: Python :: 3'
      ],
      keywords='out-of-core multinest statistics visualisation',
      packages=['barrett'],
      install_requires=['numpy', 'scipy', 'h5py', 'matplotlib']
)
