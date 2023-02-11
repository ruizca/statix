# STATiX (Space and Time Algorithm for Transients in X-rays)

The Space and Time Algorithm for Transients in X-rays (STATiX) builds upon 
tools from the image and signal processing fields and in particular the 
Multi-Scale Variance Stabilisation Transform 
([Zhang et al. 2008](http://dx.doi.org/10.1109/TIP.2008.924386); 
[Starck et al. 2009](http://dx.doi.org/10.1051/0004-6361/200811388)) 
to provide a complete detection analysis pipeline optimised for finding 
transient sources on X-ray imaging observations. Unlike standard source 
detection codes, STATiX operates on 3-dimensional data cubes with 2-spatial 
and one temporal dimensions. It is therefore sensitive to short and faint 
X-ray flares that may be hidden in the background once the data cube is 
collapsed in time to produce 2-dimensional images. Although the algorithm 
is motivated by transient source searches, it also provides a competitive tool 
for the detection of the general, typically less variable, X-ray source
population present in X-ray observations. See Ruiz et al. 2023 (in preparation)
for a detailed explanation of the algorithm.

STATiX is distributed as a Python package. The current implementation 
only allows the processing of data for the XMM-Newton EPIC-pn camera. In the near
future we will extend the code for all XMM-Newton cameras. Upgrading the code
for other X-ray imaging missions is possible, but beyond our current capabilities.


Installation
------------

STATix needs the following software and libraries for a correct installation:
- C/C++ compiler and make
- [CMake](http://www.cmake.org)
- [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) (>V3.31)
- [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/)

In Ubuntu (and other Debian based Linux distributions) these dependencies can be installed via ``apt``:

    sudo apt install gcc make cmake libcfitsio* pkg-config


Once these prerequisites are installed, STATiX can be easily installed using ``pip``:

    pip install xstatix

Although the STATiX source detection pipeline does not need any additional software, some of its side functions related with XMM-Newton data manipulation need a working installation of [SAS](https://www.cosmos.esa.int/web/xmm-newton/what-is-sas). If all the initial data products are already available (images, data cubes, exposure maps, etc) SAS is not needed. Otherwise these products will be generated during running time if SAS is available. All SAS-related functions are in the [xmmsas](src/statix/xmmsas.py) module.


Examples
---------

We provide [Jupyter notebooks and scripts](docs/) with examples on how to use STATiX
with XMM-Newton data. 


[![ahead2020](ahead2020_logo.png)](http://ahead.astro.noa.gr/)

[![astropy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/) 
