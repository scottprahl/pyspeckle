pyspeckle
=========

by Scott Prahl

.. image:: https://img.shields.io/pypi/v/pyspeckle
   :target: https://pypi.org/project/pyspeckle/
   :alt: PyPI

.. image:: https://img.shields.io/conda/v/conda-forge/pyspeckle.svg
   :target: https://anaconda.org/conda-forge/pyspeckle
   :alt: Conda

.. image:: https://zenodo.org/badge/131667397.svg
   :target: https://zenodo.org/badge/latestdoi/131667397
      
|

.. image:: https://img.shields.io/github/license/scottprahl/pyspeckle
   :target: https://github.com/scottprahl/pyspeckle/blob/master/LICENSE.txt
   :alt: MIT License

.. image:: https://github.com/scottprahl/miepython/actions/workflows/test.yml/badge.svg
   :target: https://github.com/scottprahl/miepython/actions/workflows/test.yml
   :alt: Testing

.. image:: https://readthedocs.org/projects/pyspeckle2/badge
   :target: https://pyspeckle2.readthedocs.io
   :alt: Read the Docs

.. image:: https://img.shields.io/pypi/dm/pyspeckle
   :target: https://pypi.org/project/pyspeckle/
   :alt: Number of PyPI downloads

________

A collection of routines to track and analyze laser speckle.  This is a python
port of SimSpeckle Matlab routines described in::

    `Duncan & Kirkpatrick, "Algorithms for simulation of speckle (laser and otherwise)," in SPIE Vol. 6855 (2008) 
    <https://www.researchgate.net/profile/Sean-Kirkpatrick-2/publication/233783056_Algorithms_for_simulation_of_speckle_laser_and_otherwise/links/09e4150b78c4e8fe5f000000/Algorithms-for-simulation-of-speckle-laser-and-otherwise.pdf>`_

To cite the code, then use::

    Prahl, S. (2023). pyspeckle: a python module for creation and analysis of laser speckle. (Version 0.5.1) https://doi.org/10.5281/zenodo.8311678

1D speckle
----------
.. image:: https://raw.githubusercontent.com/scottprahl/pyspeckle/master/docs/oneD_example.png
   :alt: 1D speckle plot

2D speckle
----------
.. image:: https://raw.githubusercontent.com/scottprahl/pyspeckle/master/docs/twoD_speckle.png
   :alt: 2D speckle plot

Documentation and examples for 1D, 2D, and 3D speckle are available at <https://pyspeckle2.readthedocs.io>

Installation
-------------

Use ``pip``::
    
    pip install pyspeckle

or use ``conda``::
    
    conda install -c conda-forge pyspeckle

License
-------

``pyspeckle`` is licensed under the terms of the MIT license.