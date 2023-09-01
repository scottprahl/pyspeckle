pyspeckle
=========

by Scott Prahl

.. image:: https://img.shields.io/pypi/v/pyspeckle
   :target: https://pypi.org/project/pyspeckle/
   :alt: PyPI

.. image:: https://img.shields.io/conda/v/conda-forge/pyspeckle.svg
   :target: https://anaconda.org/conda-forge/pyspeckle
   :alt: Conda

.. image:: https://readthedocs.org/projects/pyspeckle2/badge
   :target: https://pyspeckle2.readthedocs.io
   :alt: Read the Docs

.. image:: https://img.shields.io/github/license/scottprahl/pyspeckle
   :target: https://github.com/scottprahl/pyspeckle/blob/master/LICENSE.txt
   :alt: MIT License

.. image:: https://zenodo.org/badge/99259684.svg
   :target: https://zenodo.org/badge/latestdoi/99259684

.. image:: https://github.com/scottprahl/miepython/actions/workflows/test.yml/badge.svg
   :target: https://github.com/scottprahl/miepython/actions/workflows/test.yml
   :alt: Testing

s.. image:: https://img.shields.io/pypi/dm/pyspeckle
   :target: https://pypi.org/project/pyspeckle/
   :alt: Number of PyPI downloads


________

A collection of routines to track and analyze laser speckle.  This is a python
port of SimSpeckle Matlab routines described in
`Duncan & Kirkpatrick, "Algorithms for simulation of speckle (laser and otherwise)," in SPIE Vol. 6855 (2008) <https://www.researchgate.net/profile/Sean-Kirkpatrick-2/publication/233783056_Algorithms_for_simulation_of_speckle_laser_and_otherwise/links/09e4150b78c4e8fe5f000000/Algorithms-for-simulation-of-speckle-laser-and-otherwise.pdf>`_

This implementation contains code for

    * 1D exponential and gaussian speckle 
    * 2D speckle algorithms
    * 3D speckle generation

Documentation and examples are available at <https://pyspeckle2.readthedocs.io>

Using pyspeckle
-------------------

1. Install with ``pip``::
    
    pip install pyspeckle

2. or `run this code in the cloud using Google Collaboratory <https://colab.research.google.com/github/scottprahl/pyspeckle/blob/master>`_ by selecting the Jupyter notebook that interests you.

3. use `binder <https://mybinder.org/v2/gh/scottprahl/pyspeckle/master?filepath=docs>`_ which will create a new environment that allows you to run Jupyter notebooks.  This takes a bit longer to start, but it automatically installs ``pyspeckle``.

4. clone the `pyspeckle github repository <https://github.com/scottprahl/pyspeckle>`_ and then add the repository to your ``PYTHONPATH`` environment variable


License
-------

pyspeckle is licensed under the terms of the MIT license.