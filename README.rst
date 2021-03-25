pyspeckle
=========

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/scottprahl/pyspeckle/blob/master

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scottprahl/pyspeckle/master?filepath=docs

.. image:: https://img.shields.io/badge/readthedocs-latest-blue.svg
   :target: https://pyspeckle2.readthedocs.io

.. image:: https://img.shields.io/badge/github-code-green.svg
   :target: https://github.com/scottprahl/pyspeckle

.. image:: https://img.shields.io/badge/MIT-license-yellow.svg
   :target: https://github.com/scottprahl/pyspeckle/blob/master/LICENSE.txt

__________

A collection of routines to track and analyze laser speckle.  This is a python
port of SimSpeckle Matlab routines described in
`Duncan & Kirkpatrick, "Algorithms for simulation of speckle (laser and otherwise)," in SPIE Vol. 6855 (2008) <https://www.researchgate.net/profile/Sean-Kirkpatrick-2/publication/233783056_Algorithms_for_simulation_of_speckle_laser_and_otherwise/links/09e4150b78c4e8fe5f000000/Algorithms-for-simulation-of-speckle-laser-and-otherwise.pdf>`_

This implementation contains code for

    * 1D exponential and gaussian speckle 
    * 2D speckle algorithms

Using pyspeckle
-------------------

1. Install with ``pip``::
    
    pip install --user pyspeckle

2. or `run this code in the cloud using Google Collaboratory <https://colab.research.google.com/github/scottprahl/pyspeckle/blob/master>`_ by selecting the Jupyter notebook that interests you.

3. use `binder <https://mybinder.org/v2/gh/scottprahl/pyspeckle/master?filepath=docs>`_ which will create a new environment that allows you to run Jupyter notebooks.  This takes a bit longer to start, but it automatically installs ``pyspeckle``.

4. clone the `pyspeckle github repository <https://github.com/scottprahl/pyspeckle>`_ and then add the repository to your ``PYTHONPATH`` environment variable


License
-------

pyspeckle is licensed under the terms of the MIT license.