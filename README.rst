.. |pypi-badge| image:: https://img.shields.io/pypi/v/pyspeckle?color=68CA66
   :target: https://pypi.org/project/pyspeckle/
   :alt: pypi

.. |github-badge| image:: https://img.shields.io/github/v/tag/scottprahl/pyspeckle?label=github&color=68CA66
   :target: https://github.com/scottprahl/pyspeckle
   :alt: github

.. |conda-badge| image:: https://img.shields.io/conda/v/conda-forge/pyspeckle?label=conda&color=68CA66
   :target: https://anaconda.org/conda-forge/pyspeckle
   :alt: conda-forge

.. |doi-badge| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8311677.svg
   :target: https://doi.org/10.5281/zenodo.8311677
   :alt: doi

.. |license| image:: https://img.shields.io/github/license/scottprahl/pyspeckle?color=68CA66
   :target: https://github.com/scottprahl/pyspeckle/blob/main/LICENSE.txt
   :alt: License

.. |test-badge| image:: https://github.com/scottprahl/pyspeckle/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/scottprahl/pyspeckle/actions/workflows/test.yaml
   :alt: Testing

.. |docs-badge| image:: https://readthedocs.org/projects/pyspeckle2/badge?color=68CA66
   :target: https://pyspeckle2.readthedocs.io
   :alt: Docs

.. |downloads-badge| image:: https://img.shields.io/pypi/dm/pyspeckle?color=68CA66
   :target: https://pypi.org/project/pyspeckle/
   :alt: Downloads

.. |lite| image:: https://img.shields.io/badge/try-JupyterLite-68CA66.svg
   :target: https://scottprahl.github.io/pyspeckle/
   :alt: Try Online


pyspeckle
=========

|pypi-badge| |github-badge| |conda-badge| |doi-badge|

|license| |test-badge| |docs-badge| |downloads-badge|

|lite|

**pyspeckle** is a Python library for generating and analyzing laser speckle fields.
It provides reproducible numerical implementations of physically motivated speckle models used in
optical metrology, coherent imaging, and biomedical photonics.

The methods implemented in this package are derived from Duncan & Kirkpatrick  
(*Algorithms for simulation of speckle (laser and otherwise)*, Proc. SPIE 6855, 2008).
These algorithms unify a variety of simulation approaches across:

- **objective speckle** (non-imaged fields),
- **subjective speckle** (imaged fields),
- **static speckle**, and
- **dynamic speckle** including translation, strain, boiling, and decorrelation.

Scientific Context
------------------

Coherent imaging systems—including SAR, OCT, ultrasound, ESPI, and laser speckle contrast
imaging—produce granular interference patterns defined by the random phase relationships of scattered waves.

Representative Outputs
----------------------

1D speckle
~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/scottprahl/pyspeckle/main/docs/images/oneD_example.png
   :alt: synthetic 1D speckle intensity profile

2D speckle
~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/scottprahl/pyspeckle/main/docs/images/twoD_speckle.png
   :alt: simulated 2D speckle field

Documentation
-------------

Full documentation and algorithm demonstrations are available at:

https://pyspeckle2.readthedocs.io

Try in JupyterLite (no install required):

https://scottprahl.github.io/pyspeckle/


Installation
------------

``pip``::

   pip install pyspeckle

``conda``::

   conda install -c conda-forge pyspeckle


Citation
--------

If you use ``pyspeckle`` in academic, instructional, or applied technical work, please cite:

Prahl, S. (2025). *pyspeckle: Tools for objective and subjective laser speckle analysis*
(Version 0.6.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.8311677


BibTeX
~~~~~~

.. code-block:: bibtex

   @software{pyspeckle_prahl_2025,
     author    = {Scott Prahl},
     title     = {pyspeckle: Tools for objective and subjective laser speckle analysis},
     year      = {2025},
     version   = {0.6.0},
     doi       = {10.5281/zenodo.8311677},
     url       = {https://github.com/scottprahl/pyspeckle},
     publisher = {Zenodo}
   }

License
-------

``pyspeckle`` is released under the MIT License. Contributions are welcome.
