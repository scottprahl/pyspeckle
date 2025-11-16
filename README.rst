.. |pypi-badge| image:: https://img.shields.io/pypi/v/pyspeckle?color=68CA66
   :target: https://pypi.org/project/pyspeckle/
   :alt: pypi

.. |github-badge| image:: https://img.shields.io/github/v/tag/scottprahl/pyspeckle?label=github&color=68CA66
   :target: https://github.com/scottprahl/pyspeckle
   :alt: github

.. |conda-badge| image:: https://img.shields.io/conda/v/conda-forge/pyspeckle?label=conda&color=68CA66
   :target: https://anaconda.org/conda-forge/pyspeckle
   :alt: conda-forge

.. |doi-badge| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8312075.svg
   :target: https://doi.org/10.5281/zenodo.8312075
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

**pyspeckle** is a research-grade Python library for generating and analyzing laser speckle fields.  
It provides reproducible numerical implementations of physically motivated speckle models used in 
optical metrology, coherent imaging, and biomedical photonics.

The methods implemented in this package are derived from the algorithms of Duncan & Kirkpatrick  
(**"Algorithms for simulation of speckle (laser and otherwise)"**, Proc. SPIE 6855, 2008).
These algorithms unify a variety of simulation approaches across:

- **objective speckle** (non-imaged fields),
- **subjective speckle** (imaged fields),
- **static speckle**, and
- **dynamic speckle** including translation, strain, boiling, and decorrelation.

The goal of this project is to provide the research community with a reliable, transparent, 
and extensible computational reference for speckle simulation studies and validation of analytical models.

Scientific Context
------------------

Coherent imaging systems—including SAR, OCT, ultrasound, ESPI, and laser speckle contrast 
imaging—produce granular interference patterns defined by the random phase relationships of scattered waves.

The statistical properties of these patterns depend on:

- the nature of the scatterers,
- the system geometry (objective vs. subjective speckle),
- sampling in the optical transfer function domain,
- polarization state,
- mechanical motion (coordinated or uncoordinated),
- temporal evolution of scatterer phase correlations.

Using the FFT-based band-limited generation approach described by Duncan & Kirkpatrick, this library supports:

- fully developed speckle following exponential intensity statistics,
- partially polarized speckle fields parameterized by polarization degree,
- correlation-controlled temporal sequences via Gaussian copulas,
- controlled motion using the Fourier shift theorem,
- simulation of decorrelation consistent with analytical expectation (e.g., Airy-law decay in defocus imaging).

The resulting fields are suitable for testing theory, validating algorithms, benchmarking imaging systems, 
and training machine-learning models under controlled statistical conditions.

Representative Outputs
----------------------

1D speckle
~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/scottprahl/pyspeckle/main/docs/oneD_example.png
   :alt: synthetic 1D speckle intensity profile

2D speckle
~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/scottprahl/pyspeckle/main/docs/twoD_speckle.png
   :alt: simulated 2D speckle field

Documentation
-------------

Full documentation and algorithm demonstrations are available at:

   https://pyspeckle2.readthedocs.io

A browser-run JupyterLite environment requires no installation:

   https://scottprahl.github.io/pyspeckle/

Installation
------------

``pip``::

    pip install pyspeckle

or ``conda``::

    conda install -c conda-forge pyspeckle


Citation
--------

If you use ``pyspeckle`` in research or publication, please cite:

::

   Prahl, S. (2025). *pyspeckle: A Python module for creation and analysis of laser speckle.*
   Version 0.6.0. https://doi.org/10.5281/zenodo.8312075

::

    Duncan & Kirkpatrick (2008) "Algorithms for simulation of speckle (laser and otherwise)",
    Proc. of SPIE Vol. 6855, 685505.
    
License
-------

``pyspeckle`` is released under the MIT License. Contributions are welcome.

