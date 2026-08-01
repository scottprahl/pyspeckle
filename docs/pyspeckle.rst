API for `pyspeckle` package
===========================

Every function below is re-exported at the top level, so
``pyspeckle.create_exponential(...)`` works regardless of which module
defines it.

``core`` generates speckle and analyzes it, ``noise`` generates the correlated
Gaussian fields that are *not* speckle, and ``plots`` draws them.

.. automodapi:: pyspeckle.core
   :no-inheritance-diagram:

.. automodapi:: pyspeckle.noise
   :no-inheritance-diagram:

.. automodapi:: pyspeckle.plots
   :no-inheritance-diagram:
