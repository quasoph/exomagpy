.. _tutorials:

Tutorials
====================================

Predicting exoplanets
-------------------------

**exomagpy** can predict the existence of exoplanets around a target star for most TESS and Kepler targets.
To predict exoplanet candidates, you can use the `exomagpy.predictExo.tess()` or `exomagpy.predictExo.kepler()` functions as below:

.. code-block:: python
    
    import exomagpy.predictExo
    exomagpy.predictExo.tess()
    exomagpy.predictExo.kepler()