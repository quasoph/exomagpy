.. _tutorials:

Tutorials
====================================

.. predicting exoplanets:

Predicting exoplanets
-------------------------

**exomagpy** can predict the existence of exoplanets around a target star for most TESS and Kepler targets.
To predict exoplanet candidates, you can use the `exomagpy.predictExo.tess()` or `exomagpy.predictExo.kepler()` functions as below:

.. code-block:: python
    import exomagpy.predictExo
    exomagpy.predictExo.tess(train1,size1,train2,size2,test,testsize)
    exomagpy.predictExo.kepler(train1,size1,train2,size2,test,testsize)