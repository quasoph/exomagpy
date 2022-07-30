.. _tutorials:

Tutorials
====================================

Preparing data
-------------------

The functions in **exomagpy.predictExo** take in data in the form of a .csv file containing a list of target names.
For TESS targets, the TIC IDs are used. For Kepler targets, KIC IDs are used.
predictExo will read the contents of any column with the header "tic_id", "tid", "kic-id" or "kid".

Predicting exoplanets
-------------------------

**exomagpy** can predict the existence of exoplanets around a target star for most TESS and Kepler targets.
To predict exoplanet candidates, you can use the ``exomagpy.predictExo.tess()`` or ``exomagpy.predictExo.kepler()`` functions as below:

.. code-block:: python

    import exomagpy.predictExo
    exomagpy.predictExo.tess(train1,size1,train2,size2,test,testsize)
    exomagpy.predictExo.kepler(train1,size1,train2,size2,test,testsize)

Here, ``train1`` is a training dataset containing targets with confirmed exoplanets, and ``train2`` is a training dataset
containing targets with confirmed no exoplanets. ``size1`` and ``size2`` are the number of targets you wish to use from 
these training datasets respectively. ``test`` is the test dataset (the data you want to make predictions for)
and ``testsize`` is the number of targets to use for that dataset.

When the functions are run, the program will search for the specified targets using **Lightkurve**.
After the targets are found, the function's neural network will train using the lightcurves of the 
given train data. After training, the program will output the plot for each lightcurve (if running in Jupyter notebook) 
alongside its predicted exoplanetary status and ID.

Downloading lightcurves
--------------------------

**exomagpy** has a function ``lc_to_array``, which finds the lightcurve for an individual target and converts it into 
an 8-bit **numpy** image array. 

.. code-block:: python

    import exomagpy.download
    exomagpy.download.lc_to_array(search)

where ``search`` is a LightCurve object.