.. _exoplanet machine learning:

predictExo(train1,size1,train2,size2,testfile,testsize)
====================================

Function to detect exoplanets using machine learning. Displays a lightcurve plot with positive or negative result for exoplanets for each input TIC ID.

train1: path or filename for a .csv file containing list of confirmed exoplanet TIC IDs.
size1: integer number of datapoints you want to use from exotrainfile.

train2: path or filename for a .csv file containing list of false alarms/false positive TIC IDs.
size2: integer number of datapoints you want to use from no_exotrainfile.

test: path or filename for a .csv file containing list of TIC IDs that you want to test.
testsize: integer number of datapoints you want to use from testfile.

Note: large size1, size2 and/or testsize will result in a longer processing time.

.. automodule:: exomagpy
    :members: