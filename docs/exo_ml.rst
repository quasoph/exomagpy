.. _exoplanet machine learning:

predictExo(exotrainfile,size1,no_exotrainfile,size2,testfile,testsize)
====================================

Function to detect exoplanets using machine learning.

exotrainfile: path or filename for a .csv file containing list of confirmed exoplanet TIC IDs.
size1: integer number of datapoints you want to use from exotrainfile.

no_exotrainfile: path or filename for a .csv file containing list of false alarms/false positive TIC IDs.
size2: integer number of datapoints you want to use from no_exotrainfile.

testfile: path or filename for a .csv file containing list of TIC IDs that you want to test.
testsize: integer number of datapoints you want to use from testfile.

.. automodule:: exomagpy
    :members: