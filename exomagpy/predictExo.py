# IMPORTS

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
import lightkurve as lk
import io
import warnings

from tensorflow import keras

from .get_lightcurves import get_lightcurves_kep
from .get_lightcurves import get_lightcurves_jwst
from .get_lightcurves import get_lightcurves

warnings.filterwarnings("ignore")

# CREATE TRAIN AND TEST DATASETS

def tess(train1,size1,train2,size2,test,testsize):

    exotraindata, trainshape = get_lightcurves(train1,size1) # / 255 for the data
    noexotraindata, train2shape = get_lightcurves(train2,size2)

    exotraindata = np.asarray(exotraindata)
    noexotraindata = np.asarray(noexotraindata)

    print(exotraindata[0])

    print(np.shape(exotraindata))

    exolabels = np.ones(trainshape)
    noexolabels = np.zeros(train2shape)

    traindata = np.concatenate((exotraindata,noexotraindata))
    trainlabels = np.concatenate((exolabels,noexolabels))

    print("Train data shape is " + str(np.shape(traindata)))
    traindata = tensorflow.reshape(traindata,[trainshape+train2shape,1,288,1728])

    # NON-2D CNN MODEL

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1,288,1728)),
        keras.layers.Dense(16,activation="relu"),
        keras.layers.Dense(16,activation="relu"),
        keras.layers.Dense(1,activation="sigmoid")
    ])

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    # TRAIN DATA

    model.fit(
        traindata,
        trainlabels,
        batch_size = 8,
        epochs = trainshape+train2shape,
    )

    Y, testshape = get_lightcurves(test,testsize) # this returns an array of images!
    Y = np.asarray(Y)

    tble = pd.read_csv(os.path.abspath(test),delimiter=",",comment="#",chunksize=5)

    tble.__next__()

    col_names = tble.columns.values.tolist()
    if "tid" in col_names:
        TICid = tble["tid"].astype("category")
    elif "tic_id" in col_names:
        TICid = (tble["tic_id"].astype(str).str[4:]).astype("category")
    else:
        print("No TIC ID column found.")
    
    X = tensorflow.reshape(Y,[testshape,1,288,1728])
    probability = model.predict(X)
    val = np.round(probability).tolist()
    
    def fig_gen():
        for x in range(0,len(val)):
            yield x

    figures = fig_gen()
    for i in figures:
        fig, ax = plt.subplots()
        ax.imshow(Y[i],aspect=4,cmap="gray")
        plt.show()

        if val[i] == [1.0]:
            print("Exoplanet candidate detected! ID: " + str(TICid[i]))
        elif val[i] == [0.0]:
            print("No exoplanet detected. ID: " + str(TICid[i]))

def kepler(train1,size1,train2,size2,test,testsize):

    exotraindata, trainshape = get_lightcurves_kep(train1,size1) # / 255 for the data
    noexotraindata, train2shape = get_lightcurves_kep(train2,size2)

    exotraindata = np.asarray(exotraindata)
    noexotraindata = np.asarray(noexotraindata)

    print(exotraindata[0])

    print(np.shape(exotraindata))

    exolabels = np.ones(trainshape)
    noexolabels = np.zeros(train2shape)

    traindata = np.concatenate((exotraindata,noexotraindata))
    trainlabels = np.concatenate((exolabels,noexolabels))

    print("Train data shape is " + str(np.shape(traindata)))
    traindata = tensorflow.reshape(traindata,[trainshape+train2shape,1,288,1728])

    # NON-2D CNN MODEL

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1,288,1728)),
        keras.layers.Dense(16,activation="relu"),
        keras.layers.Dense(16,activation="relu"),
        keras.layers.Dense(1,activation="sigmoid")
    ])

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    # TRAIN DATA

    model.fit(
        traindata,
        trainlabels,
        batch_size = 8,
        epochs = trainshape+train2shape,
    )

    Y, testshape = get_lightcurves_kep(test,testsize) # this returns an array of images!
    Y = np.asarray(Y)

    tble = pd.read_csv(os.path.abspath(test),delimiter=",",comment="#",chunksize=5)

    tble.__next__()

    col_names = tble.columns.values.tolist()
    if "kid" in col_names:
        KICid = tble["kid"].astype("category")
    elif "kic_id" in col_names:
        KICid = (tble["kic_id"].astype(str).str[4:]).astype("category")
    else:
        print("No KIC ID column found.")
    
    X = tensorflow.reshape(Y,[testshape,1,288,1728])
    probability = model.predict(X)
    val = np.round(probability).tolist()
    
    def fig_gen():
        for x in range(0,len(val)):
            yield x

    figures = fig_gen()
    for i in figures:
        fig, ax = plt.subplots()
        ax.imshow(Y[i],aspect=4,cmap="gray")
        plt.show()

        if val[i] == [1.0]:
            print("Exoplanet candidate detected! ID: " + str(KICid[i]))
        elif val[i] == [0.0]:
            print("No exoplanet detected. ID: " + str(KICid[i]))

# CREATE TRAIN AND TEST DATASETS

def jwst(train1,size1,train2,size2,test,testsize):

    exotraindata, trainshape = get_lightcurves_jwst(train1,size1) # / 255 for the data
    noexotraindata, train2shape = get_lightcurves_jwst(train2,size2)

    exotraindata = np.asarray(exotraindata)
    noexotraindata = np.asarray(noexotraindata)

    print(exotraindata[0])

    print(np.shape(exotraindata))

    exolabels = np.ones(trainshape)
    noexolabels = np.zeros(train2shape)

    traindata = np.concatenate((exotraindata,noexotraindata))
    trainlabels = np.concatenate((exolabels,noexolabels))

    print("Train data shape is " + str(np.shape(traindata)))
    traindata = tensorflow.reshape(traindata,[trainshape+train2shape,1,288,1728])

    # NON-2D CNN MODEL

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1,288,1728)),
        keras.layers.Dense(16,activation="relu"),
        keras.layers.Dense(16,activation="relu"),
        keras.layers.Dense(1,activation="sigmoid")
    ])

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    # TRAIN DATA

    model.fit(
        traindata,
        trainlabels,
        batch_size = 8,
        epochs = trainshape+train2shape,
    )

    Y, testshape = get_lightcurves_jwst(test,testsize) # this returns an array of images!
    Y = np.asarray(Y)

    tble = pd.read_csv(os.path.abspath(test),delimiter=",",comment="#",chunksize=5)

    tble.__next__()

    col_names = tble.columns.values.tolist()
    if "tid" in col_names:
        TICid = tble["tid"].astype("category")
    elif "tic_id" in col_names:
        TICid = (tble["tic_id"].astype(str).str[4:]).astype("category")
    else:
        print("No TIC ID column found.")
    
    X = tensorflow.reshape(Y,[testshape,1,288,1728])
    probability = model.predict(X)
    val = np.round(probability).tolist()
    
    def fig_gen():
        for x in range(0,len(val)):
            yield x

    figures = fig_gen()
    for i in figures:
        fig, ax = plt.subplots()
        ax.imshow(Y[i],aspect=4,cmap="gray")
        plt.show()

        if val[i] == [1.0]:
            print("Exoplanet candidate detected! ID: " + str(TICid[i]))
        elif val[i] == [0.0]:
            print("No exoplanet detected. ID: " + str(TICid[i]))