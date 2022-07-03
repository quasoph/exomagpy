# IMPORTS

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
import lightkurve as lk

from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# PHASE AND BIN FUNCTION

def phase_and_bin(data,period,t0,bin_time):
    data_phased = data.fold(period=period,epoch_time=t0)
    data_binned = data_phased.bin(bin_time)
    return data_binned

# CONVERT PLOT TO ARRAY FUNCTION

def plot_to_array(pltdata):

    lc = pltdata.plot(linewidth=0,marker=".")
    img = np.asarray(bytearray(lc.read()),dtype=np.uint8)
    
    return img

# GET ARRAY OF IMAGE ARRAYS FOR A FILE

def get_lightcurves(filename):

    tbl = pd.read_csv(os.path.abspath(str(filename)),delimiter=",",comment="#")
    TICs = tbl["tid"]

    x = 0

    imgarray = []

    while x < len(TICs):
        name = TICs[x]
        try:
            search = lk.search_lightcurve("TIC " + str(name),author="SPOC",sector=1)
            lc = search.download()
            array = plot_to_array(lc)
            imgarray.append(array)
        except:
            search = lk.search_lightcurve("TIC " + str(name),author="TESS",sector=1)
            lc = search.download()
            array = plot_to_array(lc)
            imgarray.append(array)
        else:
            print("No lightcurve found.")
    
        x = x + 1

        return imgarray

# CREATE TRAIN AND TEST DATASETS

def predictExo(exotrainfile,noexotrainfile,testfile):

    train = ImageDataGenerator(rescale=1/255)
    test = ImageDataGenerator(rescale=1/255)
    batchsize = 7

    train_ds = train.flow(
        get_lightcurves(exotrainfile),
        get_lightcurves(noexotrainfile),
        target_size=(150,150),
        batch_size = batchsize,
        class_mode = 'binary')

    test_ds = test.flow(
        get_lightcurves(testfile),
        target_size=(150,150),
        batch_size = batchsize,
        class_mode = 'binary')

    # BUILD CNN MODEL

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Conv2D(64,(3,3),activation="relu"))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Conv2D(128,(3,3),activation="relu"))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Conv2D(128,(3,3),activation="relu"))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(512,activation="relu"))

    model.add(keras.layers.Dense(1,activation="sigmoid"))

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    # TRAIN DATA

    model.fit(
        train_ds,
        steps_per_epoch = 250,
        epochs = 10,
        validation_data = test_ds
    )

    for y in testfile:

        Y = get_lightcurves(testfile[y])
        pic = Image.fromarray(Y)
        pic.show()
        X = np.expand_dims(Y,axis=0)

        val = model.predict(X)
        print(val)

        if val == 1:
            plt.xlabel("Exoplanet detected!",fontsize=30)
        elif val == 0:
            plt.xlabel("No exoplanet detected.",fontsize=30)