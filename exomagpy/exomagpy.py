# IMPORTS

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
import lightkurve as lk
import cv2
import io
import warnings
from sklearn.model_selection import train_test_split
import sys


from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

warnings.filterwarnings("ignore") # didnt work
np.set_printoptions(threshold=sys.maxsize)

# DOWNLOAD (& ADD TO BIG ARRAY)

def download(search):

    lc = search.download()
    
    if lc is not None:
        
        fig,ax = plt.subplots()
        ax.scatter(lc.time.value.tolist(), lc.flux.value.tolist(), color='k')
        ax.autoscale()
        ax.set_xlabel('Time (BTJD)')
        ax.set_ylabel('Flux')
        fig.show()
        io_buf = io.BytesIO()
        fig.savefig(io_buf,format="raw")
        io_buf.seek(0)
        img_arr = np.frombuffer(io_buf.getvalue(),dtype=np.uint8)
        io_buf.close()
    
        return img_arr

# GET ARRAY OF IMAGE ARRAYS FOR A FILE

def get_lightcurves(filename,length):

    tbl = pd.read_csv(os.path.abspath(filename),delimiter=",",comment="#")
    
    colnames = tbl.columns.values.tolist()
    if "tid" in colnames:
        TICs = tbl["tid"].astype(str)
    elif "tic_id" in colnames:
        TICs = tbl["tic_id"].astype(str).str[4:]
    else:
        print("No TIC ID column found.")

    #print(np.shape(TICs))

    pics = []

    for x in range(0,length): # change upper bound as needed
        name = TICs[x]
        
        search = lk.search_lightcurve(target=("TIC " + name),author="SPOC")
        pic = download(search)
        
        if pic is not None:
            pics.append(pic)
    
    shape = int(len(pics))

    print("Shape is " + str(shape))
        
    return pics, shape

# CREATE TRAIN AND TEST DATASETS

def predictExo(exotrainfile,noexotrainfile,testfile):

    exotraindata, trainshape = get_lightcurves(exotrainfile,200) # / 255 for the data
    noexotraindata, train2shape = get_lightcurves(noexotrainfile,69)

    exotraindata = np.asarray(exotraindata)
    noexotraindata = np.asarray(noexotraindata)

    print(exotraindata[0])

    print(np.shape(exotraindata))

    exolabels = np.ones(trainshape)
    noexolabels = np.zeros(train2shape)

    traindata = np.concatenate((exotraindata,noexotraindata))
    trainlabels = np.concatenate((exolabels,noexolabels))

    #train_exo, test_exo, train_labels, test_labels = train_test_split(traindata,trainlabels)

    traindata = tensorflow.reshape(traindata,[trainshape+train2shape,1,1228800])

    # NON-2D CNN MODEL

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1,1228800)),
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

    Y, testshape = get_lightcurves(testfile,10) # this returns an array of images!
    Y = np.asarray(Y)

    tble = pd.read_csv(os.path.abspath(testfile),delimiter=",",comment="#")
    TICid = tble["tic_id"].astype(str)
    
    X = tensorflow.reshape(Y,[testshape,1,1228800])
    probability = model.predict(X)
    val = np.argmax(probability,axis=1).tolist()
    
    for x in range(0,len(val)):
        
        print(x)
        print(val[x])

        if val[x] == 1:
            print("Exoplanet detected!" + TICid[x])
        elif val[x] == 0:
            print("No exoplanet detected." + TICid[x])


predictExo("PS_2022.06.27_08.12.38.csv","TOI_2022.06.29_08.07.35.csv","PS_2022.07.04_02.32.20.csv")