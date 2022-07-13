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

warnings.filterwarnings("ignore")

# DOWNLOAD (& ADD TO BIG ARRAY)

def download(search):

    lc = search.download()
    
    if lc is not None:
        
        fig,ax = plt.subplots()
        ax.scatter(lc.time.value.tolist(), lc.flux.value.tolist(),s=0.1, color='k')
        ax.autoscale()
        ax.set_xlabel('Time (BTJD)')
        ax.set_ylabel('Flux')
        plt.close(fig)
        io_buf = io.BytesIO()
        fig.savefig(io_buf,format="raw")
        io_buf.seek(0)
        img_arr = (np.frombuffer(io_buf.getvalue(),dtype=np.uint8)).reshape(288,-1)
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
        
    return pics, shape

# CREATE TRAIN AND TEST DATASETS

def predictExo(exotrainfile,size1,no_exotrainfile,size2,testfile,testsize):

    exotraindata, trainshape = get_lightcurves(exotrainfile,size1) # / 255 for the data
    noexotraindata, train2shape = get_lightcurves(no_exotrainfile,size2)

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

    Y, testshape = get_lightcurves(testfile,testsize) # this returns an array of images!
    Y = np.asarray(Y)

    tble = pd.read_csv(os.path.abspath(testfile),delimiter=",",comment="#")
    TICid = tble["tic_id"].astype(str)

    col_names = tble.columns.values.tolist()
    if "tid" in col_names:
        TICid = tble["tid"].astype(str)
    elif "tic_id" in col_names:
        TICid = tble["tic_id"].astype(str)
    else:
        print("No TIC ID column found.")
    
    X = tensorflow.reshape(Y,[testshape,1,288,1728])
    probability = model.predict(X)
    val = np.round(probability).tolist()
    
    for x in range(0,len(val)):
        
        fig, ax = plt.subplots()
        ax.imshow(Y[x],aspect=4,cmap="gray")
        plt.show()

        if val[x] == [1.0]:
            print("Exoplanet candidate detected! ID: " + TICid[x])
        elif val[x] == [0.0]:
            print("No exoplanet detected. ID: " + TICid[x])