# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import lightkurve as lk

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# FETCH DATA

train_df_true = lk.search_lightcurve("Trappist-1",radius=180.,campaign=12,exptime=1800) # has exoplanets
train_df_false = lk.search_lightcurve("Trappist-1",radius=180.,campaign=12,exptime=1800) # has no exoplanets

lc_exo = train_df_true.download()
lc_no_exo = train_df_false.download()

test_df = str(input("Enter your target: "))
test_df1 = str(input("Enter mission: "))
testsearch = lk.search_lightcurve(test_df,author=test_df1)
lc_test = testsearch.download()

ls_imgs_true = []
ls_imgs_false = []

#for x in lc_exo:
    #plt.plot(x)
    #plt.savefig("Lightcurve_Exo_" + str(lc_exo.index(x)) + ".png")
    # export to folder

#for y in lc_no_exo:
    #plt.plot(y)
    #plt.savefig("Lightcurve_No_Exo_" + str(lc_no_exo.index(y)) + ".png")
    # export to folder

#for z in lc_test:
    #plt.plot(z)
    #plt.savefig("Lightcurve_Test_" + str(lc_test.index(z)) + ".png")
    # export to folder

lc_exo.plot()
plt.show()

# EXPORT LIGHTKURVE GRAPHS AS IMAGES IN 2 FOLDERS (EXOPLANETS AND NO EXOPLANETS) IN A LARGER SHARED FOLDER
# CREATE LIGHTKURVE GRAPHS FOR TEST_DF AND EXPORT TO A "TEST" FOLDER

# CREATE TRAIN AND TEST DATASETS

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_ds = train.flow_from_directory(
    "IMAGE FOLDER PATH",
    target_size=(150,150),
    batch_size = 32,
    class_mode = 'binary')

test_ds = test.flow_from_directory(
    "IMAGE FOLDER PATH",
    target_size=(150,150),
    batch_size = 32,
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

model.fit_generator(
    train_ds,
    steps_per_epoch = 250,
    epochs = 10,
    validation_data = test_ds
)

# PREDICTIONS (FUNCTION)

def predictExo(filename):
    img = image.load_img(filename,target_size=(150,150))
    plt.imshow(img)

    Y = image.img_to_array(img)
    X = np.expand_dims(Y,axis=0)

    val = model.predict(X)
    print(val)

    if val == 1:
        plt.xlabel("Exoplanet detected!",fontsize=30)
    elif val == 0:
        plt.xlabel("No exoplanet detected.",fontsize=30)

predictExo(lc_test)