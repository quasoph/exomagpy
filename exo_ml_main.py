import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import lightkurve

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

train_df_true = pd.read_table(r"C:\Users\ItIsO\Documents\GitHub\AMATEUR_exoplanet_data\UID_0013192_data_AXA_002.tbl") # has exoplanets
train_df_false = pd.read_table(r"C:\Users\ItIsO\Documents\GitHub\AMATEUR_exoplanet_data\UID_0013192_data_AXA_002.tbl") # has no exoplanets

test_df = input("Enter your data: ")

ls_imgs_true = []
ls_imgs_false = []

for x in train_df_true:
    # plot data with lightkurve
    # save as image
    # append image to list
    x=2

for y in train_df_false:
    # plot data with lightkurve
    # save as image
    # append image to list
    y=2 # just needed a space filler here

# export as images in folders

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
