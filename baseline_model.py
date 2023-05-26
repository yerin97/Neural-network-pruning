import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers import *
import sys

train_images = pickle.load(open('train_images.pkl', 'rb'))
train_labels = pickle.load(open('train_labels.pkl', 'rb'))
# load val
val_images = pickle.load(open('val_images.pkl', 'rb'))
val_labels = pickle.load(open('val_labels.pkl', 'rb'))


def define_model():
  # Define the neural network architecture (don't change this)

  model = models.Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5), input_shape=(25,25,3)))
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(1e-5)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5)))
  model.add(Activation('relu'))
  model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(1e-5)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(5))
  model.add(Activation('softmax'))

  return model

if __name__ == '__main__':
    model = define_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-6),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, batch_size=32, epochs=50, 
                    validation_data=(val_images, val_labels)) # train for 50 epochs, with batch size 32

    results = model.evaluate(val_images, val_labels, batch_size=128)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_baseline.h5")
    print("Saved model to disk")
