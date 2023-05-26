import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers import *
from keras.models import model_from_json
import sys

train_images = pickle.load(open('train_images.pkl', 'rb'))
train_labels = pickle.load(open('train_labels.pkl', 'rb'))
# load val
val_images = pickle.load(open('val_images.pkl', 'rb'))
val_labels = pickle.load(open('val_labels.pkl', 'rb'))


def network_slimming(model, sparsity_level):
  layer_idx = 0
  for layer in model.layers:
    if isinstance(layer, layers.Conv2D): # or isinstance(layer, layers.Dense):
      weights = model.layers[layer_idx].get_weights()
      # Importance score = L1 Norm of each layer
      importance_scores = np.abs(model.layers[layer_idx].get_weights()[0]).sum(axis=(0, 1, 2)) 
      # predetermined and fixed number of neurons will be removed 
      num_neurons_to_prune = int(sparsity_level * len(importance_scores)) 
      pruned_neurons = np.argsort(importance_scores)[:num_neurons_to_prune]
      mask = np.ones(weights[0].shape)
      mask[:, :, :, pruned_neurons] = 0
      pruned_weights = np.copy(weights)
      pruned_weights[0] *= mask
      model.layers[layer_idx].set_weights(pruned_weights)
      
    elif isinstance(layer, layers.Dense):
      weights = model.layers[layer_idx].get_weights()
      # Importance score = L1 Norm of each layer
      importance_scores = np.abs(model.layers[layer_idx].get_weights()[0]).sum(axis=0) 
      # predetermined and fixed number of neurons will be removed 
      num_neurons_to_prune = int(sparsity_level * len(importance_scores)) 
      pruned_neurons = np.argsort(importance_scores)[:num_neurons_to_prune]
      mask = np.ones(weights[0].shape)
      mask[:, pruned_neurons] = 0
      pruned_weights = np.copy(weights)
      pruned_weights[0] *= mask
      model.layers[layer_idx].set_weights(pruned_weights)
    layer_idx += 1
  return model



if __name__ == '__main__':
  sparsity = float(sys.argv[1])
  sparsity_str = str(sys.argv[1])

  print("-----Baseline-----")
  # load json and create model
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights("model_baseline.h5")
  print("Loaded model from disk")
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-6),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

  results = model.evaluate(val_images, val_labels, batch_size=128)
  
  pruned_model = network_slimming(model, sparsity)
  pruned_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-6),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

  history = pruned_model.fit(train_images, train_labels, batch_size=32, epochs=50,
                      validation_data=(val_images, val_labels)) # train for 50 epochs, with batch size 32

  print("-----Network Slimming-----")
  # evaluate again to see how the accuracy changes
  results = pruned_model.evaluate(val_images, val_labels, batch_size=128)
  # you need to save the model's weights, naming it 'my_model_weights.h5'
  pruned_model.save_weights(f"network_slimming_{sparsity_str}.h5")

