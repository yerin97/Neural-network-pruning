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

def thinet_prune(model, prune_ratio):
    # Create dictionary to store each layer's L1-norm and shape
    layer_l1 = {}
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            weights = layer.get_weights()[0]
            l1_norms = np.sum(np.abs(weights), axis=(0, 1, 2))
            layer_l1[layer.name] = {'l1_norms': l1_norms, 'shape': weights.shape}
        elif isinstance(layer, layers.Dense):
            weights = layer.get_weights()[0]
            l1_norms = np.sum(np.abs(weights), axis=0)
            layer_l1[layer.name] = {'l1_norms': l1_norms, 'shape': weights.shape}

    # Calculate threshold value for pruning
    total_l1_norm = sum(np.concatenate([layer_l1[name]['l1_norms'] for name in layer_l1]))
    target_pruned_norm = total_l1_norm * prune_ratio * 0.1
    pruned_norm = 0

    pruned_layers = []
    for name in layer_l1.keys():
      mask_default = np.array([0 for i in range(len(layer_l1[name]['l1_norms']))])
      pruned_layers.append({"name":name,"mask":mask_default,"pruned_idx":[]})
    print(total_l1_norm, target_pruned_norm)

    while pruned_norm < target_pruned_norm:
      for i, name in enumerate(layer_l1.keys()):
        if pruned_norm > target_pruned_norm: 
          break
        l1_norms = layer_l1[name]['l1_norms']
          #shape = layer_l1[name]['shape']
        if isinstance(model.get_layer(name), layers.Conv2D):
            #num_filters = shape[3]
            if len(pruned_layers[i]['pruned_idx']) == 0:
              idx = np.argsort(l1_norms)[0]
            else:
              sorted = np.argsort(l1_norms)
              #print(sorted)
              latest = list(sorted).index(pruned_layers[i]['pruned_idx'][-1])
              if latest+1 >= len(l1_norms): continue
              idx = sorted[latest+1]
            threshold = l1_norms[idx] # smallest
            mask = np.where(l1_norms >= threshold, 1, 0)
            pruned_norm += threshold
            pruned_layers[i]['name'] = name
            pruned_layers[i]['mask'] = mask
            pruned_layers[i]['pruned_idx'].append(idx)

        elif isinstance(model.get_layer(name), layers.Dense):
            #num_filters = shape[3]
            if len(pruned_layers[i]['pruned_idx']) == 0:
              idx = np.argsort(l1_norms)[0]
            else:
              sorted = np.argsort(l1_norms)
              #print(sorted)
              latest = list(sorted).index(pruned_layers[i]['pruned_idx'][-1])
              if latest+1 >= len(l1_norms): continue
              idx = sorted[latest+1]
            threshold = l1_norms[idx] # smallest
            mask = np.where(l1_norms >= threshold, 1, 0)
            pruned_norm += threshold
            pruned_layers[i]['name'] = name
            pruned_layers[i]['mask'] = mask
            pruned_layers[i]['pruned_idx'].append(idx)

   # print(pruned_norm)
    #print(pruned_layers)
    # Apply the pruning mask to the model
    new_weights = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            name = layer.name
            mask = pruned_layers[[l['name'] for l in pruned_layers].index(name)]['mask']
            weights = layer.get_weights()
            new_kernel = np.zeros_like(weights[0])
            new_bias = weights[1]
            for i in range(new_kernel.shape[3]):
                new_kernel[:, :, :, i] = weights[0][:, :, :, i] * mask[i]
            new_weights.append(new_kernel)
            new_weights.append(new_bias)
        elif isinstance(layer, layers.Dense):
            name = layer.name
            mask = pruned_layers[[l['name'] for l in pruned_layers].index(name)]['mask']
            weights = layer.get_weights()
            new_kernel = weights[0] * mask.reshape((1, -1))
            new_bias = weights[1]
            new_weights.append(new_kernel)
            new_weights.append(new_bias)
        else:
            new_weights.extend(layer.get_weights())

    model.set_weights(new_weights)

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
  
  pruned_model = thinet_prune(model, sparsity)
  pruned_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-6),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

  history = pruned_model.fit(train_images, train_labels, batch_size=32, epochs=50,
                      validation_data=(val_images, val_labels)) # train for 50 epochs, with batch size 32

  print("-----ThiNet-----")
  # evaluate again to see how the accuracy changes
  results = pruned_model.evaluate(val_images, val_labels, batch_size=128)
  # you need to save the model's weights, naming it 'my_model_weights.h5'
  pruned_model.save_weights(f"thinet_{sparsity_str}.h5")
