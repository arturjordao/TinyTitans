import os
import sys
import time
import argparse

import numpy as np 
import tensorflow as tf
import custom_callbacks
import custom_functions as func
import rebuild_layers as rl
import rebuild_filters as rf
import criteria_filter as cf
import criteria_layer as cl

from datetime import datetime

import tensorflow as tf
import keras.backend as K

from tensorflow import keras
from keras.layers import *
from keras.activations import *
from tensorflow.data import Dataset

from sklearn.utils import gen_batches
from sklearn.metrics._classification import accuracy_score

X_train, y_train, X_test, y_test = func.cifar_resnet_data(debug=False)

architecture_name = 'ResNet32'

rf.architecture_name = architecture_name
rl.architecture_name = architecture_name

def pruneByLayer(model, criteria, p_layer):
    allowed_layers = rl.blocks_to_prune(model)
    layer_method = cl.criteria(criteria)
    scores = layer_method.scores(model, X_train, y_train, allowed_layers)    
    
    return rl.rebuild_network(model, scores, p_layer)

def pruneByFilter(model, criteria, p_filter):
    allowed_layers_filters = rf.layer_to_prune_filters(model)
    numberToFilterToRemove = int(func.count_filters(model)*p_filter)
    filter_method = cf.criteria(criteria)
    scores = filter_method.scores(model, X_train, y_train, allowed_layers_filters)    
    
    return  rf.rebuild_network(model, scores, p_filter, numberToFilterToRemove)
             
def statistics(model):
    n_params = model.count_params()
    
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
    n_filters = func.count_filters(model)
    flops, _ = func.compute_flops(model)
    blocks = rl.count_blocks(model)

    memory = func.memory_usage(1, model)

    print('Accuracy [{}] Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
        'Memory [{:.6f}]'.format(acc, blocks, n_params, n_filters, flops, memory), flush=True)

