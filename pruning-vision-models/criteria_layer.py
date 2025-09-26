import numpy as np
from numpy.linalg import matrix_rank
import copy
import time
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.utils import gen_batches
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.extmath import softmax
import gc

n_samples = 500

class CKA():
    __name__ = 'CKA'
    def __init__(self):
        pass

    def _debiased_dot_product_similarity_helper(self, xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n):
        return ( xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y) + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

    def feature_space_linear_cka(self, features_x, features_y, debiased=False):
        features_x = features_x - np.mean(features_x, 0, keepdims=True)
        features_y = features_y - np.mean(features_y, 0, keepdims=True)

        dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
        normalization_x = np.linalg.norm(features_x.T.dot(features_x))
        normalization_y = np.linalg.norm(features_y.T.dot(features_y))

        if debiased:
            n = features_x.shape[0]
            # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
            sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
            sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
            squared_norm_x = np.sum(sum_squared_rows_x)
            squared_norm_y = np.sum(sum_squared_rows_y)

            dot_product_similarity = self._debiased_dot_product_similarity_helper(
                dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
                squared_norm_x, squared_norm_y, n)
            normalization_x = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
                squared_norm_x, squared_norm_x, n))
            normalization_y = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
                squared_norm_y, squared_norm_y, n))

        return dot_product_similarity / (normalization_x * normalization_y)

    def scores(self,  model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        F = Model(model.input, model.get_layer(index=-2).output)
        features_F = F.predict(X_train[sub_sampling], verbose=0)

        F_line = Model(model.input, model.get_layer(index=-2).output)#TODO: Check if this is correct
       #It will probability not work for MobileNetV2 and other convolutional architectures
        for layer_idx in allowed_layers:
            # Resblock: Conv2d, Batch N., Activation, Conv2d, Batch N.
            #if isinstance(model.get_layer(index=layer_idx + self.layer_offset), BatchNormalization):
            #_layer = model.get_layer(index=layer_idx - 1)
            _layer = F_line.get_layer(index=layer_idx - 1)
            _w = _layer.get_weights()
            _w_original = copy.deepcopy(_w)

            for i in range(0, len(_w)):
                _w[i] = np.zeros(_w[i].shape)

            _layer.set_weights(_w)
            #F_line = Model(model.input, model.get_layer(index=-2).output)
            features_line = F_line.predict(X_train[sub_sampling], verbose=0)

            _layer.set_weights(_w_original)

            score = self.feature_space_linear_cka(features_F, features_line)
            output.append((layer_idx, 1 - score))

        return output

class rank():
    __name__ = 'HRank: Filter Pruning using High-Rank Feature Map, CVPR 2020'

    def __init__(self):
        pass

    def scores(self,  model, X_train=None, y_train=None, allowed_layers=[]):
        output = []
        for layer_idx in allowed_layers:

            layer = model.get_layer(index=layer_idx).output
            act = Activation('relu', name='feat{}'.format(layer_idx))(layer)

            tmp_model = Model(model.input, act)
            n_samples = X_train.shape[0]
            w, h, n_filters = tmp_model.output_shape[1:]
            scores = np.zeros((n_filters))

            for batch in gen_batches(n_samples, 32):
                #if self.preprocess_input is not None:
                 #   samples = self.preprocess_input(X_train[batch].astype(float))
                  #  features = tmp_model.predict(samples, batch_size=32)
                #else:
                features = tmp_model.predict(X_train[batch], batch_size=32, verbose=0)

                for filter in range(0, n_filters):
                    for feat in features:
                        scores[filter] += matrix_rank(feat[:, :, filter])

            scores = scores/X_train.shape[0]
            #We need to normalize by the size of the feature map (32x32) (16x16) (8, 8)
            scores = scores/max(w, h)
            #print('Layer [{}] Score[{:.4f}]'.format(i, np.mean(scores)), flush=True)
            output.append((layer_idx,np.mean(scores)))

        return output

class klDivergence():
    __name__ = 'Neural Network Pruning with Residual-Connections and Limited-Data, CVPR 2020'
    # Code adapted from https://github.com/Roll920/CURL

    def __init__(self):
        pass

    def bn_idx(self, model, layer_idx):
        idx = -1
        #Looking for the closest relu activatin
        for i in range(layer_idx, len(model.layers)):
            layer = model.get_layer(index=i)
            if isinstance(layer, BatchNormalization):
                idx = i
                break

        # if idx == -1:
        #     print('Problem to find BN layer', flush=True)

        return idx

    def zeroed_out(self, model, layer_idx):
        layer = model.get_layer(index=layer_idx)
        w = layer.get_weights()

        # Zeroed out the conv2d filter
        for i in range(0, len(w)):
            w[i] = np.zeros(w[i].shape)

        layer.set_weights(w)

        return None

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

       #n_samples = 256 Original Paper
        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        p = softmax(model.predict(X_train[sub_sampling], verbose=0))  # Softmax -- Logits

        unchaged_weights = model.get_weights()

        for layer_idx in allowed_layers:

            #'Removes' the layers. The weights are updated by reference
            self.zeroed_out(model, layer_idx-1)#i=Add i-1 is the batch index

            # y_pred = np.zeros((X_small_indices.shape[0], y_train.shape[1]))
            # for batch in gen_batches(X_small_indices.shape[0], 32):
            #     y_pred[batch] = model.predict(X_train[X_small_indices[batch]], verbose=0)

            q = softmax(model.predict(X_train[sub_sampling], verbose=0))

            # Compute KL Divergence -- See generate_mask.py line 129
            kl_loss = q * (np.log(q) - np.log(p))
            kl_loss = np.sum(kl_loss, axis=1)
            kl_loss = np.mean(kl_loss)

            # Restore the original weights (unpruned)
            model.set_weights(unchaged_weights)

            #print('Layer [{}] Score[{:.4f}]'.format(i, np.mean(kl_loss)), flush=True)
            output.append((layer_idx, kl_loss))

        return output

class expectedABS():
    __name__ = 'DropNet: Reducing Neural Network Complexity via Iterative Pruning John, ICML 2020'

    def __init__(self):
        pass

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []
        for layer_idx in allowed_layers:

            layer = model.get_layer(index=layer_idx).output
            act = Activation('relu', name='feat{}'.format(layer_idx))(layer)

            tmp_model = Model(model.input, act)
            n_samples = X_train.shape[0]
            w, h, n_filters = tmp_model.output_shape[1:]
            scores = np.zeros((n_filters))

            for batch in gen_batches(n_samples, 32):
                #if self.preprocess_input is not None:
                 #   samples = self.preprocess_input(X_train[batch].astype(float))
                 #   features = tmp_model.predict(samples, batch_size=32)
                #else:
                features = tmp_model.predict(X_train[batch], batch_size=32, verbose=False)

                for filter in range(0, n_filters):
                    for feat in features:
                        scores[filter] += np.sum(np.abs(feat[:, :, filter]))

            scores = scores/(w*h*X_train.shape[0])
            #print('Layer [{}] Score[{:.4f}]'.format(i, np.mean(scores)), flush=True)
            output.append((layer_idx, np.mean(scores)))

        return output

class random():
    def __init__(self):
        pass

    def scores(self,  model, X_train=None, y_train=None, allowed_layers=[]):
        output = [(x, np.random.rand()) for x in allowed_layers]

        return output

class L1():
    __name__ = 'Pruning Layers for Efficient ConvNets'

    def __init__(self):
        pass

    def compute_l1(self, weights):
        filter_w, filter_h, n_channels, n_filters =  weights[0].shape[0],  weights[0].shape[1], weights[0].shape[2], weights[0].shape[3]
        l1 = np.zeros((n_filters))
        for channel in range(0, n_channels):
            for filter in range(0, n_filters):
                kernel = weights[0][:, :, channel, filter]
                l1[filter] += np.sum(np.absolute(kernel))

        return l1

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []
        idx_Conv2D = 0
        for layer_idx in allowed_layers:

            layer = model.get_layer(index=layer_idx-2)
            weights = layer.get_weights()# weights have the format: w,h, channel, filters
            score1 = self.compute_l1(weights)

            layer = model.get_layer(index=layer_idx-5)
            weights = layer.get_weights()# weights have the format: w,h, channel, filters
            score2 = self.compute_l1(weights)
            
            output.append((layer_idx, np.mean(score1) + np.mean(score2)))
        #print(output)
        return output
    
class template_SimilarityMetric():
    __name__ = 'Template for implementing similarity metric criteria'
    def __init__(self):
        pass

    def metric(self, X, X_line):
        euclidian = np.linalg.norm(X-X_line, axis=1)
        return np.sum(euclidian)

    def scores(self,  model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        # n_samples = 256 Original Paper
        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        F = Model(model.input, model.get_layer(index=-2).output)
        features_F = F.predict(X_train[sub_sampling], verbose=0)

        F_line = Model(model.input, model.get_layer(index=-2).output)

        for layer_idx in allowed_layers:
            _layer = F_line.get_layer(index=layer_idx - 1)
            _w = _layer.get_weights()
            _w_original = copy.deepcopy(_w)

            for i in range(0, len(_w)):
                _w[i] = np.zeros(_w[i].shape)

            _layer.set_weights(_w)
            features_line = F_line.predict(X_train[sub_sampling], verbose=0)

            _layer.set_weights(_w_original)

            score = self.metric(features_F, features_line)
            output.append((layer_idx, 1 - score))

        return output

def criteria(method='random'):

    if method == 'rank':
        return rank()

    if method == 'expectedABS':
        return expectedABS()

    if method == 'klDivergence':
        return klDivergence()

    if method == 'random':
        return random()

    if method == 'CKA':
        return CKA()
    
    if method == 'L1':
        return L1()

    if method == 'template_SimilarityMetric':
        return template_SimilarityMetric()