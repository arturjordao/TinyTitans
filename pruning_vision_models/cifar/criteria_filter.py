import numpy as np
import copy
import time
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import os.path
import sys

from numpy.linalg import matrix_rank
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.extmath import softmax
from sklearn.utils import gen_batches

#architecture_name = 'ResNet'
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

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:#It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        F = Model(model.input, model.get_layer(index=-2).output)
        F_features = F.predict(X_train[sub_sampling], verbose=0)

        F_line = Model(model.input, model.get_layer(index=-2).output)
        for layer_idx in allowed_layers:
            scores = []

            layer = F_line.get_layer(index = layer_idx)
            n_filters = layer.filters
            for f in range(n_filters):
                weights = layer.get_weights()
                original_weights = copy.deepcopy(weights)

                weights[0][:, :, :, f] = 0
                weights[1][f] = 0
                layer.set_weights(weights)

                #F_line = Model(model.input, model.get_layer(index=-2).output)
                F_line_features = F_line.predict(X_train[sub_sampling], verbose=0)

                layer.set_weights(original_weights)

                score = self.feature_space_linear_cka(F_features, F_line_features)
                scores.append(1-score)

            output.append((layer_idx, scores))

        return output

class L1():
    __name__ = 'Pruning Filters for Efficient ConvNets'

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

            layer = model.get_layer(index=layer_idx)

            weights = layer.get_weights()# weights have the format: w,h, channel, filters
            score = self.compute_l1(weights)

            output.append((layer_idx, score))

        return output

class random():
    __name__ = 'Random Pruning'

    def __init__(self,):
        pass

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        for idx in allowed_layers:
            num_filters = model.get_layer(index=idx).filters
            scores = np.random.rand(num_filters)
            output.append((idx, scores))

        return output

class rank():
    __name__ = 'HRank: Filter Pruning using High-Rank Feature Map, CVPR 2020'

    def __init__(self):
        pass

    def relu_idx(self, model, layer_idx):
        idx = -1
        #Looking for the closest relu activatin
        for i in range(layer_idx, len(model.layers)):
            layer = model.get_layer(index=i)
            if isinstance(layer, Activation):
                idx = i
                break

        #The activation in VGG16 is inside Conv2D
        if idx == -1:
           idx = layer_idx

        return idx

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        #n_samples = 500
        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        for layer_idx in allowed_layers:

            idx = self.relu_idx(model, layer_idx)
            feature_maps = Model(model.input, model.get_layer(index=idx).output).predict(X_train[sub_sampling], batch_size=32, verbose=False)
            n_filters = feature_maps.shape[-1]
            ranking = np.zeros((n_filters))

            for sample in feature_maps:
                for filter in range(0, n_filters):
                    ranking[filter] = ranking[filter] + matrix_rank(sample[:, :, filter])

            ranking = ranking/feature_maps.shape[0]

            output.append((layer_idx, ranking))

        return output

class klDivergence():
    __name__ = 'Neural Network Pruning with Residual-Connections and Limited-Data'

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

        return idx

    def zeroed_out(self, model, layer_idx, filter_idx):
        layer = model.get_layer(index=layer_idx)
        w = layer.get_weights()

        # Zeroed out the conv2d filter
        w[0][:, :, :, filter_idx] = np.zeros(w[0].shape[0:-1])
        w[1][filter_idx] = 0
        layer.set_weights(w)

        #Find the index of the BN layer based on layer_idx
        layer_idx = self.bn_idx(model, layer_idx)
        layer = model.get_layer(index=layer_idx)

        #VGG16 on ImageNet224x224 does not contain BN layers.
        if isinstance(layer, BatchNormalization):
            # Zeroed out the batch norm filter
            w = layer.get_weights()
            w[0][filter_idx] = 0
            w[1][filter_idx] = 0
            w[2][filter_idx] = 0
            w[3][filter_idx] = 0
            layer.set_weights(w)

        return None

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        #n_samples = 256 # Original paper subsample is 256
        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        p = softmax(model.predict(X_train[sub_sampling], verbose=False))  # Softmax -- Logits
        unchaged_weights = model.get_weights()

        for layer_idx in allowed_layers:

            layer = model.get_layer(index=layer_idx)

            scores = np.zeros((layer.filters))
            for filter_idx in range(0, layer.filters):
                self.zeroed_out(model, layer_idx, filter_idx) #The weights are updated by reference
                q = softmax(model.predict(X_train[sub_sampling], verbose=False))

                # Compute KL Divergence -- See generate_mask.py line 129
                kl_loss = q * (np.log(q) - np.log(p))
                kl_loss = np.sum(kl_loss, axis=1)
                kl_loss = np.mean(kl_loss)
                scores[filter_idx] = kl_loss

                #Restore the original weights (unpruned)
                model.set_weights(unchaged_weights)

            output.append((layer_idx, scores))

        return output

class expectedABS():
    __name__ = 'DropNet: Reducing Neural Network Complexity via Iterative Pruning. ICML, 2020'

    def __init__(self):
       pass

    def relu_idx(self, model, layer_idx):
        idx = -1
        #Looking for the closest relu activatin
        for i in range(layer_idx, len(model.layers)):
            layer = model.get_layer(index=i)
            if isinstance(layer, Activation):
                idx = i
                break

        #The activation in VGG16 is inside Conv2D
        if idx == -1:
           idx = layer_idx

        return idx

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        for layer_idx in allowed_layers:

            layer = model.get_layer(index=layer_idx)

            idx = self.relu_idx(model, layer_idx)
            extractor = Model(model.input, model.get_layer(index=idx).output)
            w, h, n_filters = model.get_layer(index=idx).output_shape[1:]
            expected_abs_value = np.zeros((n_filters))

            #Memory efficiency
            for batch in gen_batches(X_train.shape[0], 512):
                feature_maps = extractor.predict(X_train[batch], batch_size=32, verbose=False)
                for filter in range(0, n_filters):
                    expected_abs_value[filter] += np.sum(np.abs(feature_maps[:, :, :, filter]))

            score = expected_abs_value/(w*h*X_train.shape[0])

            output.append((layer_idx, score))

        return output

class template():
    __name__ = 'Template for implementing a novel criterion'

    def __init__(self,):
        pass

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        for idx in allowed_layers:
            num_filters = model.get_layer(index=idx).filters
            scores = []
            for filter in num_filters:
                print('Do something')
                scores.append(1)

            output.append((idx, scores))

        return output

class template_SimilarityMetric():
    __name__ = 'Template for implementing similarity metric criteria'

    def __init__(self,):
        pass

    def metric(self, X, X_line):
        euclidian = np.linalg.norm(X-X_line, axis=1)
        return np.sum(euclidian)

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        F = Model(model.input, model.get_layer(index=-2).output)#Often, it is the Flatten layer
        F_features = F.predict(X_train[sub_sampling], verbose=0)

        F_line = Model(model.input, model.get_layer(index=-2).output)
        for layer_idx in allowed_layers:
            scores = []

            layer = F_line.get_layer(index = layer_idx)
            n_filters = layer.filters
            for f in range(n_filters):
                weights = layer.get_weights()
                original_weights = copy.deepcopy(weights)

                weights[0][:, :, :, f] = 0
                weights[1][f] = 0
                layer.set_weights(weights)

                F_line_features = F_line.predict(X_train[sub_sampling], verbose=0)

                layer.set_weights(original_weights)

                #A simple euclidian distance-based metric
                score = self.metric(F_features, F_line_features)
                scores.append(1-score)#Depending on the metric it could be score only

            output.append((layer_idx, scores))

        return output

def criteria(method='random'):

    if method == 'random':
        return random()

    if method == 'CKA':
        return CKA()

    if method == 'expectedABS':
        return expectedABS()

    if method == 'L1':
        return L1()

    if method == 'klDivergence':
        return klDivergence()

    if method == 'rank':
        return rank()

    if method == 'template_SimilarityMetric':
        return template_SimilarityMetric()