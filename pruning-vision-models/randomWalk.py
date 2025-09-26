import os
import sys
import time
import argparse
import random

import numpy as np

from datetime import datetime

import tensorflow as tf
import keras.backend as K

from tensorflow import keras
from keras.layers import *
from keras.activations import *
from tensorflow.data import Dataset

from sklearn.utils import gen_batches
from sklearn.metrics._classification import accuracy_score

import rebuild_layers as rl
import rebuild_filters as rf
from pruning_criteria import criteria_filter as cf
from pruning_criteria import criteria_layer as cl

sys.path.insert(0, '../utils')
from utils import custom_functions as func
from utils import custom_callbacks

class NeuralNetwork():
    def __init__(self):
        self.model = None
        self.modelsFilter = []
        self.modelsLayer = []
        self.architecture_name = None
        self.criterion_layer = None
        self.criterion_filter = None
        self.p_filter = None
        self.p_layer = None
        self.acc = None
        
        self.directory = 'saved_models/saved_model_' + datetime.now().strftime("%Y_%m_%d")

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f'A pasta "{self.directory}" foi criada.')
        else:
            print(f'A pasta "{self.directory}" já existe.')

    def statistics(self, model, i, acc):
        n_params = model.count_params()
        n_filters = func.count_filters(model)
        flops, _ = func.compute_flops(model)
        blocks = rl.count_blocks(model)

        memory = func.memory_usage(1, model)

        print('Iteration [{}] Accuracy [{}] Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
            'Memory [{:.6f}]'.format(i, acc, blocks, n_params, n_filters, flops, memory), flush=True)    
    
    def finetuning(self, epochs,model, X_train, y_train, X_test, y_test):
        lr = 0.01
        schedule = [(100, lr / 10), (150, lr / 100)]
        lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
        callbacks = [lr_scheduler]

        sgd = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
        print('Accuracy before fine-tuning  [{:.4f}]'.format(acc), flush=True)

        for ep in range(0, epochs):
            y_tmp = np.concatenate((y_train, y_train, y_train))
            X_tmp = np.concatenate(
                (func.data_augmentation(X_train),
                func.data_augmentation(X_train),
                func.data_augmentation(X_train)))

            with tf.device("CPU"):
                X_tmp = Dataset.from_tensor_slices((X_tmp, y_tmp)).shuffle(4 * 128).batch(128)

            model.fit(X_tmp, batch_size=128,
                    callbacks=callbacks, verbose=2,
                    epochs=ep, initial_epoch=ep - 1)

            if ep % 5 == 0: # % 5
                acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
                print('Accuracy [{:.4f}]'.format(acc), flush=True)

        return model
    
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
    
    def randomScore(self, unpruned, pruned):
        bestScore = ["",0.0]
        bestModel = None

        # F = keras.Model(unpruned.input, unpruned.get_layer(index=-2).output)
        # featuresUnpruned = F.predict(self.X_test, verbose=0)
        
        for criteria, model in pruned:
            print("Criteria: " + str(criteria))
            # model = self.finetuning(10, model, self.X_train, self.y_train, self.X_test, self.y_test)
            # F = keras.Model(model.input, model.get_layer(index=-2).output)
            # featuresPruned = F.predict(self.X_test, verbose=0)
            
            score = random.random() #self.feature_space_linear_cka(featuresUnpruned, featuresPruned)
            print(f"Score {score}")
            if score > bestScore[1]:
                bestScore[0] = criteria
                bestScore[1] = score
                bestModel = model
                
        return bestScore[0], bestScore[1], bestModel
    
    
    def mixtureMetric(self, unpruned, pruned):
        bestScore = ["",0.0]
        bestModel = None

        F = keras.Model(unpruned.input, unpruned.get_layer(index=-2).output)
        featuresUnpruned = F.predict(self.X_test, verbose=0)
        
        for criteria, model in pruned:
            print("Criteria: " + str(criteria))
            F = keras.Model(model.input, model.get_layer(index=-2).output)
            featuresPruned = F.predict(self.X_test, verbose=0)
            scoreCKA = self.feature_space_linear_cka(featuresUnpruned, featuresPruned)
            flops = (1 - (func.compute_flops(model)[0]/func.compute_flops(unpruned)[0]))
            parametersNumber = (1 - (model.count_params()/unpruned.count_params()))
            # lattency = (1 - (func.meanLattency(model,self.X_test)/func.meanLattency(unpruned,self.X_test)))
            
            print(f"CKA: {scoreCKA} \nFLOPs: {flops} \nParameter Numbers: {parametersNumber} \nLattency: {lattency} \n")
            score = scoreCKA*flops*parametersNumber*lattency
            
            print(f"Score {score}")
            if score > bestScore[1]:
                bestScore[0] = criteria
                bestScore[1] = score
                bestModel = model
                
        return bestScore[0], bestScore[1], bestModel
        
    def selectPrunedNetwork(self, selectedCriteriaLayer, scoreMetricLayer, bestModelLayer, selectedCriteriaFilter, scoreMetricFilter, bestModelFilter):
        if scoreMetricLayer >= scoreMetricFilter:
            self.model = bestModelLayer
            self.usedCriteria = selectedCriteriaLayer
            self.prunedType = "layer"
        else:
            self.model = bestModelFilter
            self.usedCriteria = selectedCriteriaFilter
            self.prunedType = "filter"
            
    def clearVariables(self):
        self.modelsFilter = []
        self.modelsLayer = []
    
    def saveCKALog(self, iteration, scoreLayer,scoreFilter,selectedMode):
        with open(self.directory + ".txt", "a") as arquivo:
            arquivo.write(f"{iteration} {scoreLayer} {scoreFilter} {selectedMode}\n")
    
if __name__ == '__main__':
    neuralNetwork = NeuralNetwork()
    
    np.random.seed(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture_name', type=str, default='16_ResNet56_klDivergence_filter_iterations[16]')
    parser.add_argument('--criterion_layer', type=list, default=['klDivergence']) #'rank', 'klDivergence', 'expectedABS', 'random', 'CKA', template_SimilarityMetric
    parser.add_argument('--criterion_filter', type=list,default=['klDivergence']) # 'L1', 'rank', 'klDivergence', 'expectedABS', 'CKA', 'template_SimilarityMetric'
    parser.add_argument('--p_layer', type=int, default=1)
    parser.add_argument('--p_filter', type=float, default=0.1)

    args = parser.parse_args()
    neuralNetwork.architecture_name = args.architecture_name
    neuralNetwork.criterion_layer = args.criterion_layer
    neuralNetwork.criterion_filter = args.criterion_filter
    neuralNetwork.p_filter = args.p_filter
    neuralNetwork.p_layer = args.p_layer

    rf.architecture_name = neuralNetwork.architecture_name
    rl.architecture_name = neuralNetwork.architecture_name
    
    #Use it to subsampling the training data inside a criterion
    #cf.n_samples = 2

    print(args, flush=False)
    print('Architecture [{}] p_filter[{}] p_layer[{}]'.format(neuralNetwork.architecture_name, neuralNetwork.p_filter, neuralNetwork.p_layer), flush=True)

    neuralNetwork.X_train, neuralNetwork.y_train, neuralNetwork.X_test, neuralNetwork.y_test = func.cifar_resnet_data(debug=False)

    neuralNetwork.model = func.load_model('{}'.format(neuralNetwork.architecture_name),
                          '{}++'.format(neuralNetwork.architecture_name))

    # neuralNetwork.model = func.load_model('{}'.format(neuralNetwork.architecture_name),
    #                         '{}'.format(neuralNetwork.architecture_name))
    
    neuralNetwork.acc = accuracy_score(np.argmax(neuralNetwork.y_test, axis=1), np.argmax(neuralNetwork.model.predict(neuralNetwork.X_test, verbose=0), axis=1))
    neuralNetwork.statistics(neuralNetwork.model, 'Unpruned', neuralNetwork.acc)
            
    # print(f"A latencia média é de {func.meanLattency(neuralNetwork.model,neuralNetwork.X_test)}")
    
    i = 17
    while rl.count_res_blocks(neuralNetwork.model) != [2, 2, 2]:
        if not rf.isFiltersAvailable:
            print("Só é possível remover layers")
            print("Pruning by layer")
            allowed_layers = rl.blocks_to_prune(neuralNetwork.model)
            for criteria in neuralNetwork.criterion_layer:
                print(f"{criteria}")
                layer_method = cl.criteria(criteria)
                scores = layer_method.scores(neuralNetwork.model, neuralNetwork.X_train, neuralNetwork.y_train, allowed_layers)    
                neuralNetwork.modelsLayer.append([criteria, rl.rebuild_network(neuralNetwork.model, scores, neuralNetwork.p_layer)])

            selectedCriteriaLayer, scoreMetricLayer, bestModelLayer = neuralNetwork.randomScore(neuralNetwork.model, neuralNetwork.modelsLayer)
            neuralNetwork.usedCriteria = selectedCriteriaLayer
            neuralNetwork.prunedType = 'layer'
            neuralNetwork.model = bestModelLayer
        else:
            print("Pruning by layer")
            allowed_layers = rl.blocks_to_prune(neuralNetwork.model)
            for criteria in neuralNetwork.criterion_layer:
                layer_method = cl.criteria(criteria)
                scores = layer_method.scores(neuralNetwork.model, neuralNetwork.X_train, neuralNetwork.y_train, allowed_layers)    
                neuralNetwork.modelsLayer.append([criteria, rl.rebuild_network(neuralNetwork.model, scores, neuralNetwork.p_layer)])

            neuralNetwork.p_filter = (func.count_filters(neuralNetwork.model) - func.count_filters(neuralNetwork.modelsLayer[0][1]))/(func.count_filters(neuralNetwork.model))

            print(f"No modelo original, temos {func.count_filters(neuralNetwork.model)}, no podado por layer temos {func.count_filters(neuralNetwork.modelsLayer[0][1])}, logo vamos setar para remover {neuralNetwork.p_filter} %")
            print("Pruning by filter")
            allowed_layers_filters = rf.layer_to_prune_filters(neuralNetwork.model)

            for criteria in neuralNetwork.criterion_filter:
                filter_method = cf.criteria(criteria)
                scores = filter_method.scores(neuralNetwork.model, neuralNetwork.X_train, neuralNetwork.y_train, allowed_layers_filters)    
                neuralNetwork.modelsFilter.append([criteria, rf.rebuild_network(neuralNetwork.model, scores, neuralNetwork.p_filter, (func.count_filters(neuralNetwork.model) - func.count_filters(neuralNetwork.modelsLayer[0][1])))])

            print(f"No modelo original, temos {func.count_filters(neuralNetwork.model)}, no podado por filtros temos {func.count_filters(neuralNetwork.modelsFilter[0][1])}")

            selectedCriteriaLayer, scoreMetricLayer, bestModelLayer = neuralNetwork.randomScore(neuralNetwork.model, neuralNetwork.modelsLayer, )
            selectedCriteriaFilter, scoreMetricFilter, bestModelFilter = neuralNetwork.randomScore(neuralNetwork.model, neuralNetwork.modelsFilter, )

            neuralNetwork.selectPrunedNetwork(selectedCriteriaLayer, scoreMetricLayer, bestModelLayer, selectedCriteriaFilter, scoreMetricFilter, bestModelFilter)

            print(f"Select pruned type {neuralNetwork.prunedType}, using {neuralNetwork.usedCriteria}")

            neuralNetwork.saveCKALog(i,scoreMetricLayer, scoreMetricFilter, neuralNetwork.prunedType)
            
        neuralNetwork.model = neuralNetwork.finetuning(200, neuralNetwork.model, neuralNetwork.X_train, neuralNetwork.y_train, neuralNetwork.X_test, neuralNetwork.y_test)#or neuralNetwork.model = finetuning(pruned_model_layer,...)

        neuralNetwork.acc = accuracy_score(np.argmax(neuralNetwork.y_test, axis=1), np.argmax(neuralNetwork.model.predict(neuralNetwork.X_test, verbose=0), axis=1))

        neuralNetwork.statistics(neuralNetwork.model, i, neuralNetwork.acc)
        # meanLattency = func.meanLattency(neuralNetwork.model,neuralNetwork.X_test)
        
        # print(f"A latencia média é de {meanLattency}")
        if i < 10:
            func.save_model(neuralNetwork.directory + '/0{}_{}_{}_{}_iterations[{}]'.format(i, neuralNetwork.architecture_name, neuralNetwork.usedCriteria, neuralNetwork.prunedType, i), neuralNetwork.model)
        else:
            func.save_model(neuralNetwork.directory + '/{}_{}_{}_{}_iterations[{}]'.format(i, neuralNetwork.architecture_name, neuralNetwork.usedCriteria, neuralNetwork.prunedType, i), neuralNetwork.model)
            
        neuralNetwork.clearVariables()
        i+=1
