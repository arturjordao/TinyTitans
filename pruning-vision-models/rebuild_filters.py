import numpy as np
import random
import copy
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import rebuild_layers as rl#It implements some particular functions we need to use here

isFiltersAvailable = True

#Used by MobileNet
def relu6(x):
    return K.relu(x, max_value=6)

def rw_bn(w, index):
    w[0] = np.delete(w[0], index)
    w[1] = np.delete(w[1], index)
    w[2] = np.delete(w[2], index)
    w[3] = np.delete(w[3], index)
    return w

def rw_cn(index_model, idx_pruned, model):
    # This function removes the weights of the Conv2D considering the previous prunning in other Conv2D
    config = model.get_layer(index=index_model).get_config()
    weights = model.get_layer(index=index_model).get_weights()
    weights[0] = np.delete(weights[0], idx_pruned, axis=2)
    return create_Conv2D_from_conf(config, weights)

def create_Conv2D_from_conf(config, weights):
    n_filters = weights[0].shape[-1]
    return Conv2D(activation=config['activation'],
                  activity_regularizer=config['activity_regularizer'],
                  bias_constraint=config['bias_constraint'],
                  bias_regularizer=config['bias_regularizer'],
                  data_format=config['data_format'],
                  dilation_rate=config['dilation_rate'],
                  filters=n_filters,
                  kernel_constraint=config['kernel_constraint'],
                  kernel_regularizer=config['kernel_regularizer'],
                  kernel_size=config['kernel_size'],
                  name=config['name'],
                  padding=config['padding'],
                  strides=config['strides'],
                  trainable=config['trainable'],
                  use_bias=config['use_bias'],
                  weights=weights
                  )

def create_depthwise_from_config(config, weights):
    return DepthwiseConv2D(activation=config['activation'],
                    activity_regularizer=config['activity_regularizer'],
                    bias_constraint=config['bias_constraint'],
                    bias_regularizer=config['bias_regularizer'],
                    data_format=config['data_format'],
                    dilation_rate=config['dilation_rate'],
                    depth_multiplier=config['depth_multiplier'],
                    depthwise_constraint=config['depthwise_constraint'],
                    depthwise_initializer=config['depthwise_initializer'],
                    depthwise_regularizer=config['depthwise_regularizer'],
                    kernel_size=config['kernel_size'],
                    name=config['name'],
                    padding=config['padding'],
                    strides=config['strides'],
                    trainable=config['trainable'],
                    use_bias=config['use_bias'],
                    weights=weights
                    )

def remove_conv_weights(index_model, idxs, model):
    config, weights = (model.get_layer(index=index_model).get_config(),
                       model.get_layer(index=index_model).get_weights())
    weights[0] = np.delete(weights[0], idxs, axis=3)
    weights[1] = np.delete(weights[1], idxs)
    config['filters'] = weights[1].shape[0]
    return idxs, config, weights

def rebuild_resnet(model, blocks, layer_filters, num_classes=10):
    num_filters = 16
    num_res_blocks = blocks

    inputs = Input(shape=(model.inputs[0].shape.dims[1].value,
                          model.inputs[0].shape.dims[2].value,
                          model.inputs[0].shape.dims[3].value))

    #The first bock is not allow to prune
    _, config, weights = remove_conv_weights(1, [], model)
    conv = create_Conv2D_from_conf(config, weights)

    H = conv(inputs)
    H = BatchNormalization(weights=model.get_layer(index=2).get_weights())(H)
    H = Activation.from_config(model.get_layer(index=3).get_config())(H)
    x = H

    i = 4

    remove_Conv2D = [item[1] for item in layer_filters]
    remove_Conv2D.reverse()
    layer_block = False
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks[stack]):

            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            #This is the layer we can prune
            idx_previous, config, weights = remove_conv_weights(i, remove_Conv2D.pop(), model)
            conv = create_Conv2D_from_conf(config, weights)
            i = i + 1
            y = conv(x)
            wb = model.get_layer(index=i).get_weights()
            y = BatchNormalization(weights=rw_bn(wb, idx_previous))(y)
            i = i + 1
            y = Activation.from_config(model.get_layer(index=i).get_config())(y)
            i = i + 1

            #Second Module
            conv = rw_cn(index_model=i, idx_pruned=idx_previous, model=model)
            i = i + 1
            y = conv(y)#Aqui embaixo vai ter que ter um if relacionado ao bloco
            if layer_block == False:
                y = BatchNormalization(weights=model.get_layer(index=i).get_weights())(y)
            else:
                y = BatchNormalization(weights=model.get_layer(index=i+1).get_weights())(y)
                layer_block = False
            i = i + 1

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                _, config, weights = remove_conv_weights(i-1, [], model)
                conv = create_Conv2D_from_conf(config, weights)
                x = conv(x)
                i = i + 1

            x = Add()([x, y])
            i = i + 1
            #x = Activation('relu')(x)
            x = Activation.from_config(model.get_layer(index=i).get_config())(x)
            i = i + 1
        num_filters *= 2
        layer_block = True

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    layer = model.get_layer(index=-1)
    config = layer.get_config()
    weights = layer.get_weights()
    outputs = Dense(units=config['units'],
              activation=config['activation'],
              activity_regularizer=config['activity_regularizer'],
              bias_constraint=config['bias_constraint'],
              bias_regularizer=config['bias_regularizer'],
              kernel_constraint=config['kernel_constraint'],
              kernel_regularizer=config['kernel_regularizer'],
              name=config['name'],
              trainable=config['trainable'],
              use_bias=config['use_bias'],
              weights=weights)(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def allowed_layers_resnet(model):
    global isFiltersAvailable
    
    allowed_layers = []
    all_add = []
    n_filters = 0
    available_filters = 0
    
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)
        if isinstance(layer, Add):
            all_add.append(i)
        if isinstance(layer, Conv2D) and layer.strides == (2, 2) and layer.kernel_size != (1, 1):
            allowed_layers.append(i)

    allowed_layers.append(all_add[0] - 5)

    for i in range(1, len(all_add)):
        allowed_layers.append(all_add[i] - 5)
        

    #To avoid bug due to keras architecture (i.e., order of layers)
    #This ensure that only Conv2D are "allowed layers"
    tmp = allowed_layers
    allowed_layers = []

    for i in tmp:
        if isinstance(model.get_layer(index=i), Conv2D):
            allowed_layers.append(i)
            layer = model.get_layer(index=i)
            config = layer.get_config()
            # print(f"{config['filters']}")
            n_filters += config['filters']

    #allowed_layers.append(all_add[-1] - 5)
    available_filters = n_filters - len(allowed_layers)
    
    if available_filters == 0:
        isFiltersAvailable = False
        
    return allowed_layers

def idx_to_conv2Didx(model, indices):
    #Convert index onto Conv2D index (required by pruning methods)
    idx_Conv2D = 0
    output = []
    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Conv2D):
            if i in indices:
                output.append(idx_Conv2D)

            idx_Conv2D = idx_Conv2D + 1

    return output

def layer_to_prune_filters(model):

    if architecture_name.__contains__('ResNet'):
        if architecture_name.__contains__('50'):#ImageNet archicttures (ResNet50, 101 and 152)
            allowed_layers = allowed_layers_resnetBN(model)
        else:#CIFAR-like archictures (low-resolution datasets)
            allowed_layers = allowed_layers_resnet(model)

    if architecture_name.__contains__('MobileNetV2'):
        allowed_layers = allowed_layers_mobilenetV2(model)

    #allowed_layers = idx_to_conv2Didx(model, allowed_layers)
    return allowed_layers

def rebuild_network(model, scores, p_filter, totalFiltersToRemove = 0, wasPfilterZero = False):
    global isFiltersAvailable
    numberFiltersRemoved = 0
    scores = sorted(scores, key=lambda x: x[0])

    allowed_layers = [x[0] for x in scores]
    scores = [x[1] for x in scores]
    filtersToRemove = copy.deepcopy(scores)
    
    for i in range(0, len(scores)):
        num_remove = round(p_filter * len(scores[i]))
        numberFiltersRemoved += num_remove
        filtersToRemove[i] = np.argpartition(scores[i], num_remove)[:num_remove]

    layerSelectedList = [i for i in range(0,len(scores))]
    if totalFiltersToRemove != 0 and not wasPfilterZero:
        while ((totalFiltersToRemove - numberFiltersRemoved) != 0) and (len(layerSelectedList) != 0):
            layerSelected = random.choice(layerSelectedList)
            if (len(scores[layerSelected]) - (len(filtersToRemove[layerSelected])) - 1) > 0:
                filterToRemove = np.argpartition(scores[layerSelected], (len(filtersToRemove[layerSelected])+1))[:(len(filtersToRemove[layerSelected])+1)]
                filtersToRemove[layerSelected] = filterToRemove
                numberFiltersRemoved += 1
            else:
                layerSelectedList.remove(layerSelected)
            
    if len(layerSelectedList) == 0:
        isFiltersAvailable = False
        
    scores = [x for x in zip(allowed_layers, filtersToRemove)]
    
    #sort

    if architecture_name.__contains__('ResNet'):
        blocks = rl.count_blocks(model)
        return rebuild_resnet(model=model,
                              blocks=blocks,
                              layer_filters=scores)

    else:  # If not ResNet nor mobile then it is VGG-Based
        print('TODO: We need to implement (just update) this function')
        return None
