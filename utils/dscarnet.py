import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.layers import GlobalMaxPool2D, GlobalMaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, Concatenate, Dense, AveragePooling2D
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Dropout, SpatialDropout2D, Multiply       

def Conv2D_BN(x, filters, num_row, num_col, padding = "same", strides = (1, 1), batchnorm = False):
    if batchnorm:
        x = Conv2D(filters, (num_row, num_col), strides = strides, padding = padding, use_bias = False, activation = "relu")(x)
        x = BatchNormalization(momentum = 0.9)(x)
    return x

def dual_dscarnet(input_shape, input_shape2,
                  n_outputs = 6, batchnorm = True, dense_avf = "relu", last_avf = "softmax"):
    tf.keras.backend.clear_session()

    inputs = Input(input_shape)
    x = Conv2D_BN(inputs, 16, 5, 5, batchnorm = batchnorm)
    x = Conv2D_BN(x, 24, 3, 3, batchnorm = batchnorm)
    x = Conv2D_BN(x, 32, 3, 3, batchnorm = batchnorm)

    unit = 16
    b1 = Conv2D_BN(x, unit, 1, 1, batchnorm = batchnorm)
    b2 = Conv2D_BN(x, unit // 2, 1, 1, batchnorm = batchnorm)
    b2 = Conv2D_BN(b2, unit, 5, 5, batchnorm = batchnorm)
    b3 = Conv2D_BN(x, unit, 1, 1, batchnorm = batchnorm)
    b3 = Conv2D_BN(b3, unit*2, 3, 3, batchnorm = batchnorm)
    b3 = Conv2D_BN(b3, unit*2, 3, 3, batchnorm = batchnorm)
    bp = AveragePooling2D((3, 3), strides = 1, padding = "same")(x)
    bp = Conv2D_BN(bp, unit, 1, 1, batchnorm = batchnorm)

    x = Concatenate()([b1, b2, b3, bp])
    feat1 = GlobalAveragePooling2D()(x)

    inputs2 = Input(input_shape2)
    x2 = Conv2D_BN(inputs2, 16, 5, 5, batchnorm = batchnorm)
    x2 = Conv2D_BN(x2, 24, 3, 3, batchnorm = batchnorm)
    x2 = Conv2D_BN(x2, 32, 3, 3, batchnorm = batchnorm)

    b1 = Conv2D_BN(x2, unit, 1, 1, batchnorm = batchnorm)
    b2 = Conv2D_BN(x2, unit // 2, 1, 1, batchnorm = batchnorm)
    b2 = Conv2D_BN(b2, unit, 5, 5, batchnorm = batchnorm)
    b3 = Conv2D_BN(x2, unit, 1, 1, batchnorm = batchnorm)
    b3 = Conv2D_BN(b3, unit*2, 3, 3, batchnorm = batchnorm)
    b3 = Conv2D_BN(b3, unit*2, 3, 3, batchnorm = batchnorm)
    bp = AveragePooling2D((3, 3), strides = 1, padding = "same")(x2)
    bp = Conv2D_BN(bp, unit, 1, 1, batchnorm = batchnorm)
    x2 = Concatenate()([b1, b2, b3, bp])
    feat2 = GlobalAveragePooling2D()(x2)

    x = Concatenate()([feat1, feat2])
    x = Dropout(0.3)(x)
    x = Dense(64, activation = dense_avf)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(n_outputs, activation = last_avf)(x)
    return Model([inputs, inputs2], outputs)
