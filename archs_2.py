import keras.backend as K
import keras
from keras.models import Sequential

from keras.metrics import top_k_categorical_accuracy

from keras import (losses,
                   models,
                   optimizers)
from keras.activations import relu, softmax

from keras.layers import (Convolution1D,
                          Dense,
                          Dropout,
                          GlobalAveragePooling1D,
                          GlobalMaxPool1D,
                          Input,
                          MaxPool1D,
                          concatenate)


from keras.layers import (Convolution2D,
                          GlobalAveragePooling2D,
                          BatchNormalization,
                          Flatten,
                          GlobalMaxPool2D,
                          MaxPool2D,
                          concatenate,
                          Activation)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


def get_general_2d_conv_model(config):
    """
    any depth_convolution
    any depth_dense
    any drop_rate
    any kernel_size
    """
    nclass = config.n_classes  # 41
    filters = config.filters  # 32
    kernel_size = config.kernel_size  # (4, 10)
    dropout_rate = config.dropout_rate  # 0.15
    depth_conv = config.depth_conv  # 4
    depth_dense = config.depth_dense  # 1

    inp = Input(shape=(config.dim[0], config.dim[1], 1))
    x = None
    for index_conv in range(depth_conv):
        x = Convolution2D(
            filters, kernel_size,
            padding="same")(inp if index_conv == 0 else x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        if index_conv < 4:
            x = MaxPool2D()(x)
        x = Dropout(rate=dropout_rate)(x)
    x = Flatten()(x)
    for index_dense in range(depth_dense):
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)
    model.compile(
        optimizer=opt, loss=losses.categorical_crossentropy,
        metrics=['acc', top_3_accuracy, top_2_accuracy, top_1_accuracy])
    return model


def get_1d_dummy_model(config):
    
    nclass = config.n_classes
    input_length = config.audio_length
    
    inp = Input(shape=(input_length,1))
    x = GlobalMaxPool1D()(inp)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(
        optimizer=opt, loss=losses.categorical_crossentropy,
        metrics=['acc', top_3_accuracy, top_2_accuracy, top_1_accuracy])
    return model


def get_1d_conv_model(config):
    nclass = config.n_classes
    input_length = config.audio_length

    depth_dense = config.depth_dense
    dense_sizes = config.dense_sizes
    do_batch_normalization = config.do_batch_normalization
    dropout_rate = config.dropout_rate

    kernel_sizes = config.kernel_sizes_1d
    filter_sizes = config.filter_sizes
    maxpool_sizes = config.maxpool_sizes_1d

    if len(kernel_sizes) != len(filter_sizes) or len(kernel_sizes) != len(maxpool_sizes):
        raise ValueError(
            'kernel_sizes, filters and maxpool_size must have the same len.\n'
            '{} {} {}'.format(len(kernel_sizes), len(filter_sizes), len(maxpool_sizes)))

    inp = Input(shape=(input_length, 1))
    x = None
    for index_layer, (kernel_size, filters, maxpool_size) in enumerate(
            zip(kernel_sizes, filter_sizes, maxpool_sizes)):
        x = Convolution1D(
            filters, kernel_size,
            padding="valid")(inp if index_layer == 0 else x)
        if do_batch_normalization:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Convolution1D(
            filters, kernel_size,
            padding="valid")(x)
        if do_batch_normalization:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool1D(maxpool_size)(x)
        x = Dropout(rate=dropout_rate)(x)

    # x = Flatten()(x)
    x = GlobalMaxPool1D()(x)

    for index_dense, dense_size in enumerate(dense_sizes):
        x = Dense(dense_size)(x)
        if do_batch_normalization:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)

    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)
    
    model.compile(
        optimizer=opt, loss=losses.categorical_crossentropy,
        metrics=['acc', top_3_accuracy, top_2_accuracy, top_1_accuracy])
    return model
