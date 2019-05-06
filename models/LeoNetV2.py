from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD


def LeoNetV2_model(X_shape, nb_classes=2, nb_layers=4):
    # Inputs:
    #    X_shape = [ # spectrograms per batch, # audio channels, # spectrogram freq bins, # spectrogram time bins ]
    #    nb_classes = number of output n_classes
    #    nb_layers = number of conv-pooling sets in the CNN
    from keras import backend as K
    K.set_image_data_format('channels_last')

    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.5  # conv. layer dropout
    dl_dropout = 0.6  # dense layer dropout

    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape, name="Input"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # Leave this relu & BN here.  ELU is not good here (my experience)

    for layer in range(nb_layers - 1):  # add more layers than just the first
        model.add(Conv2D(nb_filters, kernel_size))
        # model.add(BatchNormalization(axis=1))  # ELU authors reccommend no BatchNorm.  I confirm.
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(cl_dropout))

    model.add(Flatten())
    model.add(Dense(128))  # 128 is 'arbitrary' for now
    # model.add(Activation('relu'))   # relu (no BN) works ok here, however ELU works a bit better...
    model.add(Activation('elu'))
    model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax", name="Output"))
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
