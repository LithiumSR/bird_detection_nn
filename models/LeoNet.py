from keras import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, \
    ELU, Conv2D
from keras.optimizers import SGD


def leonet_model(input_shape):
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    nb_layers = 4
    nb_classes = 2

    model = Sequential()
    model.add(Conv2D(nb_filters, (3, 3), padding='same',
                     input_shape=input_shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    for layer in range(nb_layers - 1):
        model.add(Conv2D(nb_filters, (3, 3), padding='same',
                         input_shape=input_shape[1:]))
        model.add(BatchNormalization())
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
