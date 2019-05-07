"""
Module with machine learning model
"""

import keras


def get_model(input_shape, categories_count):
    """
    Builds and returns deep convolutional network for categorization
    :param input_shape: tuple of 3 ints, specifies input shape
    :param categories_count: int, number of categories model will predict from
    :return: keras functional model instance
    """

    input_layer = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='elu')(input_layer)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='elu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=categories_count, activation='softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])

    return model
