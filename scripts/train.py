"""
Training script
"""

import sys

import keras

import traffic.data
import traffic.ml
import traffic.utilities


def main():
    """
    Script entry point
    """

    config = traffic.utilities.get_yaml_configuration(sys.argv[1:])

    data_bunch = traffic.data.TrafficDataBunch(
        config["data_directory"],
        categories_count=config["categories_count"],
        batch_size=config["train"]["batch_size"])

    # Create model
    model = traffic.ml.get_model(input_shape=config["input_shape"], categories_count=config["categories_count"])

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, verbose=1),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.1, verbose=1)
    ]

    # Train
    model.fit_generator(
        generator=data_bunch.training_data_generator,
        steps_per_epoch=data_bunch.training_samples_count // config["train"]["batch_size"],
        epochs=100, callbacks=callbacks,
        validation_data=data_bunch.validation_data_generator,
        validation_steps=data_bunch.validation_samples_count // config["train"]["batch_size"]
    )


if __name__ == "__main__":

    main()
