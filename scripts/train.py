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
        config["DATA_DIRECTORY"],
        categories_count=config["CATEGORIES_COUNT"],
        batch_size=config["TRAIN"]["batch_size"])

    # Create model
    model = traffic.ml.get_model(input_shape=config["INPUT_SHAPE"], categories_count=config["CATEGORIES_COUNT"])

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, verbose=1),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.1, verbose=1),
        keras.callbacks.ModelCheckpoint(config["MODEL_WEIGHTS_PATH"], save_best_only=True)
    ]

    # Train
    model.fit_generator(
        generator=data_bunch.training_data_generator,
        steps_per_epoch=data_bunch.training_samples_count // config["TRAIN"]["batch_size"],
        epochs=1, callbacks=callbacks,
        validation_data=data_bunch.validation_data_generator,
        validation_steps=data_bunch.validation_samples_count // config["TRAIN"]["batch_size"]
    )


if __name__ == "__main__":

    main()
