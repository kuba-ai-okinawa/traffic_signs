"""
Script for running flask server
"""

import sys
import datetime

import flask
import pandas
import tensorflow as tf
import numpy as np

import traffic.utilities
import traffic.ml


def setup_prediction_models(app, config):
    """
    Adds prediction models to application object
    :param app: flask application instance
    :param config: dictionary with options
    """

    # Set up traffic signs categorization model
    app.traffic_signs_model = traffic.ml.get_model(config["INPUT_SHAPE"], config["CATEGORIES_COUNT"])

    if config["ENVIRONMENT"] != "test":

        app.traffic_signs_model.load_weights(filepath=config["MODEL_WEIGHTS_PATH"])

    categories_data_frame = pandas.read_csv(config["CATEGORIES_IDS_TO_NAMES_CSV_PATH"], comment="#")
    app.traffic_signs_categories = categories_data_frame["SignName"]

    app.default_graph = tf.get_default_graph()


APP = flask.Flask("traffic_signs")

config = traffic.utilities.get_yaml_configuration(sys.argv[1:])

APP.debug = config["FLASK"]["DEBUG"]
APP.config.from_mapping(config)

setup_prediction_models(APP, config)


@APP.route("/ping")
def ping():
    """
    Simple health probe checkpoint
    """

    return "ping at {}".format(datetime.datetime.utcnow())


@APP.route("/top_prediction", methods=["POST"])
def top_prediction():
    """
    Top prediction endpoint, outputs top prediction category and confidence
    """

    with flask.current_app.default_graph.as_default():

        raw_image_file = flask.request.files["image"]
        image = traffic.utilities.binary_string_image_to_numpy_image(raw_image_file.read())

        # Preprocessing
        image = image.astype(np.float32) / 255

        # Magic here

        return "whatever"


def main():
    """
    Script entry point
    """

    APP.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":

    main()
