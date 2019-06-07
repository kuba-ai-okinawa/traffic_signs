"""
Script for running flask server
"""

import sys
import os
import datetime
import argparse

import flask
import pandas
import tensorflow as tf
import numpy as np

import traffic.utilities
import traffic.ml
from config import load_yaml


def setup_prediction_models(app, model_weight_path, ids_name_path):
    """
    Adds prediction models to application object
    :param app: flask application instance
    """

    # Set up traffic signs categorization model
    app.traffic_signs_model = traffic.ml.get_model(input_shape=(32, 32, 3), categories_count=43)
    app.traffic_signs_model.load_weights(filepath=model_weight_path)

    categories_data_frame = pandas.read_csv(ids_name_path, comment="#")
    app.traffic_signs_categories = categories_data_frame["SignName"]

    app.default_graph = tf.get_default_graph()


APP = flask.Flask("traffic_signs")

APP.debug = True

# setup_prediction_models(APP)


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
        image = traffic.utilities.binary_rgb_image_string_to_numpy_image(raw_image_file.read())

        # Preprocessing
        image = image.astype(np.float32) / 255

        # Magic here

        return "whatever"


def main():
    """
    Script entry point
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()
    # print(load_yaml(args.config_path))
    config = load_yaml(args.config_path)
    print(config)

    setup_prediction_models(APP, config["model_weight_path"], config["ids_name_path"])

    APP.run(host=config["host"], port=config["port"])


if __name__ == "__main__":

    main()
