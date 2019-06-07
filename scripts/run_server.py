"""
Script for running flask server
"""

import datetime
import argparse

import flask
import numpy as np
import pandas
import tensorflow as tf
import yaml

import traffic.ml
import traffic.utilities
# import sys
# sys.path.append(".")
# from config import load_yaml


def load_yaml(path):
    """
    hoge
    """
    with open(path) as file:
        return yaml.load(file)


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


GENERAL = flask.Blueprint('general', __name__)


@GENERAL.route("/ping")
def ping():
    """
    Simple health probe checkpoint
    """

    return "ping at {}".format(datetime.datetime.utcnow())


@GENERAL.route("/top_prediction", methods=["POST"])
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


#     APP.run(host=config["host"], port=config["port"])
    app = flask.Flask('traffic_signs')
    app.debug = True
    setup_prediction_models(app, config["model_weight_path"], config["ids_name_path"])
#     setup_prediction_models(app)
    app.register_blueprint(GENERAL)
    app.run(host=config["host"], port=config["port"])


if __name__ == "__main__":
    main()
