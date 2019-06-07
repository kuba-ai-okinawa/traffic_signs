"""
Script for running flask server
"""

import datetime

import flask
import numpy as np
import pandas
import tensorflow as tf

import traffic.ml
import traffic.utilities


def setup_prediction_models(app):
    """
    Adds prediction models to application object
    :param app: flask application instance
    """

    # Set up traffic signs categorization model
    app.traffic_signs_model = traffic.ml.get_model(input_shape=(32, 32, 3), categories_count=43)
    app.traffic_signs_model.load_weights(filepath="./data/untracked_data/traffic-signs-model.h5")

    categories_data_frame = pandas.read_csv("./data/signs_ids_to_names.csv", comment="#")
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
    app = flask.Flask('traffic_signs')
    app.debug = True
    setup_prediction_models(app)
    app.register_blueprint(GENERAL)
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
