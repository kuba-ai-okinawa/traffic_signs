"""
Script for running flask server
"""

import datetime

import json
import flask
import numpy as np
import pandas
import tensorflow as tf
import cv2

import traffic.ml
import traffic.utilities


def setup_prediction_models(app, is_test_env):
    """
    Adds prediction models to application object
    :param app: flask application instance
    """

    # Set up traffic signs categorization model
    app.traffic_signs_model = traffic.ml.get_model(input_shape=(32, 32, 3), categories_count=43)
    if not is_test_env:
        app.traffic_signs_model.load_weights(filepath="./data/untracked_data/traffic-signs-model.h5")

    categories_data_frame = pandas.read_csv("./data/signs_ids_to_names.csv", comment="#")
    app.traffic_signs_categories = categories_data_frame["SignName"]

    app.default_graph = tf.get_default_graph()


GENERAL_ENDPOINT = flask.Blueprint('general', __name__)


@GENERAL_ENDPOINT.route("/ping")
def ping():
    """
    Simple health probe checkpoint
    """

    return "ping at {}".format(datetime.datetime.utcnow())


@GENERAL_ENDPOINT.route("/top_prediction", methods=["POST"])
def top_prediction():
    """
    Top prediction endpoint, outputs top prediction category and confidence
    """
    app = flask.current_app
    with app.default_graph.as_default():
        image = load_image()

        predictions = app.traffic_signs_model.predict(image)[0]

        top_dict = generate_top_k_dicts(predictions, k=1)[0]

        return json.dumps(top_dict)


@GENERAL_ENDPOINT.route("/top_k_prediction", methods=["POST"])
def top_k_prediction():
    """
    Top prediction endpoint, outputs top prediction category and confidence
    """

    with flask.current_app.default_graph.as_default():
        image = load_image()
        k = int(flask.request.form['k'])

        predictions = flask.current_app.traffic_signs_model.predict(image)[0]
        top_k_dicts = generate_top_k_dicts(predictions, k)

        return json.dumps(top_k_dicts)


def load_image():
    """
    get image data from POST and convert acceptable np.ndarray for CNN
    :return:
    """
    raw_image_file = flask.request.files["image"]
    image = traffic.utilities.binary_rgb_image_string_to_numpy_image(raw_image_file.read())
    return preprocess(image)


def preprocess(np_image: np.ndarray) -> np.ndarray:
    """
    convert np_image into acceptable shape and range for CNN
    :param np_image: np.ndarray [W, H, C]
    :return: np.ndarray [1, 32, 32, 3]
    """
    resized_image = cv2.resize(np_image, (32, 32), cv2.INTER_LANCZOS4)
    resized_image = resized_image.astype(np.float32) / 255
    return resized_image[np.newaxis, :, :, :]


def compute_top_k_indexes(predictions: np.ndarray, k: int):
    """
    :param predictions: CNN outputs (confidence values)
    :param k: number of top results
    :return: top-k index list
    """
    sorted_indexes = np.argsort(predictions)[::-1]
    assert k > 0, 'k must be more than 0'
    return sorted_indexes[:k]


def generate_top_k_dicts(predictions: np.ndarray, k: int):
    """
    generate top-k dictionaries from CNN output
    :param predictions: CNN outputs (confidence values)
    :param k: number of top results
    :return: top-k list of dictionary
    """
    app = flask.current_app
    top_k_indexes = compute_top_k_indexes(predictions, k)
    return [{'rank': rank + 1,
             'category': app.traffic_signs_categories[top_k_index],
             'confidence': float(predictions[top_k_index])}
            for rank, top_k_index in enumerate(top_k_indexes)]


def create_app(is_test_env=False):
    """Create flask app"""
    app = flask.Flask('traffic_signs')
    app.debug = True
    app.config['TESTING'] = is_test_env
    app.register_blueprint(GENERAL_ENDPOINT)
    setup_prediction_models(app, is_test_env)
    return app


def main():
    """
    Script entry point
    """
    app = create_app(is_test_env=False)
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
