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


def setup_prediction_models(app, model_weight_path, ids_name_path, is_test_env):
    """
    Adds prediction models to application object
    :param app: flask application instance
    """

    # Set up traffic signs categorization model
    app.traffic_signs_model = traffic.ml.get_model(input_shape=(32, 32, 3), categories_count=43)
    if not is_test_env:
        app.traffic_signs_model.load_weights(filepath=model_weight_path)

    categories_data_frame = pandas.read_csv(ids_name_path, comment="#")
    app.traffic_signs_categories = categories_data_frame["SignName"]

    app.default_graph = tf.get_default_graph()


GENERAL_ENDPOINT = flask.Blueprint('general', __name__)


def compute_top_k_indexes(predicted: np.ndarray, k: int):
    """
    :param predicted: CNN outputs (confidence values)
    :param k: number of top results
    :return: top-k index list
    """
    sorted_indexes = np.argsort(predicted)[::-1]
    traffic_sign_categories = flask.current_app.traffic_signs_categories
    assert k > 0

    if k > len(traffic_sign_categories):
        k = len(traffic_sign_categories)

    return sorted_indexes[:k]


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
        raw_image_file = flask.request.files["image"]
        image = traffic.utilities.binary_rgb_image_string_to_numpy_image(raw_image_file.read())

        # Preprocessing
        image = preprocess(image)

        # Magic herei

        y = app.traffic_signs_model.predict(image)[0]

        top_1_dict = generate_top_k_dicts(y, 1)[0]

        return json.dumps(top_1_dict)


def generate_top_k_dicts(predicted: np.ndarray, k: int):
    """
    generate top-k dictionaries from CNN output
    :param predicted: CNN outputs (confidence values)
    :param k: number of top results
    :return: top-k list of dictionary
    """
    app = flask.current_app
    top_k_indexes = compute_top_k_indexes(predicted, k)
    return [{'rank': rank + 1,
             'category': app.traffic_signs_categories[top_k_index],
             'confidence': float(predicted[top_k_index])}
            for rank, top_k_index in enumerate(top_k_indexes)]


def preprocess(np_image: np.ndarray) -> np.ndarray:
    """
    convert np_image into acceptable shape and range for CNN
    :param np_image: np.ndarray [W, H, C]
    :return: np.ndarray [1, 32, 32, 3]
    """
    resized_image = cv2.resize(np_image, (32, 32), cv2.INTER_LANCZOS4)
    resized_image = resized_image.astype(np.float32) / 255
    return resized_image[np.newaxis, :, :, :]


def create_app(config, is_test_env=False):
    """Create flask app"""
    app = flask.Flask('traffic_signs')
    app.debug = True
    app.config['TESTING'] = is_test_env
    setup_prediction_models(app, config["model_weight_path"], config["ids_name_path"], is_test_env)
    app.register_blueprint(GENERAL_ENDPOINT)

    return app


def main():
    """
    Script entry point
    """
    config = traffic.utilities.get_yaml_configuration(None)

    app = create_app(config, is_test_env=False)
    app.run(host=config["host"], port=config["port"])


if __name__ == "__main__":
    main()
