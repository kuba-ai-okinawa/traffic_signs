"""
Module with server endpoints
"""

import datetime
import json

import flask
import numpy as np

import traffic.utilities


BLUEPRINT = flask.blueprints.Blueprint(name='blueprint', import_name=__name__)


@BLUEPRINT.route("/ping")
def ping():
    """
    Simple health probe checkpoint
    """

    return "ping at {}".format(datetime.datetime.utcnow())


@BLUEPRINT.route("/dummy_top_prediction")
def dummy_top_prediction():
    """
    Dummy prediction endpoint. Uses actual prediction model, but creates dummy input
    """

    with flask.current_app.default_graph.as_default():

        data = np.random.rand(1, 32, 32, 3)
        raw_predictions = flask.current_app.traffic_signs_model.predict(data)[0]

        category_index = np.argmax(raw_predictions)

        result_dictionary = {
            "category_index": int(category_index),
            "confidence": float(raw_predictions[category_index])
        }

        return json.dumps(result_dictionary)


@BLUEPRINT.route("/top_prediction", methods=["POST"])
def top_prediction():
    """
    Top prediction endpoint, outputs top prediction category and confidence
    """

    with flask.current_app.default_graph.as_default():

        raw_image_file = flask.request.files["image"]
        image = traffic.utilities.binary_string_image_to_numpy_image(raw_image_file.read())

        # Make sure image of correct size is provided
        if image.shape != tuple(flask.current_app.config["INPUT_SHAPE"]):

            result_dictionary = {"error": "invalid image shape"}
            return json.dumps(result_dictionary)

        # Preprocessing
        image = image.astype(np.float32) / 255

        single_image_batch = np.array([image])
        raw_predictions = flask.current_app.traffic_signs_model.predict(single_image_batch)[0]

        category_index = np.argmax(raw_predictions)

        result_dictionary = {
            "category": flask.current_app.traffic_signs_categories[category_index],
            "confidence": float(raw_predictions[category_index])
        }

        return json.dumps(result_dictionary)


@BLUEPRINT.route("/top_k_predictions", methods=["POST"])
def top_k_predictions():
    """
    Top k predictions endpoint, outputs top prediction category and confidence for top k predictions
    """

    with flask.current_app.default_graph.as_default():

        raw_image_file = flask.request.files["image"]
        image = traffic.utilities.binary_string_image_to_numpy_image(raw_image_file.read())

        # Make sure image of correct size is provided
        if image.shape != tuple(flask.current_app.config["INPUT_SHAPE"]):

            result_dictionary = {"error": "invalid image shape"}
            return json.dumps(result_dictionary)

        # Preprocessing
        image = image.astype(np.float32) / 255

        single_image_batch = np.array([image])
        raw_predictions = flask.current_app.traffic_signs_model.predict(single_image_batch)[0]

        # k, or how many top predictions we should return
        k = int(flask.request.form["k"])

        # Get indices of top k predictions
        # argsort uses ascending sort, so we use negative of original values to get descending sorting order
        top_k_indices = np.argsort(-raw_predictions)[:k]

        top_k_categories = flask.current_app.traffic_signs_categories[top_k_indices]
        top_k_confidences = raw_predictions[top_k_indices]

        # Create a list of tuples (category, confidence), in descending order.
        # We need to cast confidences from numpy floats to normal floats, as numpy floats can't be json serialized.
        # For same reason we need to cast zip iterator to list,
        category_confidence_tuples = list(zip(top_k_categories, top_k_confidences.tolist()))
        return json.dumps(category_confidence_tuples)
