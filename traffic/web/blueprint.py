"""
Module with server endpoints
"""

import datetime
import json

import flask
import numpy as np
import cv2


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

        raw_image = flask.request.files["image"]
        image_string = raw_image.read()

        # flat_numpy_array = np.fromstring(image_string, np.uint8)
        flat_numpy_array = np.frombuffer(image_string, np.uint8)
        image = cv2.imdecode(flat_numpy_array, cv2.IMREAD_ANYCOLOR)

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
