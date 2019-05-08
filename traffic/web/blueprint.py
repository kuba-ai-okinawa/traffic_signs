"""
Module with server endpoints
"""

import datetime
import json

import flask
import numpy as np


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
