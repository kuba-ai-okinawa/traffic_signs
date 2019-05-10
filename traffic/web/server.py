"""
Module for building configurable flask app
"""

import flask
import tensorflow as tf
import pandas

import traffic.web.blueprint
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


def get_configured_web_app(config):
    """
    Creates and configures flask application instance
    :param config: dictionary with configuration options
    :return: flask.Flask instance
    """

    app = flask.Flask("traffic_signs")
    app.debug = config["FLASK"]["DEBUG"]
    app.config.from_mapping(config)

    setup_prediction_models(app, config)
    app.register_blueprint(traffic.web.blueprint.BLUEPRINT)

    return app
