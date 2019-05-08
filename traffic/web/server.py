"""
Module for building configurable flask app
"""

import flask
import tensorflow as tf

import traffic.web.blueprint
import traffic.ml


def setup_prediction_models(app, config):
    """
    Adds prediction models to application object
    :param app: flask application instance
    :param config: dictionary with options
    """

    # Set up traffic signs categorization model
    app.traffic_signs_model = traffic.ml.get_model(config["input_shape"], config["categories_count"])

    if config["environment"] != "test":
        print("\n\nWarning! Skipping loading weights! Need to sort this part out!\n\n".upper())
        # app.traffic_signs_model.load_weights(filepath=config["positive_negative"]["best_model_weights_path"])

    app.default_graph = tf.get_default_graph()


def get_configured_web_app(config):
    """
    Creates and configures flask application instance
    :param config: dictionary with configuration options
    :return: flask.Flask instance
    """

    app = flask.Flask("traffic_signs")
    app.debug = config["server"]["run_in_debug_mode"]

    setup_prediction_models(app, config)
    app.register_blueprint(traffic.web.blueprint.BLUEPRINT)

    return app
