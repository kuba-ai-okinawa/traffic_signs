"""
Module for building configurable flask app
"""

import flask

import traffic.web.blueprint


def get_configured_app(config):
    """
    Creates and configures flask application instance
    :param config: dictionary with configuration options
    :return: flask.Flask instance
    """

    app = flask.Flask("traffic_signs")
    app.debug = config["server"]["run_in_debug_mode"]

    app.register_blueprint(traffic.web.blueprint.BLUEPRINT)

    return app
