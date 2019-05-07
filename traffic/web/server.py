"""
Module for building configurable flask app
"""

import flask

import traffic.web.blueprint


def get_configured_app(configuration):

    app = flask.Flask("traffic_signs")

    app.register_blueprint(traffic.web.blueprint.BLUEPRINT)

    return app
