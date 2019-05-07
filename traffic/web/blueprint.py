import datetime

import flask


BLUEPRINT = flask.blueprints.Blueprint(name='blueprint', import_name=__name__)


@BLUEPRINT.route("/ping")
def ping():
    """
    Simple health probe checkpoint
    """

    return "ping at {}".format(datetime.datetime.utcnow())
