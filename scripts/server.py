import flask


app = flask.Flask("traffic_signs")


@app.route("/")
def ping():

    return "Much trafficking"


app.run(host="0.0.0.0", port=5000, debug=True)
