"""
Script for running flask server
"""

import sys

import traffic.web.server
import traffic.utilities


def main():
    """
    Script entry point
    """

    config = traffic.utilities.get_yaml_configuration(sys.argv[1:])

    app = traffic.web.server.get_configured_web_app(config)
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":

    main()
