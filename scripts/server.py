import argparse
import sys

import traffic.web.server
import traffic.utilities


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="path to config file", required=True)

    config = traffic.utilities.get_yaml_configuration(sys.argv[1:])

    print("Passing configuration")
    print(config)

    app = traffic.web.server.get_configured_app(configuration=config)
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":

    main()
