"""
Dummy tests module
"""

import pytest
import yaml

import traffic.web.server


@pytest.fixture(scope="function")
def app_fixture():
    """
    Simple fixture, returns flask app
    """

    with open("./configurations/test_config.yaml") as file:

        config = yaml.safe_load(file)

    app = traffic.web.server.get_configured_web_app(config)
    yield app


def test_ping_endpoint(app_fixture):
    """
    Test dummy positive negative endpoint
    """

    app = app_fixture

    with app.test_client() as client:

        response = client.get("/ping")

        assert response.status_code == 200

        response_text = response.get_data(as_text=True)

        assert "ping" in response_text
