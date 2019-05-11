"""
Dummy tests module
"""

import io
import json

import pytest
import yaml
import numpy as np
import cv2

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
    Test ping endpoint
    """

    app = app_fixture

    with app.test_client() as client:

        response = client.get("/ping")

        assert response.status_code == 200

        response_text = response.get_data(as_text=True)

        assert "ping" in response_text


def test_top_prediction_endpoint(app_fixture):
    """
    Test top prediction endpoint
    """

    app = app_fixture

    with app.test_client() as client:

        image = np.ones(shape=(32, 32, 3))
        _, encoded_image = cv2.imencode('.jpg', image)

        data = {
            'image': (io.BytesIO(encoded_image.tostring()), 'image')
        }

        response = client.post("/top_prediction", data=data)

        assert response.status_code == 200

        response_text = response.get_data(as_text=True)

        assert "category" in response_text
        assert "confidence" in response_text


def test_top_k_predictions_endpoint(app_fixture):
    """
    Test top k predictions endpoint
    """

    app = app_fixture

    with app.test_client() as client:

        image = np.ones(shape=(32, 32, 3))
        _, encoded_image = cv2.imencode('.jpg', image)

        k = 3

        data = {
            'image': (io.BytesIO(encoded_image.tostring()), 'image'),
            'k': k
        }

        response = client.post("/top_k_predictions", data=data)

        assert response.status_code == 200

        response_text = response.get_data(as_text=True)

        data = json.loads(response_text)

        # Assert we got a list of expected length
        assert isinstance(data, list)
        assert len(data) == 3

        # Assert each element of a list is a dictionary with expected structure
        for element in data:

            assert isinstance(element, dict)
            assert "category" in element.keys()
            assert "confidence" in element.keys()

        confidences = [element["confidence"] for element in data]

        # Assert elements are order by confidences in descending order
        assert sorted(confidences, reverse=True) == confidences
