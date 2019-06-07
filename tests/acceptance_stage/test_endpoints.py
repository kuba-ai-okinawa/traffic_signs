"""
Endpoints tests
"""

<<<<<<< HEAD
import io
import json
import flask
import pytest
import numpy as np
import cv2

=======
import pytest
>>>>>>> 96b93c80f99a921c3d44f615cccd91360dd3113e
import scripts.run_server


@pytest.fixture
def client():
    """Prepare client"""
<<<<<<< HEAD
    app = flask.Flask('test')
    scripts.run_server.setup_prediction_models(app)
    app.register_blueprint(scripts.run_server.GENERAL_ENDPOINT)
=======

    config = {
        "model_weight_path": "",
        "ids_name_path": "./data/signs_ids_to_names.csv"
    }
    app = scripts.run_server.create_app(config, is_test_env=True)
>>>>>>> 96b93c80f99a921c3d44f615cccd91360dd3113e
    client = app.test_client()
    yield client


@pytest.fixture
def sample_image():
    """
    Make image for test
    """
    image = np.ones(shape=(32, 32, 3))
    _, encoded_image = cv2.imencode('.jpg', image)

    data = {
        'image': (io.BytesIO(encoded_image.tostring()), 'image')
    }
    yield data


def test_ping_endpoint(client):
    """
    Test ping endpoint
    """
    resp = client.get('/ping')
    assert resp.status_code == 200
    assert "ping" in str(resp.data)


def test_top_prediction_endpoint(client, sample_image):
    """
    Test top prediction endpoint
    """

    result_dict_byte = client.post("/top_prediction", data=sample_image)
    result_dict = json.loads(str(result_dict_byte.data))

    expected_dict = {'rank': int, 'category': str, 'confidence': float}

    result_key = list(result_dict.keys())
    for expected_key, expected_type in expected_dict.items():
        assert expected_key in result_key
        assert isinstance(result_dict[expected_key], expected_type)


def test_topk_prediction_endpoint(client, sample_image):
    """
    Test top-k prediction endpoint
    """

    result_list_byte = client.post("/top_prediction", data=sample_image, k=5)
    result_dicts = json.loads(str(result_list_byte.data))

    expected_dict = {'rank': int, 'category': str, 'confidence': float}
    for result_dict in result_dicts:
        result_key = list(result_dict.keys())
        for expected_key, expected_type in expected_dict.items():
            assert expected_key in result_key
            assert isinstance(result_dict[expected_key], expected_type)
    assert len(result_dicts) == 5
