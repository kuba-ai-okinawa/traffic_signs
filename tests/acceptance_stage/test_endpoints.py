"""
Endpoints tests
"""

import pytest

import scripts.run_server


@pytest.fixture
def client():
    """Prepare client"""
    app = scripts.run_server.create_app(is_test_env=True)
    client = app.test_client()
    yield client


def test_ping_endpoint(client):
    """
    Test ping endpoint
    """
    resp = client.get('/ping')
    assert resp.status_code == 200
    assert "ping" in str(resp.data)


def test_top_prediction_endpoint():
    """
    Test top prediction endpoint
    """

    # app = ...

    # with app.test_client() as client:
    #
    #     image = np.ones(shape=(32, 32, 3))
    #     _, encoded_image = cv2.imencode('.jpg', image)
    #
    #     data = {
    #         'image': (io.BytesIO(encoded_image.tostring()), 'image')
    #     }

    # response = client.post("/top_prediction", data=data)

    # Assert correct results here
    assert 1 == 1
