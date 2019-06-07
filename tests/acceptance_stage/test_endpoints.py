"""
Endpoints tests
"""

from scripts.run_server import APP


def test_ping_endpoint():
    """
    Test ping endpoint
    """

    with APP.test_client() as client:
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
