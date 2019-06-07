"""
Tests for utilities module
"""

import cv2
import numpy as np

import traffic.utilities


def test_binary_string_image_to_numpy_image():
    """
    Test function for decoding binary string into numpy array image works
    """

    expected = np.ones(shape=(32, 32, 3))

    _, encoded_image = cv2.imencode('.jpg', expected)
    binary_string = encoded_image.tostring()

    actual = traffic.utilities.binary_rgb_image_string_to_numpy_image(binary_string)

    assert np.all(expected == actual)
