"""
Tests for traffic.utilities module
"""

import traffic.utilities


def test_batched_generator():

    iterable = iter(range(6))
    batch_size = 2

    expected = [(0, 1), (2, 3), (4, 5)]

    batched_generator = traffic.utilities.get_batched_generator(iterable, batch_size)
    actual = [next(batched_generator), next(batched_generator), next(batched_generator)]

    assert expected == actual
