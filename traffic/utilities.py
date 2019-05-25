"""
Module with various utilities
"""

import argparse
import os
import logging

import yaml
import cv2
import numpy as np


def get_yaml_configuration(command_line_arguments):
    """
    Reads yaml config from path provided in command line arguguments
    :param command_line_arguments: list, command line arguments
    :return: dictionary
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', action="store", required=True)
    arguments = parser.parse_args(command_line_arguments)

    with open(arguments.config_path) as file:
        config = yaml.safe_load(file)

        return config


def binary_string_image_to_numpy_image(binary_string):
    """
    Decodes binary string into a numpy array representation of an image
    :param binary_string: bytes structure
    :return: numpy array
    """

    flat_numpy_array = np.frombuffer(binary_string, np.uint8)
    return cv2.imdecode(flat_numpy_array, cv2.IMREAD_ANYCOLOR)
