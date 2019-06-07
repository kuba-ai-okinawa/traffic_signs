"""
doc string here
"""
import yaml


def load_yaml(path):
    """
    doc string here
    """
    with open(path) as file:
        return yaml.load(file)
