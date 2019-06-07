import yaml


def load_yaml(path):
	with open(path) as file:
		return yaml.load(file)