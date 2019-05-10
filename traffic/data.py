"""
Module with data related functionality
"""
import copy
import os
import pickle

import imgaug
import keras
import numpy as np
import sklearn.utils
import cv2


def get_datasets(datasets_directory):
    """
    Given directory to where train and validation datasets are stored, return them
    :param datasets_directory: str, path to datasets directory
    :return: tuple (train_data, validation_data). Both elements are dictionaries
    """

    training_path = os.path.join(datasets_directory, "train.p")
    validation_path = os.path.join(datasets_directory, "valid.p")

    with open(training_path, mode='rb') as training_file:

        training_data = pickle.load(training_file)

    with open(validation_path, mode='rb') as validation_file:

        validation_data = pickle.load(validation_file)

    training_data["dataset_name"] = "training"
    validation_data["dataset_name"] = "validation"

    return training_data, validation_data


def get_data_generator(images, labels, batch_size, shuffle=False, augmentations_pipeline=None):
    """
    Creates a generator that combines features and labels into batches, optionally shuffling and augmenting
    :param images: list of images
    :param labels: list of ints
    :param batch_size: int, batch size
    :param shuffle: boolean
    :param augmentations_pipeline: optional imgaug augmentation pipeline instance, if used data is augmented
    :return: generator
    """

    local_images = copy.deepcopy(images)
    local_labels = copy.deepcopy(labels)

    images_batch = []
    labels_batch = []

    while True:

        # Shuffle features and labels if desired
        if shuffle is True:

            local_images, local_labels = sklearn.utils.shuffle(local_images, local_labels)

        # Iterate over image and labels
        for image, label in zip(local_images, local_labels):

            images_batch.append(image)
            labels_batch.append(label)

            # Keep on adding features and labels to their respective batches until requested size is reached
            if len(images_batch) == batch_size:

                if augmentations_pipeline is not None:

                    images_batch = augmentations_pipeline.augment_images(images_batch)

                # Yield batch
                yield np.array(images_batch), np.array(labels_batch)

                # Clear containers so they can start accumulating new batches
                images_batch.clear()
                labels_batch.clear()


class TrafficDataBunch:
    """
    Data bunch class representing traffic data
    """

    def __init__(self, datasets_directory, categories_count, batch_size):
        """
        Constructor
        :param datasets_directory: str, path to datasets directory
        :param categories_count: int, number of categories
        :param batch_size: int, batch size
        """

        training_data, validation_data = get_datasets(datasets_directory)

        # Flip images to BGR order
        training_images = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in training_data["features"]])
        validation_images = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in validation_data["features"]])

        # Turn them to floats in <0,1> range
        training_images = training_images.astype(np.float32) / 255
        validation_images = validation_images.astype(np.float32) / 255

        augmentations_pipeline = imgaug.augmenters.SomeOf(
            (0, None),
            [imgaug.augmenters.Affine(rotate=(-15, 15)),
             imgaug.augmenters.Add((-0.4, 0.4), per_channel=0.5),
             imgaug.augmenters.ContrastNormalization((0.5, 1.5))],
            random_order=True)

        self.training_data_generator = get_data_generator(
            training_images,
            keras.utils.to_categorical(training_data["labels"], num_classes=categories_count),
            batch_size=batch_size, shuffle=True, augmentations_pipeline=augmentations_pipeline)

        self.training_samples_count = len(training_data["labels"])

        self.validation_data_generator = get_data_generator(
            validation_images,
            keras.utils.to_categorical(validation_data["labels"], num_classes=categories_count),
            batch_size=batch_size, shuffle=False)

        self.validation_samples_count = len(validation_data["labels"])
