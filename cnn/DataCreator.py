import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from tqdm import tqdm # Für den Fortschrittsbalken
import pickle

class DataCreator:

    def __init__(self, image_size):
        """
        :param image_size: Defines the size of each image wich will be prepared for neuronal network
        """
        self.__image_size = image_size
        self.__X = []
        self.__y = []
        self.__categories = None

    @property
    def categories(self):
        return self.__categories

    @property
    def image_size(self):
        return self.__image_size

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y

    @X.setter
    def X(self, new_X):
        self.__X = new_X

    @y.setter
    def y(self, new_y):
        self.__y = new_y

    def prepare_data_set(self, directory) -> list:
        """
        :param directory: The directory containing the trainings-dataset

        Every image for each category will be converted to a multi-dimensional array containing integer values.
        By using a gray-color-filter the convolutional neuronal network is capable of detecting even more distinctive patterns
        such as edges.
        Further more images will be resized to a certain value which was passed to the TrainingsDataCreator-Object.

        Last but not least the trainings-data will be shuffeled to avoid that the neuronal network begins to memorize certain patterns

        :return: Well prepared data-set.
        """
        categories = next(os.walk(directory))[1]
        self.__categories = categories
        data_set = []
        for category_index in tqdm(range(0, len(categories))):
            category = categories[category_index]
            path = os.path.join(directory, category)
            class_num = categories.index(category)
            for img in os.listdir(path):
                try:
                    filepath = os.path.join(path, img)
                    data_set.append([self.__image_to_array(filepath), class_num])
                except Exception as e:
                    print(e)
                    pass
        random.shuffle(data_set)
        for features, label in data_set:
            self.__X.append(features)
            self.__y.append(label)
        self.__normalize(data_set)

    def __normalize(self, data_set):
        self.__X = np.array(self.__X).reshape(-1, self.__image_size, self.__image_size, 3)
        self.__X = self.__X.astype('float32')
        self.__X /= 255

    def __image_to_array(self, filepath):
        img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
        return cv2.resize(img_array, (self.__image_size, self.__image_size))

    def prepare_image(self, filepath):
        return self.__image_to_array(filepath).reshape(-1, self.__image_size, self.__image_size, 3)

    def save(self):
        self.__save(self.__X, "X")
        self.__save(self.__y, "y")

    def __save(self, set, name):
        pickle_out = open(name +".pickle", "wb")
        pickle.dump(set, pickle_out)
        pickle_out.close()

    def load(self):
        self.__X = self.__load("X")
        self.__y = self.__load("y")

    def __load(self, name):
        pickle_in = open(name + ".pickle", "rb")
        return pickle.load(pickle_in)
