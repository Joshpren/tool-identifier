import json
import pickle

import cv2
import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras import models
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from numpy import shape
from keras.utils import np_utils

from cnn.DataCreator import DataCreator, DataCreator


class ConvolutionalNeuronalNetwork:

        def __init__(self, image_size):
                self.__dc = DataCreator(image_size)
                self.__categories = None
                try:
                        self.__load()
                except:
                        self.__cnn = self.__create_model()

        def force(self):
                self.__cnn = self.__create_model()

        def __create_model(self):
                cnn = Sequential()
                cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', input_shape=[75, 75, 3]))
                cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
                cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu'))
                cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
                cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
                cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
                cnn.add(tf.keras.layers.Flatten())
                cnn.add(tf.keras.layers.Dense(256, activation='relu'))
                cnn.add(tf.keras.layers.Dense(128, activation='relu'))
                cnn.add(tf.keras.layers.Dense(64, activation='relu'))
                cnn.add(tf.keras.layers.Dense(8, activation='softmax'))

                return cnn

        def train(self, train_directory, epochs):
                self.__dc.prepare_data_set(train_directory)
                self.__cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                self.__cnn.fit(x=self.__dc.X, y=np.array(self.__dc.y), epochs=epochs)
                self.__categories = self.__dc.categories
                self.__save()


        def __save(self):
                self.__cnn.save('C:/Users/joshu/PycharmProjects/tool-identifier/cnn/model')
                try:
                        with open('C:/Users/joshu/PycharmProjects/tool-identifier/cnn/categories.txt', 'w') as fp:
                                json.dump(self.__categories, fp)
                except FileNotFoundError as e:
                        print(e)

        def __load(self):
                self.__cnn = models.load_model("C:/Users/joshu/PycharmProjects/tool-identifier/cnn/model")
                try:
                        with open('C:/Users/joshu/PycharmProjects/tool-identifier/cnn/categories.txt', 'r') as fp:
                                self.__categories = json.load(fp)

                except FileNotFoundError:
                        print("The 'docs' directory does not exist")
                print(self.__categories)

        def predict(self, image_filepath):
                array = self.__dc.prepare_image(image_filepath)
                return self.__cnn.predict(array)



        def resolve_prediction(self, prediction):
                index = 0
                for prop in prediction[0]:
                        if int(prop) == 1:
                                print(self.__categories[index])
                                break
                        index += 1




cnn = ConvolutionalNeuronalNetwork(75)
#cnn.force()
#cnn.train('C:/Users/joshu/PycharmProjects/tool-identifier/resources/archive/train_data/train_data', 15)
prediction = cnn.predict("C:/Users/joshu/PycharmProjects/tool-identifier/resources/test/41Eb9hWTgmL._AC_SY355_.jpg")
print(prediction)
cnn.resolve_prediction(prediction)


