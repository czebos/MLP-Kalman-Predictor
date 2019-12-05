import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import softmax

class Linear(tf.keras.Model):
	def __init__(self):

		super(GoogleNet, self).__init__()
		self.batch_size = 1024
		self.dense1 = tf.keras.layers.Dense(1024)
		self.dense2 = tf.keras.layers.Dense(512)
		self.dense3 = tf.keras.layers.Dense(256)
		self.dropout = Dropout(.4)
		self.dense4 = tf.keras.layers.Dense(87, activation = 'softmax')

	def call(self, inputs):
		"""
        :params inputs, firing rates
		:return pos, estimated position for the marker labels
		"""
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		x = self.dropout(x)
		pos = self.dense4(x)
		return pos

	def loss(self, pos, labels):
		"""
		Calculates the loss after one forward pass
		:return: the loss of the model as a tensor
		"""
		return tf.reduce_sum(tf.keras.losses.mean_squared_error(pos,labels))

def create_basicconv():
    model_m = tf.keras.Sequential()
    model_m.add(Dense((8190), input_shape=(260,)))
    model_m.add(Reshape((105, 78), input_shape=(8190,)))
    model_m.add(Conv1D(100, 10, activation='relu', input_shape=(105, 78)))
    model_m.add(Conv1D(100, 10, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(87, activation='softmax'))
    return model_m
