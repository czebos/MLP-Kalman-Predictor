import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import softmax
import tensorflow_probability as tfp
from tensorflow_probability import sts


class Linear(tf.keras.Model):
	"""
	Creates a Basic Linear model
	"""
	def __init__(self):

		super(Linear, self).__init__()
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


def create_conv():
	"""
	Creates a convolutional model
	"""
	model = tf.keras.Sequential()
	model.add(Dense((8190), input_shape=(260,)))
	model.add(Reshape((105, 78), input_shape=(8190,)))
	model.add(Conv1D(100, 10, activation='relu', input_shape=(105, 78)))
	model.add(Conv1D(100, 10, activation='relu'))
	model.add(MaxPooling1D(3))
	model.add(Conv1D(160, 10, activation='relu'))
	model.add(Conv1D(160, 10, activation='relu'))
	model.add(GlobalAveragePooling1D())
	model.add(Dropout(0.5))
	model.add(Dense(87, activation='softmax'))
	return model

def create_gru():
	"""
	Creates a GRU model
	"""
    model = tf.keras.Sequential()
    model.add(Dense((256), input_shape=(260,)))
    model.add(Reshape((8, 32), input_shape=(256,)))
    model.add(GRU(input_shape=(8, 32), units = 2048, activation='tanh', return_sequences=True))
    model.add(Dropout(0.3))

    model.add(GRU(1024, activation='tanh', return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(87, activation = 'softmax'))
    return model

class Kalman(tf.keras.Model):
	"""
	TensorFlow representation of the Kalman filter
	"""
	def __init__(self):
		super(Kalman, self).__init__()
		self.dense = tf.keras.layers.Dense(87, activation = 'softmax')

	def call(self, inputs, state):
		"""
        :params inputs, firing rates
		:return pos, estimated position for the marker labels
		"""
		x = self.dense(inputs)
		normal = np.random.normal(0, 1 ,size=x.shape)
		observation_model = normal + x
		pos = tf.add(observation_model, state)
		return pos

	def loss(self, pos, labels):
		"""
		Calculates the loss after one forward pass
		:return: the loss of the model as a tensor
		"""
		return tf.reduce_sum(tf.keras.losses.mean_squared_error(pos,labels))\
