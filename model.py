import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import softmax

class GoogleNet(tf.keras.Model):
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
