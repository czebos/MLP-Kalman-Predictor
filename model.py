import numpy as np
import tensorflow as tf
from tf.keras.layers import *
from tf.keras.activations import softmax

class GoogleNet(tf.keras.Model):
	def __init__(self):

		super(GoogleNet, self).__init__()
		self.batch_size = 128

        self.conv1 = Conv2D(112*112*64, (7,7), 2)
        self.pooling1 = MaxPooling2D((3,3), 2)
        self.conv2 = Conv2D(56*56*192, (3,3), 1)
        self.pooling2 = MaxPooling2D((3,3), 2)

        self.dropout = Dropour(.4)
        self.dense1 = tf.keras.layers.Dense(1000, activation = 'softmax')


	@tf.function
	def call(self, inputs):
		"""
        :params inputs, firing rates
		:return pos, estimated position for the marker labels
		"""

		# TODO:
		x = self.conv1(inputs)
        x = self.pooling1(x)

        x = self.conv2(x)
        x = self.pooling2(x)

        x = self.dropout(x)
        pos = self.dense1(x)

        return x

	def loss(self, pos, labels):
		"""
		Calculates the loss after one forward pass
		:return: the loss of the model as a tensor
		"""
		return tf.keras.losses.mean_squared_error(pos,labels)
