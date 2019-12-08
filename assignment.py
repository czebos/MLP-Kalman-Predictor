import os
import tensorflow as tf
import numpy as np
from preprocess import *
import sys
import math
from model import *

EPOCH_SIZE = 10
BATCH_SIZE = 128

def train(model, train_inputs, train_labels, is_kalman):
	"""
	Runs through one epoch - all training examples.

	:param model: the initilized model to use for forward and backward pass
	:param train_inputs for training
	:param train_labels for training
	:return: None
	"""
	optmizer = tf.keras.optimizers.Adam(learning_rate=0.001)
	indices = [i for i in range(len(train_inputs))]
	shuffled = tf.random.shuffle(indices)
	inputs = tf.gather(train_inputs, shuffled)
	labels = tf.gather(train_labels, shuffled)
	loss_batch = 0

	for j in range(int((len(train_inputs) /  BATCH_SIZE))):
		sub_inputs = np.array(inputs[j*BATCH_SIZE: (j+1)*BATCH_SIZE])
		if is_kalman:
			sub_inputs = np.array(inputs[j*BATCH_SIZE + 1: (j+1)*BATCH_SIZE + 1])
			states = np.array(labels[j*BATCH_SIZE: (j+1)*BATCH_SIZE])
		sub_labels = np.array(labels[j*BATCH_SIZE + 1: (j+1)*BATCH_SIZE + 1])

		with tf.GradientTape() as tape:
			if is_kalman:
				predictions = model(sub_inputs, states)
				loss = model.loss(predictions, sub_labels)

			else:
				predictions = model(sub_inputs)
				loss = loss_f(predictions, sub_labels)
		if j % 1000:
			break

		gradients  = tape.gradient(loss, model.trainable_variables)
		optmizer.apply_gradients(zip(gradients, model.trainable_variables))
	return None

def loss_f(predictions, labels):
	return tf.reduce_sum(tf.keras.losses.mean_squared_error(predictions,labels))

def test(model, test_inputs, test_labels, is_kalman):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initilized model to use for forward and backward pass
	:param test_inputs for testing
	:param test_labels for testing
	:returns: perplexity of the test set, per symbol accuracy on test set
	"""

	accuracy = 0
	total_pred = 0

	for j in range(math.ceil(int((len(test_inputs) / BATCH_SIZE)))):
		inputs = np.array(test_inputs[j*BATCH_SIZE: (j+1)*BATCH_SIZE])
		if is_kalman:
			inputs = np.array(test_inputs[(j*BATCH_SIZE) + 1: ((j+1)*BATCH_SIZE) + 1])
			states = np.array(test_labels[j*BATCH_SIZE: (j+1)*BATCH_SIZE])
		sub_labels = np.array(test_labels[j*BATCH_SIZE + 1: (j+1)*BATCH_SIZE + 1])

		if is_kalman:
			predictions = model(inputs, states)
		else:
			predictions = model(inputs)
		total_pred += 1

		loss = loss_f(predictions, sub_labels)
		accuracy += loss

	return (accuracy/total_pred)

def main():

	train_inputs, train_labels, test_inputs, test_labels = get_data('./../COS071212_mocap_processed.mat')
	conv_model = create_conv()
	gru_model = create_gru()
	linear_model = Linear()
	kalman_filter = Kalman()

	for i in range(EPOCH_SIZE):
		train(gru_model, train_inputs, train_labels, False)
		print("GRU Model Loss:" + str(test(gru_model, test_inputs, test_labels, False)))
		print("")
		train(linear_model, train_inputs, train_labels, False)
		print("Linear Model Loss:" + str(test(linear_model, test_inputs, test_labels, False)))
		print("")
		train(conv_model, train_inputs, train_labels, False)
		print("Conv Model Loss:" + str(test(conv_model, test_inputs, test_labels, False)))
		print("")
		train(kalman_filter, train_inputs, train_labels, True)
		print("Kalman Filter Loss:" + str(test(kalman_filter, test_inputs, test_labels, True)))
		print("")


if __name__ == '__main__':
   main()
