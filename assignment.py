import os
import tensorflow as tf
import numpy as np
from preprocess import *
import sys
import math
from model import GoogleNet

EPOCH_SIZE = 10

def train(model, train_inputs, train_labels):
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

	for j in range(int((len(train_inputs) / model.batch_size))):
		sub_inputs = np.array(inputs[j*model.batch_size: (j+1)*model.batch_size])
		sub_labels = np.array(labels[j*model.batch_size: (j+1)*model.batch_size])

		with tf.GradientTape() as tape:
			predictions = model(sub_inputs)
			loss = model.loss(predictions, sub_labels)
		loss_batch += loss
		if j%10000:
			print(loss_batch / 10000)
			loss_batch = 0
		gradients  = tape.gradient(loss, model.trainable_variables)
		optmizer.apply_gradients(zip(gradients, model.trainable_variables))
	return None

def test(model, test_inputs, test_labels):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initilized model to use for forward and backward pass
	:param test_inputs for testing
	:param test_labels for testing
	:returns: perplexity of the test set, per symbol accuracy on test set
	"""

	accuracy = 0
	total_pred = 0

	for j in range(math.ceil(int((len(test_inputs) / model.batch_size)))):
		sub_inputs = np.array(test_inputs[j*model.batch_size: (j+1)*model.batch_size])
		sub_labels = np.array(test_labels[j*model.batch_size: (j+1)*model.batch_size])

		predictions = model(sub_inputs)
		total_pred += 1
		loss = model.loss(predictions, sub_labels)
		accuracy += loss

	return (accuracy/total_pred)

def main():

	train_inputs, train_labels, test_inputs, test_labels = get_data('./../COS071212_mocap_processed.mat')
	model = GoogleNet()
	
	for i in range(EPOCH_SIZE):
		train(model, train_inputs, train_labels)
	print(test(model, test_inputs, test_labels))


if __name__ == '__main__':
   main()
