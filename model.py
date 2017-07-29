from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


class Model(object):

	def __init__(self):
		self = self

	def graph(self):

		self.N_CLASSES = 3

		self.LEARNING_RATE = 0.001
		self.BATCH_SIZE = 32
		# self.SKIP_STEP = 1
		self.DROPOUT = 0.5
		self.N_EPOCHS = 3

		# the model is conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax

		with tf.name_scope('data'):
			self.X = tf.placeholder(tf.float32, [None, 64, 64], name="X_placeholder")
			self.Y = tf.placeholder(tf.float32, [None, self.N_CLASSES], name="Y_placeholder")

		self.dropout = tf.placeholder(tf.float32, name='dropout')\

		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

		with tf.variable_scope('conv1') as scope:

			# first, reshape the image to [self.BATCH_SIZE, 64, 64, 1] to make it work with tf.nn.conv2d
			images = tf.reshape(self.X, shape=[-1, 64, 64, 1])
			kernel = tf.get_variable('kernels', [5, 5, 1, 32], 
				initializer=tf.truncated_normal_initializer())
			biases = tf.get_variable('biases', [32],
				initializer=tf.random_normal_initializer())
			conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
			conv1 = tf.nn.relu(conv + biases, name=scope.name)

			# output is of dimension self.BATCH_SIZE x 64 x 64 x 32

		with tf.variable_scope('pool1') as scope:
			pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			# output is of dimension self.BATCH_SIZE x 32 x 32 x 32

		with tf.variable_scope('conv2') as scope:
			# similar to conv1, except kernel now is of the size 5 x 5 x 32 x 64
			kernel = tf.get_variable('kernels', [5, 5, 32, 64],
				initializer=tf.truncated_normal_initializer())
			biases = tf.get_variable('biases', [64],
				initializer=tf.random_normal_initializer())
			conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
			conv2 = tf.nn.relu(conv + biases, name=scope.name)

			# output is of dimension self.BATCH_SIZE x 32 x 32 x 64

		with tf.variable_scope('pool2') as scope:
			# similar to pool1
			pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			# output is of dimension self.BATCH_SIZE x 16 x 16 x 64


		with tf.variable_scope('conv3') as scope:
			# similar to conv1, except kernel now is of the size 5 x 5 x 64 x 128
			kernel = tf.get_variable('kernels', [5, 5, 64, 128],
				initializer=tf.truncated_normal_initializer())
			biases = tf.get_variable('biases', [128],
				initializer=tf.random_normal_initializer())
			conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
			conv3 = tf.nn.relu(conv + biases, name=scope.name)

			# output is of dimension self.BATCH_SIZE x 16 x 16 x 128

		with tf.variable_scope('pool3') as scope:
			# similar to pool1
			pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			# output is of dimension self.BATCH_SIZE x 8 x 8 x 128

			# print("conv3: ", conv3.shape)
			# print("pool3: ", pool3.shape)

		with tf.variable_scope('conv4') as scope:
			# similar to conv1, except kernel now is of the size 5 x 5 x 64 x 128
			kernel = tf.get_variable('kernels', [3, 3, 128, 256],
				initializer=tf.truncated_normal_initializer())
			biases = tf.get_variable('biases', [256],
				initializer=tf.random_normal_initializer())
			conv = tf.nn.conv2d(pool3, kernel, strides=[1, 1, 1, 1], padding='SAME')
			conv4 = tf.nn.relu(conv + biases, name=scope.name)

			# output is of dimension self.BATCH_SIZE x 16 x 16 x 128

		with tf.variable_scope('pool4') as scope:
			# similar to pool1
			pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			# output is of dimension self.BATCH_SIZE x 8 x 8 x 128

			# print("conv3: ", conv3.shape)
			# print("pool3: ", pool3.shape)


		with tf.variable_scope('fc') as scope:
			# use weight of dimension 8 * 8 * 128 x 1024
			# input_features = 8 * 8 * 128
			input_features = 4 * 4 * 256
			w = tf.get_variable('weights', [input_features, 1024],
				initializer=tf.truncated_normal_initializer())
			b = tf.get_variable('biases', [1024],
				initializer=tf.random_normal_initializer())

			# reg = tf.contrib.layers.l2_regularizer(0.1, scope="L2")
			# tf.contrib.layers.apply_regularization(reg, )

			# reshape pool3 to 2 dimensional
			pool4 = tf.reshape(pool4, [-1, input_features])
			fc = tf.nn.relu(tf.matmul(pool4, w) + b, name='relu')

			fc = tf.nn.dropout(fc, self.dropout, name='relu_dropout')

		with tf.variable_scope('softmax_linear') as scope:
			w = tf.get_variable('weights', [1024, self.N_CLASSES],
				initializer=tf.truncated_normal_initializer())
			b = tf.get_variable('biases', [self.N_CLASSES],
				initializer=tf.random_normal_initializer())
			self.logits = tf.matmul(fc, w) + b

		with tf.name_scope('loss'):
			entropy = tf.nn.softmax_cross_entropy_with_logits(
				labels=self.Y, logits=self.logits)
			self.loss = tf.reduce_mean(entropy, name='loss')

		with tf.name_scope('summaries'):
			tf.summary.scalar('loss', self.loss)
			tf.summary.histogram('histogram loss', self.loss)
			summary_op = tf.summary.merge_all()

		with tf.name_scope('optimizer'):
			self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

		# self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss, 
		# 	global_step=self.global_step)

		with tf.name_scope('accuracy'):
			correct_preds = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

	def train(self, input, labels):
		g = Model.graph(self)
		with tf.Session(g) as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			# to visualize using TensorBoard
			merged = tf.summary.merge_all()
			writer = tf.summary.FileWriter('./my_graph/face', sess.graph)
			ckpt = tf.train.get_checkpoint_state(os.path.dirname(
				'checkpoints/convnet_face/checkpoint'))
			# if that checkpoint exists, restore from checkpoint
			# if ckpt and ckpt.model_checkpoint_path:
				# saver.restore(sess, ckpt.model_checkpoint_path)

			# initial_step = self.global_step.eval()

			start_time = time.time()
			n_batches = int(len(input) / self.BATCH_SIZE)

			for index in range(0, self.N_EPOCHS): # train the model self.n_epochs times
				total_loss = 0.0
				avg_accuracy_val = 0.0
				batch_count = 0
				for j in range(0, len(input), self.BATCH_SIZE):
					X_batch = input[j:j + self.BATCH_SIZE, :]
					Y_batch = labels[j:j + self.BATCH_SIZE, :]

					summary, _, loss_batch, accuracy_batch = sess.run(
						[merged, self.optimizer, self.loss, self.accuracy],
						feed_dict={
								self.X: X_batch,
								self.Y: Y_batch,
								self.dropout: self.DROPOUT
								})
					writer.add_summary(summary, j)
					total_loss += loss_batch
					avg_accuracy_val += accuracy_batch
					batch_count += 1
				avg_accuracy_val /= batch_count

				print('Average loss at epoch {}: {:5.1f}, Avg Acc: {}'.format(index + 1, 
							total_loss, avg_accuracy_val))
				saver.save(sess, 'checkpoints/convnet_face/face-convnet', index)

			print("Optimization Finished!")
			print("Total time: {0} seconds".format(time.time() - start_time))


	def test(self, input, labels, save_path):
		tf.reset_default_graph()
		g = Model.graph(self)
		with tf.Session(g) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, save_path=save_path)

			n_batches = int(len(input) / self.BATCH_SIZE)
			total_correct_preds = 0
			for i in range(0, len(input), self.BATCH_SIZE):
				X_batch = input[i:i + self.BATCH_SIZE, :]
				Y_batch = labels[i:i + self.BATCH_SIZE, :]

				_, loss_batch, logits_batch = sess.run([
					self.optimizer, self.loss, self.logits], 
					feed_dict={
							self.X: X_batch, 
							self.Y:Y_batch, 
							self.dropout: 1.0
							})

				preds = tf.nn.softmax(logits_batch)
				correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
				accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
				total_correct_preds += sess.run(accuracy)

			print("Test Accuracy {0}".format(total_correct_preds / len(input)))
			# print("Test Accuracy {0}".format(total_correct_preds / n_batches))


	def predict(self, input, save_path):
		tf.reset_default_graph()
		g = Model.graph(self)
		with tf.Session(g) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, save_path=save_path)

			logits_run = sess.run([tf.nn.softmax(self.logits)], 
				feed_dict={
					self.X: input,
					self.dropout: 1.0
					})

			def binary_class(logits):
				# quick and dirty LabelBinarizer inverse transform
				# print(logits_run)
				lab = []
				for i in logits:
					for j, k in i:
						if j > 0.5 and k <= 0.5:
							lab.append("Aman")
						elif j <= 0.5 and k > 0.5:
							lab.append("Ben")
						else:
							lab.append("Unknown")
				return lab

			def multi_class(logits):
				# quick and dirty LabelBinarizer inverse transform
				# print(logits_run)
				lab = []
				for i in logits:
					for j, k, r in i:
						if j == 1:
							lab.append("Ben")
						elif k == 1:
							lab.append("Aman")
						elif r == 1:
							lab.append("Unknown")
						else:
							lab.append("Nonee")
				return lab

		return multi_class(logits_run)
		# return logits_run