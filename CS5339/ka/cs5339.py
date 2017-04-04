from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import csv

import tensorflow as tf
import numpy as np


FLAGS = None


class DataSet(object):
	def __init__(self):
		print('init()')	

		
		#=== train data ===
		train_data=np.genfromtxt('processed_train_data.csv', dtype='float', delimiter=',')
		self.train_data= train_data

		#==== train label ===
		train_labels=np.genfromtxt('processed_train_labels.csv', dtype='int', delimiter=',')
		
		num_labels=train_labels.shape[0]
		
		index_offset=np.arange(num_labels)*26	
		labels_one_hot=np.zeros((num_labels, 26))
		labels_one_hot.flat[index_offset+train_labels.ravel()]=1

		self.train_labels=labels_one_hot

		#=== test data ===
		test_data=np.genfromtxt('processed_test_data.csv', dtype='float', delimiter=',')
		self.test_data= test_data 

		#=== result data ===
		result_data=np.genfromtxt('train.csv', dtype='string', delimiter=',')
		result_data=result_data[:,[0,1]]
		self.result_data=result_data

		test_ids=np.genfromtxt('test.csv', dtype='string', delimiter=',')	
		test_ids=test_ids[:,[0]]
		test_ids=np.delete(test_ids,0,0)
		self.test_ids=test_ids
		
		#===========================
		#print(self.train_data)
		#print(self.train_labels)
		#print(self.test_ids)

		print('train_data:'+ str(self.train_data.shape))
		print('train_labels:'+ str(self.train_labels.shape))
		print('test_ids:'+ str(self.test_ids.shape))

		self.start_index=0

	def next_batch(self, batch_size):
		#print('next_bacth()')

		print(self.start_index)
		xs=self.train_data[self.start_index:self.start_index+batch_size, :]
		ys=self.train_labels[self.start_index:self.start_index+batch_size, :]

		#print(xs)
		#print(ys)

		self.start_index+=batch_size
		
		return xs,ys


def main(_):
	print('main()')

	data= DataSet()
	
	num_of_features=16

	#create the model
	x=tf.placeholder(tf.float32, [None, num_of_features])
	W=tf.Variable(tf.zeros([num_of_features, 26]))
	b=tf.Variable(tf.zeros([26]))
	y=tf.matmul(x, W)+b

	#define loss and optimizer
	y_=tf.placeholder(tf.float32, [None, 26])

	#
	cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess=tf.InteractiveSession()
	tf.global_variables_initializer().run()

	#train: num of samples: 41680
	total_batch = 416
	for _ in range(total_batch):
		batch_xs, batch_ys = data.next_batch(100)
		rr=sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

	#classify
	#print(data.test_data)
	classification =sess.run(tf.argmax(y, axis=1), feed_dict={x:data.test_data})
	classification = classification.reshape(10473,1)
	#for x in range(10473):
	#	print(classification[x])

	sess.close()

	rr=np.concatenate((data.test_ids, classification), axis=1)
	for row in rr:
		asciic=int(row[1]) + 97
		if asciic >=0 and asciic <=256:
			row[1]= chr(asciic)
		else:
			print (asciic)
	result_data=data.result_data
	result_data=np.concatenate((result_data, rr), axis=0)	
	np.savetxt('ChenShaozhuang-A0134531R.csv', result_data, fmt='%s', delimiter=',')	



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
