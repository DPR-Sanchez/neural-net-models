from neupy.layers import *
from neupy import algorithms
import numpy as np
from sklearn.model_selection import train_test_split
from random import seed
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing

training_set = np.genfromtxt('training.csv', delimiter=',')

np.random.seed(614)
seed(2)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
							  inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)

# Split lines into examples and labels
examples = training_set[:, :-1]
labels = training_set[:, -1:]

#takes size of second dimension which is the features count of the example set
input_size = np.size(examples,1)

# Divide dataset into training and test (60% training and 40% test)
training_examples, test_examples, training_labels, test_labels = train_test_split(examples, labels, test_size=0.4)
training_examples = preprocessing.normalize(training_examples)
test_examples = preprocessing.normalize(test_examples)

#model 1
sequential_net_one = join(
		Input(input_size),
		Linear(30),
		Sigmoid(30),
		Relu(30),
		Tanh(30),
		Tanh(30),
		Relu(1)
)


parsig = Sigmoid(30) >> Sigmoid(30)
partan = Tanh(30) >> Tanh(30)
parelu = Elu(30) >> Elu(30)
parchain_negative = Tanh(30) >> Elu(30)
parchain_zero = Sigmoid(30) >> Relu(30)

#model 2 - binary classifier
parralel_network_relu_out = Input(input_size) >> Linear(30) >> (parsig | partan | parelu | parchain_negative | parchain_zero) >>\
							Concatenate() >> Tanh(30) >> Relu(1)

#model 3
parralel_network_sig_out = Input(input_size) >> Linear(30) >> (parsig | partan | parelu | parchain_negative | parchain_zero) >>\
							Concatenate() >> Tanh(30) >> Sigmoid(1)

#model 4
parsig_two = Sigmoid(30) >> Sigmoid(30)
partan_two = Tanh(30) >> Tanh(30)
parelu_two = Elu(30) >> Elu(30)
parchain_negative_two = Tanh(30) >> Elu(30)
parchain_zero_two = Sigmoid(30) >> Relu(30)

parralel_funnel_network_relu_out = Input(input_size) >> Linear(30) >>\
									(parsig | partan | parelu | parchain_negative | parchain_zero) >>\
									Concatenate() >>(parsig_two | partan_two | parelu_two)>> Concatenate()>>\
									(parchain_negative_two | parchain_zero_two)>>\
									Tanh(30) >> Sigmoid(1)

networks = [
				sequential_net_one,
				parralel_network_relu_out,
				parralel_network_sig_out,
				parralel_funnel_network_relu_out
			]

network = networks[1]

optimizer = algorithms.Adam(
	network,
	loss='binary_crossentropy',
	verbose=True,
	regularizer=algorithms.l2(0.001)
)

optimizer.train(training_examples, training_labels, test_examples, test_labels, epochs=2)
optimizer.plot_errors()
pred = [1 if i > 0 else 0 for i in optimizer.predict(test_examples)]
print(test_labels.T)
print(pred)
for i in range(len(pred)):
	print("Correct" if pred[i] == test_labels[i] else "Incorrect")

accuracy = [1 if pred[i] == test_labels[i] else 0 for i in range(len(pred))].count(1) / len(pred)
print(f'{accuracy * 100:.2f}% accuracy')