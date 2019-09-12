from random import seed
import csv

from neupy.layers import *
from neupy import algorithms
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf


def fetch_data_source(data_source:str, index:bool,dataset=False,headers=False):
	# data_source should be the string path to data csv

	training_set = pd.read_csv(data_source).to_numpy()


	#remove headers if present
	if headers:
		training_set = check_array(training_set[1:],force_all_finite=True)
	else:
		training_set = check_array(training_set,force_all_finite=True)
	training_set = preprocessing.normalize(training_set,axis=0)
	# Split lines into examples and labels
	examples = training_set[:, 1:-1] if index else training_set[:, :-1]
	labels = training_set[:, -1:]

	if dataset:
		return training_set, examples
	else:
		return examples,labels


def prediction(network, samples=[], labels=[], mode='', data_source='', index=False, save_location='', headers=False):

	if mode == 'accuracy':
		if data_source != '':
			samples, labels = fetch_data_source(data_source, index, headers=headers)

		prediction = network.predict(samples)

		#Using max min average as a rough translation to binary 1 0 for any activation function used as the output node
		prediction_average = (prediction.max()+prediction.min())/2

		prediction = [1 if i > prediction_average else 0 for i in network.predict(samples)]
		accuracy = [1 if prediction[i] == labels[i] else 0 for i in range(len(prediction))].count(1) / len(
			prediction)
		return f'{accuracy * 100:.2f}%'

	else:
		dataset, samples = fetch_data_source(data_source,index,dataset=True,headers=headers)
		opt_results = network.predict(samples)
		output = np.append(dataset, opt_results, axis=1)

		if headers:
			with open(data_source, 'r') as infile:
				reader = csv.DictReader(infile)
				fieldnames = reader.fieldnames

			fieldnames.append('prediction')
			df = pd.DataFrame(output)  # A is a numpy 2d array
			df.to_csv(f'{save_location}.csv', header=fieldnames, index=False)
		else:
			np.savetxt(f'{save_location}.csv', output , fmt="%d", delimiter=",")

def train_model(
					numpy_seed=614,
					tensor_seed=1234,
					ran_seed=2,
					data_source='training.csv',
					network_select='sequential 1',
					loss_function='binary_crossentropy',
					epochs_count=10,
					index=True,
					headers = False
				):
	examples, labels = fetch_data_source(data_source, index,headers=headers)
	np.random.seed(numpy_seed)
	seed(ran_seed)
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, 	inter_op_parallelism_threads=1)
	tf.set_random_seed(tensor_seed)
	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	keras.backend.set_session(sess)

	# takes size of second dimension which is the features count of the example set
	input_size = np.size(examples, 1)

	# Divide dataset into training and test (60% training and 40% test)
	training_examples, validation_examples, training_labels, validation_labels = train_test_split(examples, labels, test_size=0.4)

	# model 1
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

	# model 2 - binary classifier
	parralel_network_relu_out = Input(input_size) >> Linear(30) >> (
				parsig | partan | parelu | parchain_negative | parchain_zero) >> \
								Concatenate() >> Tanh(30) >> Relu(1)

	# model 3
	parralel_network_sig_out = Input(input_size) >> Linear(30) >> (
				parsig | partan | parelu | parchain_negative | parchain_zero) >> \
							   Concatenate() >> Tanh(30) >> Sigmoid(1)


	scale = int(input_size/10 * (2/3))+1

	parsig_two = Sigmoid(scale) >> BatchNorm() >> Dropout(proba=.4)>> Sigmoid(scale)
	partan_two = Tanh(scale) >> BatchNorm() >> Dropout(proba=.4)>> Tanh(scale)
	parelu_two = Elu(scale) >> BatchNorm() >> Dropout(proba=.4)>> Elu(scale)
	parchain_negative_two = Tanh(scale) >> BatchNorm() >> Dropout(proba=.4)>> Elu(scale)
	parchain_zero_two = Sigmoid(scale) >> BatchNorm() >> Dropout(proba=.4)>> Relu(scale)

	# model 4 - partially scales with input
	parralel_funnel_network_sig_out = Input(input_size) >> Linear(30) >> \
									   ((parsig | partan | parelu | parchain_negative | parchain_zero)\
									   |(parsig_two | partan_two | parelu_two)) >> Concatenate() >> \
									   (parchain_negative_two | parchain_zero_two) >> \
									   Concatenate() >> Tanh(30) >> Sigmoid(1)

	#model 5 - size scales with input
	concat_normdrop_one = Concatenate()>>BatchNorm()>>Dropout(proba=.3)
	concat_normdrop_two = Concatenate() >> BatchNorm() >> Dropout(proba=.2)
	concat_normdrop_three = Concatenate() >> BatchNorm() >> Dropout(proba=.2)
	concat_normdrop_four = Concatenate() >> BatchNorm() >> Dropout(proba=.1)
	autoscale_funnel_network_sig_out = Input(input_size) >> Linear(scale) >> \
									   ((parsig | partan | parelu | parchain_negative | parchain_zero) \
										| (parsig_two | partan_two | parelu_two)) >> concat_normdrop_one >> \
									   (parchain_negative_two | parchain_zero_two) >> \
									   concat_normdrop_two >>(Tanh(scale)|Elu(scale))>>concat_normdrop_three>>\
									   Tanh(scale) >> concat_normdrop_four >> Sigmoid(1)

	# model 6 - hybrid noisy parallel sequential
	concat_noisynormdrop_one = Concatenate() >> BatchNorm() >> Dropout(proba=.2) >> GaussianNoise(std=0.1)
	concat_noisynormdrop_two = Concatenate() >> BatchNorm() >> Dropout(proba=.2) >> GaussianNoise(std=0.1)
	concat_noisynormdrop_three = Concatenate() >> BatchNorm() >> Dropout(proba=.1) >> GaussianNoise(std=0.1)
	concat_noisynormdrop_four = Concatenate() >> BatchNorm() >> Dropout(proba=.1) >> GaussianNoise(std=0.1)
	noisy_para_seq = Input(scale)>>\
							Linear(scale)>>\
						 	(Tanh(scale)|LeakyRelu(scale))>>\
							concat_noisynormdrop_one>>\
						 	(Elu(scale)|LeakyRelu())>>\
							concat_normdrop_two>>\
						 	(Elu(scale)|Tanh(scale))>>\
							concat_normdrop_three >>\
							Tanh(scale)>>\
							(Tanh(scale) | Elu(scale))>>\
							concat_normdrop_four>>\
							HardSigmoid(1)


	net_select_dict = {
		'sequential 1':sequential_net_one,
		'par net relu':parralel_network_relu_out,
		'par net sig':parralel_network_sig_out,
		'funnel net':parralel_funnel_network_sig_out,
		'scaling funnel net':autoscale_funnel_network_sig_out,
		'noisy parallel sequential':noisy_para_seq
	}

	network = net_select_dict[network_select]

	optimizer = algorithms.Adam(
		network,
		loss=loss_function,
		verbose=False,
		regularizer=algorithms.l2(0.001),
		shuffle_data=True
	)

	optimizer.train(training_examples, training_labels, validation_examples, validation_labels, batch_size=256, epochs=epochs_count)
	accuracy = prediction(optimizer,validation_examples,validation_labels,'accuracy')



	return (optimizer,network,accuracy)
