from random import seed
import collections
import csv
import math
import operator

from neupy.layers import *
from neupy import algorithms
from neupy import architectures
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf


def fetch_data_source(data_source:str, index:bool,dataset=False,headers=False,training=False):
	# data_source should be the string path to data csv
	#remove headers if present
	if headers:
		training_set = pd.read_csv(data_source).to_numpy()[1:]
	else:
		training_set = pd.read_csv(data_source).to_numpy()

	imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp.fit(training_set)
	#multiple layers of nan & inf filters for data source because some where getting through and causing errors downstream in the code
	training_set = check_array(np.nan_to_num(imp.transform(training_set)),force_all_finite=True)

	if training:
		last_column_index = len(training_set[0])-1
		last_column = training_set[:,last_column_index]
		set_total = len(last_column)
		occurrence = collections.Counter(last_column)

		min_occurrent = min(occurrence.items(),key=operator.itemgetter(1))
		sample_size = min_occurrent[1]
		min_occurrent = min_occurrent[0]

		max_occurrent = max(occurrence.items(),key=operator.itemgetter(1))[0]

		# select all instances of the binary label that occurs the least
		min_label_array= training_set[training_set[:, last_column_index] == min_occurrent ]

		# select sample_size number of instances of the binary label that occurs the most
		max_label_array = training_set[training_set[:, last_column_index] == max_occurrent]
		row_i = np.random.choice(max_label_array.shape[0], sample_size)
		max_label_array = max_label_array[row_i, :]

		training_set = np.concatenate((min_label_array, max_label_array), axis=0)
		np.random.shuffle(training_set)

	# Split lines into examples and labels
	examples = training_set[:, 1:-1] if index else training_set[:, :-1]
	labels = training_set[:, -1:]

	examples = preprocessing.normalize(examples,axis=0)

	if dataset:
		return training_set, examples
	else:
		return examples,labels


def prediction(network, samples=[], labels=[], mode='', data_source='', index=False, save_location='', headers=False, training=False):

	if mode == 'accuracy':
		if data_source != '':
			samples, labels = fetch_data_source(data_source, index, headers=headers)

		prediction = [1 if i > .5 else 0 for i in network.predict(samples)]

		#1 if prediction matches label
		comparison = [1 if prediction[i] == labels[i] else 0 for i in range(len(prediction))]

		accuracy = comparison.count(1) / len(	prediction)
		predicted_zeros = prediction.count(0)
		predicted_ones = prediction.count(1)
		count_actual = collections.Counter(labels[:,len(labels[0])-1])
		zero_actual = count_actual[0]
		one_actual = count_actual[1]
		zero_delta = predicted_zeros - zero_actual
		one_delta = predicted_ones - one_actual
		min_bound = int(math.floor(prediction.count(0)* accuracy))
		max_bound = int(math.ceil(prediction.count(0)*(2-accuracy)))
		return f'{accuracy * 100:.2f}%',f'{zero_actual}/{one_actual}',f'{predicted_zeros}/{predicted_ones}',f'{zero_delta}/{one_delta}',f'{min_bound}/{max_bound}'

	else:
		dataset, samples = fetch_data_source(data_source,index,dataset=True,headers=headers, training=training)
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
	examples, labels = fetch_data_source(data_source, index,headers=headers,training=True)
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
	sub_model_1 = Linear(30)>>	Sigmoid(30)>>Relu(30)>>	Tanh(30)>>Tanh(30)>> Relu(1)

	sequential_net_one = Input(input_size) >> sub_model_1



	# model 2
	mod2_parsig = Sigmoid(30) >> Sigmoid(30)
	mod2_partan = Tanh(30) >> Tanh(30)
	mod2_parelu = Elu(30) >> Elu(30)
	mod2_parchain_negative = Tanh(30) >> Elu(30)
	mod2_parchain_zero = Sigmoid(30) >> Relu(30)

	sub_model_2 = Linear(30) >>\
					(mod2_parsig | mod2_partan | mod2_parelu | mod2_parchain_negative | mod2_parchain_zero) >> \
					Concatenate() >> Tanh(30)>> Relu(1)

	parralel_network_relu_out = Input(input_size) >> sub_model_2

	# model 3
	mod3_parsig = Sigmoid(30) >> Sigmoid(30)
	mod3_partan = Tanh(30) >> Tanh(30)
	mod3_parelu = Elu(30) >> Elu(30)
	mod3_parchain_negative = Tanh(30) >> Elu(30)
	mod3_parchain_zero = Sigmoid(30) >> Relu(30)
	sub_model_3 = Linear(30) >>\
					(mod3_parsig | mod3_partan | mod3_parelu | mod3_parchain_negative | mod3_parchain_zero) >> \
					Concatenate() >> Tanh(30)>> Sigmoid(1)

	parralel_network_sig_out = Input(input_size) >> sub_model_3


	scale = int(input_size * 1.5)+1

	mod4_sig = Sigmoid(30) >> Sigmoid(30)
	mod4_tan = Tanh(30) >> Tanh(30)
	mod4_elu = Elu(30) >> Elu(30)
	mod4_sub_negative = Tanh(30) >> Elu(30)
	mod4_sub_zero = Sigmoid(30) >> Relu(30)
	mod4_sig_2 = Sigmoid(scale) >> BatchNorm() >> Dropout(proba=.4)>> Sigmoid(scale)
	mod4_tan_2 = Tanh(scale) >> BatchNorm() >> Dropout(proba=.4)>> Tanh(scale)
	mod4_elu_2 = Elu(scale) >> BatchNorm() >> Dropout(proba=.4)>> Elu(scale)
	mod4_sub_negative_2 = Tanh(scale) >> BatchNorm() >> Dropout(proba=.4)>> Elu(scale)
	mod4_sub_zero_two = Sigmoid(scale) >> BatchNorm() >> Dropout(proba=.4)>> Relu(scale)

	# model 4 - partially scales with input
	sub_model_4 = Linear(30) >> \
					(mod4_sig | mod4_tan | mod4_elu | mod4_sub_negative | mod4_sub_zero|mod4_sig_2 | mod4_tan_2 | mod4_elu_2) >>\
					Concatenate() >> (mod4_sub_negative_2 | mod4_sub_zero_two) >> Concatenate() >> Tanh(30)>> Sigmoid(1)

	parralel_funnel_network_sig_out = Input(input_size)>> sub_model_4

	#model 5 - size scales with input
	mod5_parsig = Sigmoid(30) >> Sigmoid(30)
	mod5_partan = Tanh(30) >> Tanh(30)
	mod5_parelu = Elu(30) >> Elu(30)
	mod5_parchain_negative = Tanh(30) >> Elu(30)
	mod5_parchain_zero = Sigmoid(30) >> Relu(30)
	mod5_sig_2 = Sigmoid(scale) >> BatchNorm() >> Dropout(proba=.4) >> Sigmoid(scale)
	mod5_tan_2 = Tanh(scale) >> BatchNorm() >> Dropout(proba=.4) >> Tanh(scale)
	mod5_elu_2 = Elu(scale) >> BatchNorm() >> Dropout(proba=.4) >> Elu(scale)
	mod5_sub_negative_2 = Tanh(scale) >> BatchNorm() >> Dropout(proba=.4) >> Elu(scale)
	mod5_sub_zero_two = Sigmoid(scale) >> BatchNorm() >> Dropout(proba=.4) >> Relu(scale)

	concat_normdrop_one = Concatenate()>>BatchNorm()>>Dropout(proba=.3)
	concat_normdrop_two = Concatenate() >> BatchNorm() >> Dropout(proba=.2)
	concat_normdrop_three = Concatenate() >> BatchNorm() >> Dropout(proba=.2)
	concat_normdrop_four = Concatenate() >> BatchNorm() >> Dropout(proba=.1)
	sub_model_5 = Linear(scale) >> \
					(mod5_parsig | mod5_partan | mod5_parelu | mod5_parchain_negative | mod5_parchain_zero | mod5_sig_2 | mod5_tan_2 | mod5_elu_2)>>\
				  	concat_normdrop_one >> 	(mod5_sub_negative_2 | mod5_sub_zero_two) >> \
					concat_normdrop_two >>(Tanh(scale)|Elu(scale))>>concat_normdrop_three>>\
					Tanh(scale) >> concat_normdrop_four>> Sigmoid(1)

	autoscale_funnel_network_sig_out = Input(input_size) >> sub_model_5

	# model 6 - hybrid noisy parallel sequential
	fourth = int(scale/4)
	thirds = int(scale/3)

	concat_noisynormdrop_one = Concatenate() >> ((GaussianNoise(std=1) >> BatchNorm() >> Dropout(proba=.7)) | ( Identity()>> Dropout(proba=.7)))>> Concatenate()
	concat_noisynormdrop_two = Concatenate()>> ((GaussianNoise(std=1) >> BatchNorm() >> Dropout(proba=.7)) | ( Identity()>> Dropout(proba=.7)))>> Concatenate()
	concat_noisynormdrop_three = Concatenate() >> ((GaussianNoise(std=1) >> BatchNorm() >> Dropout(proba=.7)) | ( Identity()>> Dropout(proba=.7)))>> Concatenate()

	sub_tri = (Elu(fourth)|Tanh(fourth)) >> Concatenate() >> Dropout(proba=.3)>> Sigmoid(fourth)
	sub_tri_leaky_relu = (LeakyRelu(fourth)|Tanh(fourth))>>Concatenate()>>LeakyRelu(fourth)>>LeakyRelu(fourth)

	sub_model_6 = 	Linear(thirds)>>\
						(Tanh(fourth)|Elu(fourth)|LeakyRelu(fourth)|sub_tri_leaky_relu|sub_tri)>>\
						concat_noisynormdrop_one>>\
						(Tanh(fourth)>>Tanh(fourth)|Elu(fourth)>>Elu(fourth)|Sigmoid(fourth)>>Sigmoid(fourth))>>\
						concat_noisynormdrop_two >>\
						(Tanh(fourth)|Elu(fourth)|LeakyRelu(fourth)|Sigmoid(fourth))>>\
						concat_noisynormdrop_three>> Sigmoid(1)

	noisy_para_seq = Input(input_size)>> sub_model_6

	#model 7  - model 1-6 mixture of models
	models_mixture = architectures.mixture_of_experts([sequential_net_one,parralel_network_sig_out,parralel_funnel_network_sig_out,autoscale_funnel_network_sig_out,noisy_para_seq])


	net_select_dict = {
		'sequential 1':sequential_net_one,
		'par net relu':parralel_network_relu_out,
		'par net sig':parralel_network_sig_out,
		'funnel net':parralel_funnel_network_sig_out,
		'scaling funnel net':autoscale_funnel_network_sig_out,
		'noisy parallel sequential':noisy_para_seq,
		'models_mixture':models_mixture
	}

	network = net_select_dict[network_select]

	optimizer = algorithms.Adam(
		network,
		loss=loss_function,
		verbose=False,
		regularizer=algorithms.l2(0.00001),
		shuffle_data=True
	)

	optimizer.train(training_examples, training_labels, validation_examples, validation_labels, batch_size=256, epochs=epochs_count)
	accuracy = prediction(optimizer,validation_examples,validation_labels,'accuracy')



	return (optimizer,network,accuracy)
