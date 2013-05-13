#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        Neural Network Learning
# Purpose:
#
# Author:      Ilya Pershin
#
# Created:     14.03.2013
# Copyright:   (c) 11 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#
import os
os.chdir('C:/SNN/temp')
import numpy as np
import neural_network
import csv
import matplotlib.pyplot as pp

np.random.seed(111)

# Parameters:
hidden_1 = [8]  # Structure of the first subnetwork
hidden_2 = [8]  # Structure of the second subnetwork
epsilon = 0.2  # Limitation of  initial eights
alpha = 0.1  # Learning rate
R = 0.01  # Coefficient of regularization
M = 0.001  # Moment
e = 1e-4  # value of weights changing in the gradient check function
epochs_count = 10
batches_count = 20
data_proportion = 0.3
file = 'shugina_orig.csv'
online_learning = 'on' # Set 'on' to turn on online learing (one example per iteration)


# BEGINNING

# Create test data set
#X = np.arange(input_size * m).reshape(m, input_size)
#input_relation = np.arange(relation_in_size * m).reshape(m, relation_in_size)
#Y = np.ones((m * output_size))
#for l in range(m * output_size):
#    Y[l] = np.random.choice(2)
#Y = Y.reshape(m, output_size)

# Import data from file set
[item, rel, attr] = neural_network.data_preparation(file)

# Usefull veriables:
data_size = len(item)
#m = len(X)                  # Batch size
input_size = np.size(item, 1)  # Item number
relation_in_size = np.size(rel, 1)  # Relations number
output_size = np.size(attr, 1)  # Number of attributes
num_lay_1 = len(hidden_1)           # Number of layers in the first subnetwork
num_lay_2 = len(hidden_2)           # Number of layers in the second subnetwork
# Online condition
if online_learning == 'on':
    batches_count = int(data_size - round(data_proportion*data_size))
# Error history
J = range(batches_count * epochs_count + 1)
J_test = range(batches_count * epochs_count + 1)
# Weights history
theta_history = range(batches_count * epochs_count + 1)
for i in range(batches_count * epochs_count + 1):
    theta_history[i] = range(3)

# Data division (optional):
# Approximate data separation
num_test_ex = round(data_proportion*data_size)
num_train_ex = data_size - num_test_ex
# Batch size calculation
batch_size = int(round(num_train_ex / batches_count))
# Accurate data separation
num_train_ex = batch_size * batches_count
num_test_ex = data_size - num_train_ex
# Randomization of examples
idx = range(data_size)
rand_idx = np.random.permutation(idx)
training_ex_idx = rand_idx[:num_train_ex]
test_ex_idx = rand_idx[num_train_ex:]

# Test data set
test_item_set = item[test_ex_idx]
test_rel_set = rel[test_ex_idx]
test_attr_set = attr[test_ex_idx]


#  Create 3 sets of matrices of initial weights according to the given structure.
[theta_1, theta_2, theta_relation] = neural_network.initialise_weights(input_size, hidden_1, hidden_2, relation_in_size,
                                                                       output_size, num_lay_1, num_lay_2, epsilon)
# Save original theta matrices
theta_history[0] = [theta_1, theta_2, theta_relation]

#  Create initial moment for every weight
[moment_1, moment_2, moment_relation] = neural_network.initialize_moment(theta_1,theta_2,theta_relation)


for epoch in range(epochs_count):  # Beginning of epoch loop

    for batch in range(batches_count):  # Beginning of batch loop

        X = item[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
        input_relation = rel[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
        Y = attr[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
        m = len(X)

        # Compute activations of every unit in the network.
        [a_1, a_2] = neural_network.forward_propagation(m, num_lay_1, num_lay_2,
                                                        X, input_relation, theta_1, theta_2, theta_relation)

        # Compute average error with regularization (training)
        J[epoch * batches_count + batch] = neural_network.compute_cost_function(m, a_2, theta_1, theta_2, theta_relation,
                                                               num_lay_1, num_lay_2, R, Y)

        # Compute real error (test)
        [a_test_1, a_test_2] = neural_network.forward_propagation(num_test_ex, num_lay_1, num_lay_2, test_item_set, test_rel_set, theta_1, theta_2, theta_relation)

        J_test[epoch * batches_count + batch] = neural_network.compute_cost_function(num_test_ex, a_test_2, theta_1, theta_2, theta_relation,
                                                                   num_lay_1, num_lay_2, R, test_attr_set)


        # Compute derivative of the cost function with respect to matrices theta.
        [gradient_1, gradient_2, gradient_rel] = neural_network.back_propagation(m, a_1, a_2, input_relation,
                                                                                 theta_1, theta_2, theta_relation,
                                                                                 num_lay_1, num_lay_2, R, Y)

        # # Computes the numerical gradient of the function around theta for every weight (optional).
        # [numgrad_1, numgrad_2, numgrad_rel] = neural_network.gradient_check(e,
        #         m, X, Y, input_relation, theta_1, theta_2, theta_relation,
        #         num_lay_1, num_lay_2, R)

        # Change matrices of weights according to the gradient.
        [theta_1_temp, theta_2_temp, theta_relation_temp] = neural_network.descent(theta_1, theta_2, theta_relation,
                                                                                   gradient_1, gradient_2, gradient_rel,
                                                                                   num_lay_1, num_lay_2, alpha, moment_1,
                                                                                   moment_2, moment_relation, M)

        # Save weights values
        theta_history[epoch * batches_count + batch + 1] = [theta_1_temp, theta_2_temp, theta_relation_temp]

        # Update current weight matrices
        theta_1 = theta_1_temp
        theta_relation = theta_relation_temp
        theta_2 = theta_2_temp

# Compute final error after all loops of learning (Training)
[a_1, a_2] = neural_network.forward_propagation(m, num_lay_1, num_lay_2, X,
                                                input_relation, theta_1, theta_2, theta_relation)
J[-1] = neural_network.compute_cost_function(m, a_2, theta_1, theta_2,
                                             theta_relation, num_lay_1, num_lay_2, R, Y)

# Compute final real error (Test)
[a_1, a_2] = neural_network.forward_propagation(num_test_ex, num_lay_1, num_lay_2,
                                                test_item_set, test_rel_set, theta_1, theta_2, theta_relation)
J_test[-1] = neural_network.compute_cost_function(num_test_ex, a_2, theta_1, theta_2, theta_relation,
                                                num_lay_1, num_lay_2, R, test_attr_set)



# Save theta_log
# save J

# Visualisation
num_iter = range(len(J))
pp.figure(1)
pp.subplot(211)
pp.plot(num_iter, J)
pp.ylabel('Error')
pp.title('Training error')
pp.subplot(212)
pp.plot(num_iter, J_test)
pp.xlabel('Iteration')
pp.ylabel('Error')
pp.title('Test error')
pp.show()

"""
# Gradient verification
[numgrad_1, numgrad_2, numgrad_rel] = neural_network.gradient_check(e, m, X, Y,
        input_relation, theta_1, theta_2, theta_relation, num_lay_1, num_lay_2, R)

neural_network.verify_gradient(gradient_1, gradient_2, gradient_rel, numgrad_1,
        numgrad_2, numgrad_rel)"""