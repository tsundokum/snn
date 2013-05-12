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

import os
os.chdir('C:/SNN/temp')
import numpy as np
import neural_network
import csv

np.random.seed(111)

# Parameters:
hidden_1 = [8,10]  # Structure of the first subnetwork
hidden_2 = [10]  # Structure of the second subnetwork
epsilon = 0.2  # Limitation of  initial eights
alpha = 0.1  # Learning rate
R = 0.2  # Coefficient of regularization
M = 0.1  # Moment
e = 1e-4  # value of weights changing in the gradient check function
epochs_count = 100
batches_count = 1
file = 'resources/test-distance-matrix.csv'


# BEGINNING

# Create test data set
#X = np.arange(input_size * m).reshape(m, input_size)
#input_relation = np.arange(relation_in_size * m).reshape(m, relation_in_size)
#Y = np.ones((m * output_size))
#for l in range(m * output_size):
#    Y[l] = np.random.choice(2)
#Y = Y.reshape(m, output_size)

# Data set
[X, input_relation, Y] = neural_network.data_preparation(file)


# Usefull veriables:
m = len(X)                  # Batch size
input_size = np.size(X, 1)  # Item number
relation_in_size = np.size(input_relation, 1)  # Relations number
output_size = np.size(Y, 1)  # Number of attributes
num_lay_1 = len(hidden_1)           # Number of layers in the first subnetwork
num_lay_2 = len(hidden_2)           # Number of layers in the second subnetwork
J = np.zeros((epochs_count, batches_count))
# theta_history =
# Data division (optional):
# ...

# Separate validation set:
#     training_data =
#     validation_data = ...

# Divide training data by number of batches
#     X = ...
#     Y = ...


#  Create 3 sets of matrices of initial weights according to the given structure.
[theta_1, theta_2, theta_relation] = neural_network.initialise_weights(input_size, hidden_1, hidden_2, relation_in_size,
                                                                       output_size, num_lay_1, num_lay_2, epsilon)

#  Create initial moment for every weight
[moment_1, moment_2, moment_relation] = neural_network.initialize_moment(theta_1,theta_2,theta_relation)


for epoch in range(epochs_count):  # Beginning of epoch loop

    for batch in range(batches_count):  # Beginning of batch loop

        # Compute activations of every unit in the network.
        [a_1, a_2] = neural_network.forward_propagation(m, num_lay_1, num_lay_2,
                                                        X, input_relation, theta_1, theta_2, theta_relation)

        # Compute average error with regularization.
        J[epoch, batch] = neural_network.compute_cost_function(m, a_2, theta_1, theta_2, theta_relation,
                                                               num_lay_1, num_lay_2, R, Y)

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


        """# Save weights values
        theta_1_temp = theta_history[epoch, batch][0]
        theta_2_temp = theta_history[epoch, batch][1]
        theta_relation_temp = theta_history[epoch, batch][2]"""

        # Update current weight matrices
        theta_1 = theta_1_temp
        theta_relation = theta_relation_temp
        theta_2 = theta_2_temp

print "Errors by epochs = %s" % J.sum(1)
"""
# Gradient verification
[numgrad_1, numgrad_2, numgrad_rel] = neural_network.gradient_check(e, m, X, Y,
        input_relation, theta_1, theta_2, theta_relation, num_lay_1, num_lay_2, R)

neural_network.verify_gradient(gradient_1, gradient_2, gradient_rel, numgrad_1,
        numgrad_2, numgrad_rel)"""