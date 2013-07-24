#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        Neural Network Learning
# Purpose:     Contains complex functions to analyse learning of Semantic
#              neural network.
#
# Author:      Ilya Pershin
#
# Created:     14.03.2013
# Copyright:   (c) 11 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
"""
Contains complex functions to analyse learning of Semantic neural network
...
"""
import csv
import numpy as np
import matplotlib.pyplot as pp
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

import neural_network


# BEGINNING
def SNN(hidden_1, hidden_2, epsilon, alpha, S, R, M, e, epochs_count, batches_count,
        data_proportion, online_learning, file):
    """
    Main learning function.
    Takes given network structure and learning parameters.
    Returns:
        ----
        J: history of training errors (list)
        J_test: history of test errors (list)
        theta_history: weights matrices obtained during learning
        (list(iterations of learning) of lists(subnetwork)
        of lists(for multilayer subnetworks) of arrays)
        time_ext: run time of external parts of code (dictionary)
        time_int: run time of internal parts of code (list of arrays)
    """
    time_ext = dict()           # Create dictionary to store time values from external block
    time_start_ext = timer()    # Start point for timer in external block
    # Import data from file
    [item, rel, attr] = neural_network.complex_data_preparation(file)
    time = timer()
    time_ext['data_preparation'] = time - time_start_ext

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
    # Timers
    time_epoch = range(epochs_count)     # list to store time per epoch
    time_batch = np.zeros((epochs_count, batches_count))    # list to store time per batch
    # matrices to store time of performance for every function
    time_int = []
    for i in range(5):
        time_int.append(np.zeros((epochs_count, batches_count)))
    [time_forward_prop, time_cost, time_test, time_back_prop, time_descent] = time_int

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
    time_ext['variables, data_division'] = timer() - time
    time = timer()    # update timer

    #  Create 3 sets of matrices of initial weights according to the given structure.
    [theta_1, theta_2, theta_relation] = neural_network.initialise_weights(input_size, hidden_1, hidden_2, relation_in_size,
                                                                           output_size, num_lay_1, num_lay_2, epsilon)
    # Save original theta matrices
    theta_history[0] = [theta_1, theta_2, theta_relation]

    #  Create initial moment for every weight
    [moment_1, moment_2, moment_relation] = neural_network.initialize_moment(num_lay_1, theta_1, theta_2, theta_relation)
    time_ext['weights, moment_init'] = timer() - time

    for epoch in range(epochs_count):  # Beginning of epoch loop
        start_epoch = timer()

        for batch in range(batches_count):  # Beginning of batch loop
            start_batch = timer()

            X = item[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
            input_relation = rel[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
            Y = attr[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
            m = len(X)

            # Compute activations of every unit in the network.
            [a_1, a_2] = neural_network.forward_propagation(S, m, num_lay_1, num_lay_2,
                                                            X, input_relation, theta_1, theta_2, theta_relation)
            time = timer()
            time_forward_prop[epoch, batch] = time - start_batch

            # Compute average error with regularization (training)
            J[epoch * batches_count + batch] = neural_network.compute_cost_function(m, a_2, theta_1, theta_2, theta_relation,
                                                                   num_lay_1, num_lay_2, R, Y)
            time_cost[epoch, batch] = timer() - time
            time = timer()    # timer update


            # Compute real error (test)
            [a_test_1, a_test_2] = neural_network.forward_propagation(S, num_test_ex, num_lay_1, num_lay_2, test_item_set, test_rel_set, theta_1, theta_2, theta_relation)

            J_test[epoch * batches_count + batch] = neural_network.compute_cost_function(num_test_ex, a_test_2, theta_1, theta_2, theta_relation,
                                                                       num_lay_1, num_lay_2, R, test_attr_set)
            time_test[epoch, batch] = timer() - time
            time = timer()    # timer update

            # Compute derivative of the cost function with respect to matrices theta.
            [gradient_1, gradient_2, gradient_rel] = neural_network.back_propagation(S, m, a_1, a_2, input_relation,
                                                                                     theta_1, theta_2, theta_relation,
                                                                                     num_lay_1, num_lay_2, R, Y)
            time_back_prop[epoch, batch] = timer() - time
            time = timer()    # timer update

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

            time_descent[epoch, batch] = timer() - time

            time_batch[epoch, batch] = timer() - start_batch    # batch timing

        time_epoch[epoch] = timer() - start_epoch        # epoch timing

    # Compute final error after all loops of learning (Training)
    [a_1, a_2] = neural_network.forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                    input_relation, theta_1, theta_2, theta_relation)
    J[-1] = neural_network.compute_cost_function(m, a_2, theta_1, theta_2,
                                                 theta_relation, num_lay_1, num_lay_2, R, Y)

    # Compute final real error (Test)
    [a_1, a_2] = neural_network.forward_propagation(S, num_test_ex, num_lay_1, num_lay_2,
                                                    test_item_set, test_rel_set, theta_1, theta_2, theta_relation)
    J_test[-1] = neural_network.compute_cost_function(num_test_ex, a_2, theta_1, theta_2, theta_relation,
                                                    num_lay_1, num_lay_2, R, test_attr_set)
    time_int.append(time_batch)
    time_int.append(time_epoch)

    return J, J_test, theta_history, time_ext, time_int


# Gradient verification
##[numgrad_1, numgrad_2, numgrad_rel] = neural_network.gradient_check(e, m, X, Y,
##        input_relation, theta_1, theta_2, theta_relation, num_lay_1, num_lay_2, R)
##
##neural_network.verify_gradient(gradient_1, gradient_2, gradient_rel, numgrad_1,
##        numgrad_2, numgrad_rel)


# NET STRUCTURE ANALYSIS

def Structure_Analysis(hidden_1_max, hidden_2_max, num_init, overfitting,
                       hidden_1, hidden_2, epsilon, alpha, S, R, M, e, epochs_count,
                       batches_count, data_proportion, online_learning, file):
    """
    Computes efficiency of network with respect to number of neurons in every layer
    (Only for one-layer subnetworks).
    Takes maximum number of neurons in every layer and number of random weight initializations.
    If overfitting variable == 'on', function takes only the the last error(overfitted).
    Otherwise, it takes minimum error values(before overfitting).
    For every net structure learn network several times(num_init) with different random initial weights.
    Every time it takes only one value of test error (minimum or the last).
    And then computes average error for  particular net structure over all initializations.
    Returns:
        J_SA: matrix of errors hidden_1 x hidden_2 (array)

    """
    J_SA = np.zeros((hidden_1_max, hidden_2_max))

    for i in range(hidden_2_max):      # Loop over the hidden layer
        hidden_2 = [i+1]                 # Set number of neurons in the second layer(hidden)

        for j in range(hidden_1_max):  # Loop over  the representaton layer
            hidden_1 = [j+1]             # Set number of neurons in the first layer(representation)
            J_init = []

            for init in range(num_init):         # Loop over the random initializations
                [J, J_test, theta_history, time_ext, time_int] = SNN(hidden_1, hidden_2, epsilon, alpha,
                                             S, R, M, e, epochs_count, batches_count,
                                             data_proportion, online_learning, file)
                del J, theta_history, time_ext, time_int    # Delete large unnecessary variables
                if overfitting == 'on':          # Consider the overfitting effect
                    J_init.append(J_test[-1])    # Collect the last error values over the random initializations
                else:                            # Ignore overfitting
                    J_init.append(np.min(J_test))    # Collect munimum error values over the random initializations

            J_SA[j, i] = np.average(J_init)    # take average error value
            # Show progress
            print 'hidden 1: '+str(hidden_1)+'/'+str(hidden_1_max)+'   '+    \
                  'hidden 2: '+str(hidden_2)+'/'+str(hidden_2_max)

    return J_SA


def Load_SA_results(SA_file):
    """
    load result of structure analysis from csv(",") file.
    File name represents:
    StructAn[maximum_number_of_neurons_in_1st_layer,maximum_number_of_neurons_in_2nd_layer,
    number of random initializations]_name_of_testee.csv
    Returns:
        ----
        table: matrix of errors with respect to number of neurons in both layers
        [number of neurons in 1st layer] X [number of neurons in 2nd layer] (array)
    """
    infile = open(SA_file, 'r')
    table = []
    for row in csv.reader(infile):
        table.append(row)
    infile.close()
    table = np.array(table)        # Transform table into NumPy array
    table = np.array(table, dtype='float') # Change data type
    return table


# VISUALIZATION
def disp_learning_dynamic(J, J_test):
    """
    ...
    """
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

def disp_struct_analysis(J_SA, hidden_1_max, hidden_2_max):
    """
    ...
    """
    x = np.zeros((hidden_1_max, hidden_2_max))
    for i in range(hidden_2_max):
        x[:,i] = range(hidden_1_max)

    y = np.zeros((hidden_1_max, hidden_2_max))
    for i in range(hidden_1_max):
        y[i,:] = range(hidden_2_max)

    l = hidden_1_max * hidden_2_max
    x_flat = range(l)
    y_flat = range(l)
    z_flat = range(l)
    for i in range(l):
        x_flat[i] = x.flat[i]
        y_flat[i] = y.flat[i]
        z_flat[i] = J_SA.flat[i]
    x_flat = np.array(x_flat)
    y_flat = np.array(y_flat)
    z_flat = np.array(z_flat)

    fig = pp.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x_flat + 1, y_flat + 1, z_flat, cmap=cm.jet, linewidth=0.2)
    pp.xlabel('Representation layer')
    pp.ylabel('Hidden layer')
    pp.show()


def hidden_activation(item_num, rel_num, num_lay_1, num_lay_2, theta_history, iter):
    """ Computes activations of every unit for one particular iteration"""
    theta_1 = theta_history[iter][0]
    theta_2 = theta_history[iter][1]
    theta_relation = theta_history[iter][2]
    # Create list of lists to store activation matrices
    activation_1 = range(item_num)
    for i in xrange(item_num):
        activation_1[i] = range(rel_num)
    activation_2 = range(item_num)
    for i in xrange(item_num):
        activation_2[i] = range(rel_num)
    # Choose item input
    for it in range(item_num):
        X = np.zeros((1,4))
        X.flat[it] = 1
        # Choose relation input
        for rel in range(rel_num):
            input_relation = np.zeros((1,4))
            input_relation.flat[rel] = 1
            # perform forward propagation
            [a_1, a_2] = neural_network.forward_propagation(1, num_lay_1, num_lay_2, X,
                    input_relation, theta_1, theta_2, theta_relation)
            activation_1[it][rel] = a_1[1]
            activation_2[it][rel] = a_2

    return activation_1, activation_2


def PCA(X,k):
    """
    ...
    """
    m = len(X)
    Sigma = np.dot(X.T, X) / m
    [U, S, V] = np.linalg.svd(Sigma)
    Ureduce = U[:, :k]
    z = np.dot(X, Ureduce)
    return z







