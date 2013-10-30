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
import pickle

def Prepare_Learning(number_of_epochs, number_of_batches, data_proportion,
                     online_learning, data_representation, file_name):
    """
    Prepare data from given file according to learning parameters.
    Returns:
        ----
        item, rel, attr: training matrices for items, relations and attributes accordingly
        batch_size:  size of learning batch
        training_ex_idx: indexes of the training examples
        test_item_set, test_rel_set, test_attr_set: test matrices for items, relations and attributes accordingly

    """
    # Import data from file
    if data_representation == 'complex':
        [item, rel, attr] = neural_network.complex_data_preparation(file_name)
    elif data_representation == 'separate':
        [item, rel, attr_num, attr_val] = neural_network.separate_data_preparation2(file_name)
        attr = (attr_num, attr_val)
    elif data_representation == 'large':
        [item, rel, attr] = neural_network.big_data_preparation(file_name)

    data_size = len(item)
    # Online condition
    if online_learning == 'on':
        number_of_batches = int(data_size - round(data_proportion*data_size))

    # Data division (optional):
    # Approximate data separation
    num_test_ex = round(data_proportion*data_size)
    num_train_ex = data_size - num_test_ex
    # Batch size calculation
    batch_size = int(round(num_train_ex / number_of_batches))
    # Accurate data separation
    num_train_ex = batch_size * number_of_batches
    num_test_ex = data_size - num_train_ex
    # Randomization of examples
    idx = range(data_size)
    rand_idx = np.random.permutation(idx)
    training_ex_idx = rand_idx[:num_train_ex]
    test_ex_idx = rand_idx[num_train_ex:]

    # Test data set
    test_item_set = item[test_ex_idx]
    test_rel_set = rel[test_ex_idx]
    if data_representation == 'complex' or 'large':
        test_attr_set = attr[test_ex_idx]
    elif data_representation == 'separate':
        test_attr_set = range(2)
        test_attr_set[0] = attr_num[test_ex_idx]
        test_attr_set[1] = attr_val[test_ex_idx]

    return item, rel, attr, batch_size, number_of_batches, training_ex_idx, \
           test_item_set, test_rel_set, test_attr_set



# Learning
def Learning(alpha, R, S, M, hidden_1, hidden_2, epsilon, batch_size, item, rel,
             attr, data_representation, data_proportion, cost_function, number_of_epochs, number_of_batches,
             training_ex_idx, test_item_set, test_rel_set, test_attr_set):
    """
    Perform learning with given parameters.
    Returns:
        ----
        J: history of training errors (list)
        J_test: history of test errors (list)
        theta_history: weights matrices obtained during learning
        (list(iterations of learning) of lists(subnetwork)
        of lists(for multilayer subnetworks) of arrays)
    """
    # Usefull variables
    input_size = np.size(item, 1)  # Item number
    relation_in_size = np.size(rel, 1)  # Relations number
    output_size = 48                    # Number of attributes
    num_lay_1 = len(hidden_1)           # Number of layers in the first subnetwork
    num_lay_2 = len(hidden_2)           # Number of layers in the second subnetwork
    num_test_ex = len(test_item_set)    # Number of test examples
    # Error history
    J = range(number_of_batches * number_of_epochs + 1)
    J_test = range(number_of_batches * number_of_epochs + 1)
    # Weights history
    theta_history = range(number_of_batches * number_of_epochs + 1)
    for i in range(number_of_batches * number_of_epochs + 1):
        theta_history[i] = range(3)
    # unpack attr variable
    if data_representation == 'separate':
        attr_num = attr[0]
        attr_val = attr[1]

    #  Create 3 sets of matrices of initial weights according to the given structure.
    [theta_1, theta_2, theta_relation] = neural_network.initialise_weights(input_size, hidden_1, hidden_2, relation_in_size,
                                                                           output_size, num_lay_1, num_lay_2, epsilon)
    # Save original theta matrices
    theta_history[0] = [theta_1, theta_2, theta_relation]

    #  Create initial moment for every weight
    [moment_1, moment_2, moment_relation] = neural_network.initialize_moment(num_lay_1, theta_1, theta_2, theta_relation)

    for epoch in range(number_of_epochs):  # Beginning of epoch loop
        training_ex_idx = np.random.permutation(training_ex_idx)

        for batch in range(number_of_batches):  # Beginning of batch loop

            m = batch_size  # change "m" for batch_size
            X = item[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
            input_relation = rel[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
            if data_representation == 'complex' or 'large':
                Y = attr[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
            elif data_representation == 'separate':
                Y = range(2)
                Y[0] = attr_num[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
                Y[1] = attr_val[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]


            # Compute activations of every unit in the network.
            [a_1, a_2] = neural_network.forward_propagation(S, m, num_lay_1, num_lay_2,
                                                            X, input_relation, theta_1, theta_2, theta_relation)

            # Compute average error with regularization (training)
            J[epoch * number_of_batches + batch] = neural_network.compute_cost_function(cost_function, m, a_2, theta_1, theta_2, theta_relation,
                                                                   num_lay_1, num_lay_2, R, Y, data_representation)

            if data_proportion != 0:
                # Compute real error (test)
                [a_test_1, a_test_2] = neural_network.forward_propagation(S, num_test_ex, num_lay_1, num_lay_2, test_item_set, test_rel_set, theta_1, theta_2, theta_relation)

                J_test[epoch * number_of_batches + batch] = neural_network.compute_cost_function(cost_function, num_test_ex, a_test_2, theta_1, theta_2, theta_relation,
                                                                           num_lay_1, num_lay_2, R, test_attr_set, data_representation)

            # Compute derivative of the cost function with respect to matrices theta.
            [gradient_1, gradient_2, gradient_rel] = neural_network.back_propagation(S, m, a_1, a_2, input_relation,
                                                                                     theta_1, theta_2, theta_relation,
                                                                                     num_lay_1, num_lay_2, R, Y, data_representation)

            # Change matrices of weights according to the gradient.
            [theta_1_temp, theta_2_temp, theta_relation_temp] = neural_network.descent(theta_1, theta_2, theta_relation,
                                                                                       gradient_1, gradient_2, gradient_rel,
                                                                                       num_lay_1, num_lay_2, alpha, moment_1,
                                                                                       moment_2, moment_relation, M)
            # Save weights values
            theta_history[epoch * number_of_batches + batch + 1] = [theta_1_temp, theta_2_temp, theta_relation_temp]

            # Update current weight matrices
            theta_1 = theta_1_temp
            theta_relation = theta_relation_temp
            theta_2 = theta_2_temp

##        if data_representation == 'separate':
##            print 'epoch: '+str(epoch)+' / '+str(number_of_epochs)

    # Compute final error after all loops of learning (Training)
    [a_1, a_2] = neural_network.forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                    input_relation, theta_1, theta_2,theta_relation)
    J[-1] = neural_network.compute_cost_function(cost_function, m, a_2, theta_1, theta_2,
                                                 theta_relation, num_lay_1,
                                                 num_lay_2, R, Y, data_representation)
    if data_proportion != 0:
        # Compute final real error (Test)
        [a_1, a_2] = neural_network.forward_propagation(S, num_test_ex, num_lay_1, num_lay_2,
                                                        test_item_set, test_rel_set, theta_1, theta_2, theta_relation)
        J_test[-1] = neural_network.compute_cost_function(cost_function, num_test_ex, a_2, theta_1, theta_2, theta_relation,
                                                        num_lay_1, num_lay_2, R, test_attr_set, data_representation)

    return J, J_test, theta_history


# BEGINNING
def SNN(hidden_1, hidden_2, epsilon, alpha, S, R, M, number_of_epochs, number_of_batches,
        data_proportion, online_learning, data_representation, cost_function, file_name, gauge):
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
    if data_representation == 'complex':
        [item, rel, attr] = neural_network.complex_data_preparation(file_name)
    elif data_representation == 'separate':
        [item, rel, attr_num, attr_val] = neural_network.separate_data_preparation2(file_name)
    elif data_representation == 'large':
        [item, rel, attr] = neural_network.big_data_preparation(file_name)
    time = timer()
    time_ext['data_preparation'] = time - time_start_ext  # run-time of data_preparation()

    # Usefull veriables:
    data_size = len(item)
    #m = len(X)                  # Batch size
    input_size = np.size(item, 1)  # Item number
    relation_in_size = np.size(rel, 1)  # Relations number
    output_size = 48                    # Number of attributes
    num_lay_1 = len(hidden_1)           # Number of layers in the first subnetwork
    num_lay_2 = len(hidden_2)           # Number of layers in the second subnetwork
    # Online condition
    if online_learning == 'on':
        number_of_batches = int(data_size - round(data_proportion*data_size))
    # Error history
    J = range(number_of_batches * number_of_epochs + 1)
    J_test = range(number_of_batches * number_of_epochs + 1)
    # Weights history
    theta_history = range(number_of_batches * number_of_epochs + 1)
    for i in range(number_of_batches * number_of_epochs + 1):
        theta_history[i] = range(3)
    # Timers
    time_epoch = range(number_of_epochs)     # list to store time per epoch
    time_batch = np.zeros((number_of_epochs, number_of_batches))    # list to store time per batch
    # matrices to store time of performance for every function
    time_int = []
    for i in range(5):
        time_int.append(np.zeros((number_of_epochs, number_of_batches)))
    [time_forward_prop, time_cost, time_test, time_back_prop, time_descent] = time_int
    # fit the progress bar
    gauge.SetRange(number_of_epochs)

    # Data division (optional):
    # Approximate data separation
    num_test_ex = round(data_proportion*data_size)
    num_train_ex = data_size - num_test_ex
    # Batch size calculation
    batch_size = int(round(num_train_ex / number_of_batches))
    # Accurate data separation
    num_train_ex = batch_size * number_of_batches
    num_test_ex = data_size - num_train_ex
    # Randomization of examples
    idx = range(data_size)
    rand_idx = np.random.permutation(idx)
    training_ex_idx = rand_idx[:num_train_ex]
    test_ex_idx = rand_idx[num_train_ex:]

    # Test data set
    test_item_set = item[test_ex_idx]
    test_rel_set = rel[test_ex_idx]
    if data_representation == 'complex' or 'large':
        test_attr_set = attr[test_ex_idx]
    elif data_representation == 'separate':
        test_attr_set = range(2)
        test_attr_set[0] = attr_num[test_ex_idx]
        test_attr_set[1] = attr_val[test_ex_idx]
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

    for epoch in range(number_of_epochs):  # Beginning of epoch loop
        start_epoch = timer()
        training_ex_idx = np.random.permutation(training_ex_idx)

        for batch in range(number_of_batches):  # Beginning of batch loop
            start_batch = timer()

            m = batch_size  # change "m" for batch_size
            X = item[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
            input_relation = rel[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
            if data_representation == 'complex' or 'large':
                Y = attr[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
            elif data_representation == 'separate':
                Y = range(2)
                Y[0] = attr_num[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]
                Y[1] = attr_val[training_ex_idx[batch * batch_size : (batch+1) * batch_size]]


            # Compute activations of every unit in the network.
            [a_1, a_2] = neural_network.forward_propagation(S, m, num_lay_1, num_lay_2,
                                                            X, input_relation, theta_1, theta_2, theta_relation)
            time = timer()
            time_forward_prop[epoch, batch] = time - start_batch

            # Compute average error with regularization (training)
            J[epoch * number_of_batches + batch] = neural_network.compute_cost_function(cost_function, m, a_2, theta_1, theta_2, theta_relation,
                                                                   num_lay_1, num_lay_2, R, Y, data_representation)
            time_cost[epoch, batch] = timer() - time
            time = timer()    # timer update

            if data_proportion != 0:
                # Compute real error (test)
                [a_test_1, a_test_2] = neural_network.forward_propagation(S, num_test_ex, num_lay_1,
                                                                          num_lay_2, test_item_set,
                                                                          test_rel_set, theta_1, theta_2,
                                                                          theta_relation)

                J_test[epoch * number_of_batches + batch] = neural_network.compute_cost_function(cost_function, num_test_ex, a_test_2, theta_1,
                                                                                                 theta_2, theta_relation, num_lay_1,
                                                                                                 num_lay_2, R, test_attr_set,
                                                                                                 data_representation)
            time_test[epoch, batch] = timer() - time
            time = timer()    # timer update

            # Compute derivative of the cost function with respect to matrices theta.
            [gradient_1, gradient_2, gradient_rel] = neural_network.back_propagation(S, m, a_1, a_2, input_relation,
                                                                                     theta_1, theta_2, theta_relation,
                                                                                     num_lay_1, num_lay_2, R, Y, data_representation)
            time_back_prop[epoch, batch] = timer() - time
            time = timer()    # timer update

            # Change matrices of weights according to the gradient.
            [theta_1_temp, theta_2_temp, theta_relation_temp] = neural_network.descent(theta_1, theta_2, theta_relation,
                                                                                       gradient_1, gradient_2, gradient_rel,
                                                                                       num_lay_1, num_lay_2, alpha, moment_1,
                                                                                       moment_2, moment_relation, M)
            # Save weights values
            theta_history[epoch * number_of_batches + batch + 1] = [theta_1_temp, theta_2_temp, theta_relation_temp]

            # Update current weight matrices
            theta_1 = theta_1_temp
            theta_relation = theta_relation_temp
            theta_2 = theta_2_temp

            time_descent[epoch, batch] = timer() - time

            time_batch[epoch, batch] = timer() - start_batch    # batch timing

##        # show progress
##        if data_representation == 'separate':
##            print 'epoch: '+str(epoch)+' / '+str(number_of_epochs)
        gauge.SetValue(epoch) # show progress
        time_epoch[epoch] = timer() - start_epoch        # epoch timing

    # Compute final error after all loops of learning (Training)
    [a_1, a_2] = neural_network.forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                    input_relation, theta_1, theta_2,theta_relation)
    J[-1] = neural_network.compute_cost_function(cost_function, m, a_2, theta_1, theta_2,
                                                 theta_relation, num_lay_1,
                                                 num_lay_2, R, Y, data_representation)

    if data_proportion != 0:
        # Compute final real error (Test)
        [a_1, a_2] = neural_network.forward_propagation(S, num_test_ex, num_lay_1, num_lay_2,
                                                        test_item_set, test_rel_set, theta_1, theta_2, theta_relation)
        J_test[-1] = neural_network.compute_cost_function(cost_function, num_test_ex, a_2, theta_1, theta_2, theta_relation,
                                                        num_lay_1, num_lay_2, R, test_attr_set, data_representation)
    time_int.append(time_batch)
    time_int.append(time_epoch)

    return J, J_test, theta_history, time_ext, time_int


def save_cfg(cfg, name, txt):
    """ Save curent configuration given in cfg variable as dictionary.
        If txt == True, save txt-file, otherwise - pickle."""
    if txt == False:
        f = open(name+'_cfg.pkl', 'wb')
        pickle.dump(cfg, f)
        f.close()
    else:
        writer = csv.writer(open(name+'_cfg.txt', 'wb'))
        for key, value in cfg.items():
            writer.writerow([key, value])



# Gradient verification
##[numgrad_1, numgrad_2, numgrad_rel] = neural_network.gradient_check(e, m, X, Y,
##        input_relation, theta_1, theta_2, theta_relation, num_lay_1, num_lay_2, R)
##
##neural_network.verify_gradient(gradient_1, gradient_2, gradient_rel, numgrad_1,
##        numgrad_2, numgrad_rel)


# NET STRUCTURE ANALYSIS

def Rand_Inits(num_init, alpha, R, S, M, hidden_1, hidden_2, epsilon, batch_size,
               item, rel, attr, data_representation, data_proportion, cost_function, number_of_epochs,
               number_of_batches, training_ex_idx, test_item_set, test_rel_set, test_attr_set):
    """
    Perform internal loops of the Stucture_Analysis function.
    For every given net structure (hidden_1, hidden_2) perform learning several times (num_init)
    Every time weights initialise randomly within the bounds of epsilon
    Return: lists of errors over all initializations
        ----
        train_init: minimum train errors for
        train_init_of: last train errors for (consider overfitting)
        test_init: minimum test errors for
        test_init_of: last test errors for (consider overfitting)
    """
    # epmpty lists for random init-s errors
    train_init = []
    train_init_of = []
    test_init = []
    test_init_of = []
    for init in range(num_init):         # Loop over the random initializations
        [J, J_test, theta_history] = Learning(alpha, R, S, M, hidden_1, hidden_2,
                                                          epsilon, batch_size, item, rel, attr,
                                                          data_representation, data_proportion, cost_function,
                                                          number_of_epochs, number_of_batches,
                                                          training_ex_idx, test_item_set, test_rel_set,
                                                          test_attr_set)
        train_init.append(np.min(J))    # Collect munimum error values over the random initializations
        train_init_of.append(J[-1])      # Collect the last error values over the random initialization
        test_init.append(np.min(J_test))  # all the same for the test errors
        test_init_of.append(J_test[-1])

    return train_init, train_init_of, test_init, test_init_of


def cut_Structure_Analysis(hidden_1_max, hidden_2_max, num_init,
                           hidden_1, hidden_2, epsilon, alpha, S, R, M,
                           number_of_epochs, number_of_batches, data_proportion,
                           online_learning, data_representation, cost_function, file_name):
    """
    Optimised structure analysis function (internal loops in separate function)
    Returns: matrixes of errors hidden_1 x hidden_2 (array)
        ----
        SA_train: minimum train errors
        SA_train_of: last train errors (consider overfitting)
        SA_test: minimum test errors
        SA_test_of: last test errors (consider overfitting)
    """
    # Prepare date from given file
    [item, rel, attr, batch_size,
     number_of_batches, training_ex_idx,
     test_item_set, test_rel_set, test_attr_set] = Prepare_Learning(number_of_epochs,
                                                                    number_of_batches, data_proportion,
                                                                    online_learning, data_representation, file_name)
    # Prepare arrays to fill with error values
    SA_train = np.zeros((hidden_1_max, hidden_2_max))
    SA_train_of = np.zeros((hidden_1_max, hidden_2_max))
    SA_test = np.zeros((hidden_1_max, hidden_2_max))
    SA_test_of = np.zeros((hidden_1_max, hidden_2_max))

    for i in range(hidden_2_max):      # Loop over the hidden layer
        hidden_2 = [i+1]                 # Set number of neurons in the second layer(hidden)

        for j in range(hidden_1_max):  # Loop over  the representaton layer
            hidden_1 = [j+1]             # Set number of neurons in the first layer(representation)
            # Compute errors over several random initializations
            [train_init, train_init_of,
             test_init, test_init_of] = Rand_Inits(num_init, alpha, R, S, M, hidden_1, hidden_2,
                                                   epsilon, batch_size, item, rel, attr,
                                                   data_representation, data_proportion, cost_function,
                                                   number_of_epochs, number_of_batches,
                                                   training_ex_idx, test_item_set,
                                                   test_rel_set, test_attr_set)
            # take average error value
            SA_train[j, i] = np.average(train_init)
            SA_train_of[j, i] = np.average(train_init_of)
            SA_test[j, i] = np.average(test_init)
            SA_test_of[j, i] = np.average(test_init_of)
            # Show progress
            print 'hidden 1: '+str(hidden_1)+'/'+str(hidden_1_max)+'   '+    \
                  'hidden 2: '+str(hidden_2)+'/'+str(hidden_2_max)

    return SA_train, SA_train_of, SA_test, SA_test_of


def Structure_Analysis(hidden_1_max, hidden_2_max, num_init,
                       hidden_1, hidden_2, epsilon, alpha, S, R, M,
                       number_of_epochs, number_of_batches, data_proportion,
                       online_learning, data_representation, cost_function, file_name):
    """
    Computes efficiency of network with respect to number of neurons in every layer
    (Only for one-layer subnetworks).
    Takes maximum number of neurons in every layer and number of random weight initializations.
    For every net structure learn network several times(num_init) with different random initial weights.
    Every time it takes only one value of test error (minimum or the last).
    And then computes average error for  particular net structure over all initializations.
    Returns: matrixes of errors hidden_1 x hidden_2 (array)
        ----
        SA_train: minimum train errors
        SA_train_of: last train errors (consider overfitting)
        SA_test: minimum test errors
        SA_test_of: last test errors (consider overfitting)

    """
    # Prepare date from given file
    [item, rel, attr, batch_size,
     number_of_batches, training_ex_idx,
     test_item_set, test_rel_set, test_attr_set] = Prepare_Learning(number_of_epochs,
                                                                    number_of_batches, data_proportion,
                                                                    online_learning, data_representation, file_name)
    # Prepare arrays to fill with error values
    SA_train = np.zeros((hidden_1_max, hidden_2_max))
    SA_train_of = np.zeros((hidden_1_max, hidden_2_max))
    SA_test = np.zeros((hidden_1_max, hidden_2_max))
    SA_test_of = np.zeros((hidden_1_max, hidden_2_max))

    for i in range(hidden_2_max):      # Loop over the hidden layer
        hidden_2 = [i+1]                 # Set number of neurons in the second layer(hidden)

        for j in range(hidden_1_max):  # Loop over  the representaton layer
            start = timer()
            hidden_1 = [j+1]             # Set number of neurons in the first layer(representation)
            # epmpty lists for random init-s errors
            train_init = []
            train_init_of = []
            test_init = []
            test_init_of = []

            for init in range(num_init):         # Loop over the random initializations
                [J, J_test, theta_history] = Learning(alpha, R, S, M, hidden_1, hidden_2,
                                                                  epsilon, batch_size, item, rel, attr,
                                                                  data_representation, data_proportion, cost_function,
                                                                  number_of_epochs, number_of_batches,
                                                                  training_ex_idx, test_item_set,
                                                                  test_rel_set, test_attr_set)
                train_init.append(np.min(J))    # Collect munimum error values over the random initializations
                train_init_of.append(J[-1])      # Collect the last error values over the random initialization
                test_init.append(np.min(J_test))  # all the same for the test errors
                test_init_of.append(J_test[-1])

            # take average error value
            SA_train[j, i] = np.average(train_init)
            SA_train_of[j, i] = np.average(train_init_of)
            SA_test[j, i] = np.average(test_init)
            SA_test_of[j, i] = np.average(test_init_of)
            time_per_loop = timer() - start
            # Show progress
            print 'hidden 1: '+str(hidden_1)+'/'+str(hidden_1_max)+'   '+    \
                  'hidden 2: '+str(hidden_2)+'/'+str(hidden_2_max)
            print 'time per loop:'+str(time_per_loop)

    return SA_train, SA_train_of, SA_test, SA_test_of


def save_SA_results(SA_train, SA_train_of, SA_test, SA_test_of, cfg, file_name, csv_opt):
    """
    Save resultas of structure analysis in two files.
    file_name_cfg.txt contains learning parameters
    file_name.csv contains matrices of errors
    """
    writer = csv.writer(open('SA_results/'+file_name+'_cfg.txt', 'wb'))
    for key, value in cfg.items():
        writer.writerow([key, value])
    if csv_opt == True:
        np.savetxt('SA_results/'+file_name+'_train.csv', SA_train, delimiter=',')
        np.savetxt('SA_results/'+file_name+'_train_of.csv', SA_train_of, delimiter=',')
        np.savetxt('SA_results/'+file_name+'_test.csv', SA_test, delimiter=',')
        np.savetxt('SA_results/'+file_name+'_test_of.csv', SA_test_of, delimiter=',')
    else:
        J_SA = [SA_train, SA_train_of, SA_test, SA_test_of]
        f = open('SA_results/'+file_name+'.pkl', 'wb')
        pickle.dump(J_SA, f)
        f.close()


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
    pp.figure(1)
    num_iter = range(len(J))
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


def disp_struct_analysis(J_SA):
    """
    ...
    """
    hidden_1_max = np.size(J_SA, 0)
    hidden_2_max = np.size(J_SA, 1)
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


def hidden_activation(item_num, rel_num, num_lay_1, num_lay_2, theta_history, iter, S):
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
            [a_1, a_2] = neural_network.forward_propagation(S, 1, num_lay_1, num_lay_2, X,
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



def SA(alpha, R, S, M, epsilon, batch_size, item, rel, attr, data_representation,
       data_proportion, cost_function, number_of_epochs, number_of_batches, training_ex_idx,
       test_item_set, test_rel_set, test_attr_set, hidden_1_range, hidden_2_range, num_init):

    # Prepare arrays to fill with error values
    hidden_1_max = (hidden_1_range[1] + 1) - hidden_1_range[0]  # maximum range of neurons in every subnet
    hidden_2_max = (hidden_2_range[1] + 1) - hidden_2_range[0]
    SA = np.zeros((hidden_1_max, hidden_2_max))
    average_theta = range(hidden_1_max)
    for i in xrange(len(average_theta)):
        average_theta[i] = range(hidden_2_max)
        for j in xrange(len(average_theta[i])):
            average_theta[i][j] =[0] * 3
            average_theta[i][j][1] = [0,0]
    performance_time = 0

    for i in range(hidden_2_range[0], hidden_2_range[1] + 1):      # Loop over the hidden layer
        hidden_2 = [i]                 # Set number of neurons in the second layer(hidden)

        for j in range(hidden_1_range[0], hidden_1_range[1] + 1):  # Loop over  the representaton layer
            start = timer()
            hidden_1 = [j]             # Set number of neurons in the first layer(representation)
            # epmpty lists for random init-s errors
            init_error = []
            theta_accumulator = [0] * 3
            theta_accumulator[1] = [0,0]
            Iterations = []
            # Loop over the random initializations
            for init in range(num_init):
                [J, J_test, theta_history] = Learning(alpha, R, S, M, hidden_1, hidden_2,
                                                      epsilon, batch_size, item, rel, attr,
                                                      data_representation, data_proportion, cost_function,
                                                      number_of_epochs, number_of_batches,
                                                      training_ex_idx, test_item_set, test_rel_set,
                                                      test_attr_set)
                if data_proportion != 0:
                    best_iter = np.argmin(J_test)
                    init_error.append(np.min(J_test))
                    Iterations.append(best_iter)
                    theta_accumulator[0] += theta_history[best_iter][0]
                    theta_accumulator[1][0] += theta_history[best_iter][1][0]
                    theta_accumulator[1][1] += theta_history[best_iter][1][1]
                    theta_accumulator[2] += theta_history[best_iter][2]
                else:
                    init_error.append(J[-1])
                    theta_accumulator[0] += theta_history[-1][0]
                    theta_accumulator[1][0] += theta_history[-1][1][0]
                    theta_accumulator[1][1] += theta_history[-1][1][1]
                    theta_accumulator[2] += theta_history[-1][2]

            r_idx = hidden_1[0] - hidden_1_range[0]
            c_idx = hidden_2[0] - hidden_2_range[0]
            # average error value
            SA[r_idx, c_idx] = np.average(init_error)  # fill the surface of errors
            # average theta matrices
            average_theta[r_idx][c_idx][0] = theta_accumulator[0] / num_init
            average_theta[r_idx][c_idx][1][0] = theta_accumulator[1][0] / num_init
            average_theta[r_idx][c_idx][1][1] = theta_accumulator[1][1] / num_init
            average_theta[r_idx][c_idx][2] = theta_accumulator[2] / num_init
            time_per_loop = timer() - start
            # average number of the best iteration
            if data_proportion != 0:
                average_iter = round(np.average(Iterations))
            else:
                average_iter = number_of_epochs * number_of_batches

            # Time
            full_time = time_per_loop * (hidden_1_max * hidden_2_max)
            performance_time += time_per_loop
            remaining_time = full_time - performance_time
            hours = int(remaining_time / 3600)
            minutes = int((remaining_time - (hours * 3600)) / 60)
            seconds = remaining_time - (hours*3600) - (minutes*60)
            print 'approximate ramaining time :'+str(hours)+' hours  '+ \
                   str(minutes)+' minutes  '+str(seconds)+' seconds'
            # Show progress
            print 'hidden 1: '+str(hidden_1[0])+'/'+str(hidden_1_range[1])+'   '+  \
                  'hidden 2: '+str(hidden_2[0])+'/'+str(hidden_2_range[1])

    return SA, average_theta, average_iter


def save_SA(file_name, test, best_iter, SA, average_theta):
    # error surface
    f = open(file_name+'_SA_surf'+'_('+str(best_iter)+')'+'_'+test+'.pkl', 'wb')
    pickle.dump(SA, f)
    f.close()
    # theta matrices
    f = open(file_name+'_theta'+'_('+str(best_iter)+')'+'_'+test+'.pkl', 'wb')
    pickle.dump(average_theta, f)
    f.close()




