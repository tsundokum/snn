#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        neural_network
# Purpose:     Contains functions to create and learn McClelland's neural network.

#
# Author:      Ilya Pershin
#
# Created:     08.03.2013
# Copyright:   (c) 11 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
"""
Contains functions to create and learn McClelland's neural network.

Parameters:
    ----
    input_size: item number (integer)
    hidden_1: structure of the first subnetwork (list of integers)
    hidden_2: structure of the second subnetwork (list of integers)
    relation_in_size: number of input relations (integer)
    output_size: number of attributes (integer)
    epsilon: limitation of  initial eights (float)
    R: coefficient of regularization (float)
    alpha: learning rate (float)
    e: value of weights changing in the gradient check function(float)

Data:
    ----
    X: input training set for items (numpy.ndarray; rows: training examples;
            columns: dimensions)
    input_relation: input training set for relations (numpy.ndarray;
            rows: training examples; columns: dimensions)
    Y: output training set for attributes (numpy.ndarray; rows:
            training examples; columns: dimensions)

Useful variables:
    ----
    m: number of examples in the training set
    num_lay_1: number of layers in the first subnetwork
    num_lay_2: number of layers in the second subnetwork
"""

import numpy as np
import csv

def sigmoid(z, S):
    """Compute sigmoid function for given number z."""
    return 1 / (1.0 + np.exp(-S*z))


def sigmoid_gradient(z, S):
    """Compute partial derivative of sigmoid function with respect to value z"""
    q = sigmoid(z, S)
    return q * (1 - q)


# Function to prepare full(complex) training examples
def complex_data_preparation(file):
    """
    Prepare learning data from given csv-file (comma separated).
    ((Represent data in 16 examples))
    Takes file name written as string.
    Returns matrixes in format: number of examples by number of dimensions.
    Every row in input matrix represents one learning example in which the
    corresponding number equals one and others are zero.
    (Matrix for attributes contains all information for the current input pair!)
    Returns:
        ----
        item: learning matrix for items (array)
        rel: learning matrix for ralations (array)
        attr: learning matrix for attributes (array)

    """
    # Extract data from file as list of strings
    infile = open(file, 'r')
    table = []
    for row in csv.reader(infile):
        table.append(row)
    infile.close()

    # Create data matrix
    data = range(768)
    # Fill the fourth column with connection value
    for i in xrange(4):
        for j in xrange(192):
            data[j + 192*i] = table[j][i+2]
    del table

    #define number of dimentions for every learning matrix
    input_size = 4
    relation_in_size = 4
    output_size = 48
    m = input_size * relation_in_size # number of original examples

    # Create item learning matrix
    item = np.zeros((m, input_size))
    for i in xrange(input_size):
        item[i*input_size:(i+1)*input_size, i] = 1
    # Create relation learning matrix
    rel = np.zeros((m, relation_in_size))
    for j in xrange(input_size):
        for i in xrange(relation_in_size):
            rel[j*input_size+i, i] = 1
    # Create attribute learning matrix
    attr = np.zeros((m, output_size))
    for i in xrange(m):
        attr[i] = data[i*output_size:(i+1)*output_size]

    return item, rel, attr


# Function to prepare partial(separate) training examples
def separate_data_preparation(file):
    """
    Prepare learning data from given csv-file (comma separated).
    ((Represent data in 768 examples))
    Takes file name written as string.
    Returns matrixes in format: number of examples by number of dimensions.
    Every row in the matrix represents one learning example in which the
    corresponding number equals one and others are zero.
    Returns:
        ----
        item: learning matrix for items (array)
        rel: learning matrix for ralations (array)
        attr_num: numbers of activated attribute neurons (array)
        attr_val: activatation value of activated attribute neurons (array)

    """
    # Extract data from file as list of strings
    infile = open(file, 'r')
    table = []
    for row in csv.reader(infile):
        table.append(row)
    infile.close()

    # Create data matrix
    data = np.zeros((768,4))
    # Fill the first column with numbers of items
    for i in xrange(4):
        data[i*192:(i+1)*192, 0] = i + 1
    # Fill the second column with numbers of relations
        tab = data[i*192:(i+1)*192]
        for j in xrange(4):
            tab[j*48:(j+1)*48, 1] = j + 1
        data[i*192:(i+1)*192] =  tab
    # Fill the third column with numbers of attributes
    for i in xrange(16):
        tab = data[i*48:(i+1)*48]
        for j in range(48):
            tab[j, 2] = j + 1
        data[i*48:(i+1)*48] = tab
    # Fill the fourth column with connection value
    for i in xrange(4):
        for j in xrange(192):
            data[j + 192*i,3] = table[j][i+2]
    del table

    #define number of dimentions for every learning matrix
    input_size = int(np.max(data[:,0]))
    relation_in_size = int(np.max(data[:,1]))
    output_size = int(np.max(data[:,2]))

    # Create learning matrices.
    item = np.zeros((len(data), input_size))
    for i in range(len(data)):
        item[i, data[i, 0]-1] = 1
    rel = np.zeros((len(data), relation_in_size))
    for i in range(len(data)):
        rel[i, data[i, 1]-1] = 1
    attr_num = np.zeros((len(data),1), dtype = 'int8')
    attr_val = np.zeros((len(data),1))
    for i in xrange(len(data)):
        attr_num[i] = data[i, 2] - 1
        attr_val[i] = data[i, 3]
    return item, rel, attr_num, attr_val


def initialize_moment(num_lay_1, theta_1, theta_2, theta_relation):
    """ Initialize start values of moment for every weight"""
    # First
    moment_1 = range(num_lay_1)
    if num_lay_1 == 1:   # One layer condition
        moment_1 = np.zeros((np.shape(theta_1)))
    else:
        for i in xrange(num_lay_1):
            moment_1[i] = np.zeros((np.shape(theta_1[i])))
    # Second
    moment_2 = range(len(theta_2))
    for i in xrange(len(theta_2)):
        moment_2[i] = np.zeros((np.shape(theta_2[i])))
    # Relation
    moment_relation = np.zeros((np.shape(theta_relation)))
    return moment_1, moment_2, moment_relation


def generate_rand_weights_for_subnet(w_struct, epsilon):
    num_lay = len(w_struct) - 1
    theta = range(num_lay)  # create list ready to fill with matrices of weights
    for i in xrange(num_lay):  # loop over the layers of the first subnetwork
        # matrix with random values from (-1 * epsilon) to (1 * epsilon)
        theta[i] = (np.random.rand(w_struct[i] + 1, w_struct[i + 1]) * 2 - 1) * epsilon
    if num_lay == 1:        # Special condition for one layer subnetwork
        theta = theta[0]
    return theta


def initialise_weights(input_size, hidden_1, hidden_2, relation_in_size,
                       output_size, num_lay_1, num_lay_2, epsilon):
    """
    Generate 3 sets of matrices of initial weights according to the given structure.
    Returns:
        ----
        theta_1: matrices of weights for the first subnetwork (list of arrays)
        theta_2: matrices of weights for the second subnetwork (list of arrays)
        theta_relation: theta_1: matrix of weights for the relations(array)
    """

    # First subnetwork
    theta_1 = generate_rand_weights_for_subnet(np.hstack((input_size, hidden_1)), epsilon)
    # Second subnetwork
    theta_2 = generate_rand_weights_for_subnet(np.hstack((hidden_1[-1], hidden_2, output_size)), epsilon)

    # Intermediate subnetwork (relation)
    # matrix with random values from (-1 * epsilon) to (1 * epsilon)
    theta_relation = (np.random.rand(relation_in_size + 1, hidden_2[0]) * 2 - 1) * epsilon
    return theta_1, theta_2, theta_relation


def forward_propagation(S, m, num_lay_1, num_lay_2, X, input_relation, theta_1,
                        theta_2, theta_relation):
    """
    Compute activations of every unit in the network.
    Returns:
        ----
        a_1: astivations of neurons in the first subnetwork (list of arrays)
        a_2: astivations of neurons in the second subnetwork (list of arrays)
    """
    # First subnetwork:
    z_1 = range(num_lay_1 + 1)
    a_1 = range(num_lay_1 + 1)
    z_1[0] = X  # input data
    a_1[0] = np.hstack((np.ones((m, 1)), z_1[0]))  # add bias units
    if num_lay_1 == 1:    # Special condition for one layer subnetwork
        z_1[1] = z_1[1] = np.dot(a_1[0], theta_1)
        a_1[1] = np.hstack((np.ones((m, 1)), sigmoid(z_1[1], S)))
    else:
        for i in xrange(1, num_lay_1 + 1):  # loop over the first subnetwork
            z_1[i] = np.dot(a_1[i - 1], theta_1[i - 1])  # perform matrix multiplication to compute sum for every unit
            a_1[i] = np.hstack((np.ones((m, 1)), sigmoid(z_1[i], S)))  # compute sigmoid function and add bias units

    # Second subnetwork:
    z_2 = range(num_lay_2 + 1)
    a_2 = range(num_lay_2 + 1)
    rel_input_b = np.hstack((np.ones((m, 1)), input_relation))          # add bias term to the relation input,
    # a_1[-1] already have bias
    z_2[0] = np.dot(a_1[-1], theta_2[0]) + np.dot(rel_input_b, theta_relation)  # first layer in the second subnetwork
    a_2[0] = np.hstack((np.ones((m, 1)), sigmoid(z_2[0], S)))
    for i in xrange(1, num_lay_2 + 1):                   # loop over the other layers of the second subnetwork
        z_2[i] = np.dot(a_2[i - 1], theta_2[i])
        a_2[i] = np.hstack((np.ones((m, 1)), sigmoid(z_2[i], S)))
    a_2[-1] = a_2[-1][:, 1:]                             # remove bias unit from the last(output) layer
    return a_1, a_2


def compute_cost_function(m, a_2, theta_1, theta_2, theta_relation,
                          num_lay_1, num_lay_2, R, Y, data_representation):
    """
    Compute average error with regularization.
    Returns:
        ----
        J: approximate error (float)
    """
    # Average cost
    cost = 0
    if data_representation == 'complex':
        cost = np.sum(-Y * np.log(a_2[-1]) - (1 - Y) * np.log(1 - a_2[-1])) / m
    elif data_representation == 'separate':
        for i in xrange(m):  # loop over the examples in the batch
            cost_ex = -Y[1][i] * np.log(a_2[-1][i][Y[0][i]]) - (1 - Y[1][i]) * \
                      np.log(1 - a_2[-1][i][Y[0][i]])
            cost = cost + cost_ex  # cost accumulation
        cost = cost / m
    # Regularization
    if num_lay_1 == 1:
        reg_1 = np.sum(theta_1[:, 1:] ** 2)
    else:
        for i in xrange(num_lay_1):
            reg_1 = np.sum(theta_1[i][:, 1:] ** 2)
    reg_relation = np.sum(theta_relation[:, 1:] ** 2)
    for i in xrange(num_lay_2 + 1):
        reg_2 = np.sum(theta_2[i][:, 1:] ** 2)
    regularization = (reg_1 + reg_2 + reg_relation) * (R / (2 * m))
    J = cost + regularization
    return J


def back_propagation(S, m, a_1, a_2, input_relation, theta_1, theta_2, theta_relation,
                     num_lay_1, num_lay_2, R, Y, data_representation):
    """
    Compute derivative of the cost function with respect to matrices theta.
    Returns:
        ----
        grad_reg_1: partial derivatives of weights of the first subnetwork with
                respect to error of the relevant neurons (list of arrays)
        grad_reg_2: partial derivatives of weights of the second subnetwork with
                respect to error of the relevant neurons (list of arrays)
        rel_grad_reg: partial derivatives of the relation weights with respect
                to error of the relevant neurons (array)
    """
    # Errors of neurons in th second subnetwork
    rel_input_b = np.hstack((np.ones((m, 1)), input_relation)) # add bias to the relation input
    d_2 = range(num_lay_2 + 1)        # storage for delta values in the second subnet
    # compute error for the output layer
    if data_representation =='complex':
        d_2[-1] = a_2[-1] - Y
    elif data_representation == 'separate':
        d_2[-1] = np.zeros((np.shape(a_2[-1])))
        for i in xrange(m):           # loop over the examples in the batch
            d_2[-1][i][Y[0][i]] = a_2[-1][i][Y[0][i]] - Y[1][i]
    for i in xrange(2, num_lay_2 + 1):
        d_2[-i] = np.dot(d_2[-i + 1], theta_2[-i + 1][1:, :].T) * \
                  sigmoid_gradient(np.dot(a_2[-i - 1], theta_2[-i]), S)
    d_2[0] = np.dot(d_2[1], theta_2[1][1:, :].T) * \
             sigmoid_gradient(np.dot(a_1[-1], theta_2[0]), S) * \
             sigmoid_gradient(np.dot(rel_input_b, theta_relation), S)   # changed + for *
    # Errors of neurons in th first subnetwork
    d_1 = range(num_lay_1)
    if num_lay_1 == 1:
        d_1 = np.dot(d_2[0], theta_2[0][1:, :].T) * \
                sigmoid_gradient(np.dot(a_1[-2], theta_1), S)
    else:
        d_1[-1] = np.dot(d_2[0], theta_2[0][1:, :].T) * \
                  sigmoid_gradient(np.dot(a_1[-2], theta_1[-1]), S)
        for i in xrange(2, num_lay_1 + 1):
            d_1[-i] = np.dot(d_1[-i + 1], theta_1[-i + 1][1:, :].T) * \
                      sigmoid_gradient(np.dot(a_1[-i - 1], theta_1[-i]), S)

    # Gradient with regularization for the weights in the first subnetwork
    grad_1 = range(num_lay_1)
    grad_reg_1 = range(num_lay_1)
    if num_lay_1 == 1:      # One layer condition
        grad_1 = np.dot(a_1[0].T, d_1)  # gradient
        grad_reg_1 = grad_1 / m + R * theta_1 / m  # regularization term
        grad_reg_1[0, :] = grad_1[0, :] / m  # exclude weights of the bias unit
    else:
        for i in xrange(num_lay_1):
            grad_1[i] = np.dot(a_1[i].T, d_1[i])  # gradient
            grad_reg_1[i] = grad_1[i] / m + R * theta_1[i] / m  # regularization term
            grad_reg_1[i][0, :] = grad_1[i][0, :] / m  # exclude weights of the bias unit

    # Gradient with regularization for the weights in relation weights matrix
    rel_grad = np.dot(rel_input_b.T, d_2[0])
    rel_grad_reg = rel_grad / m + R * theta_relation / m
    rel_grad_reg[0, :] = rel_grad[0, :] / m

    # Gradient with regularization for the weights in the second subnetwork
    grad_2 = range(num_lay_2 + 1)
    grad_reg_2 = range(num_lay_2 + 1)
    grad_2[0] = np.dot(a_1[-1].T, d_2[0])  # first matrix
    grad_reg_2[0] = grad_2[0] / m + R * theta_2[0] / m
    grad_reg_2[0][0, :] = grad_2[0][0, :] / m
    for i in xrange(1, num_lay_2 + 1):  # other matrices
        grad_2[i] = np.dot(a_2[i - 1].T, d_2[i])
        grad_reg_2[i] = grad_2[i] / m + R * theta_2[i] / m
        grad_reg_2[i][0, :] = grad_2[i][0, :] / m
    return grad_reg_1, grad_reg_2, rel_grad_reg


def descent(theta_1, theta_2, theta_relation, gradient_1, gradient_2,
            gradient_rel, num_lay_1, num_lay_2, alpha, moment_1, moment_2,
            moment_relation, M):
    """
    Change matrices of weights according to the gradient.
    Returns:
        ----
        theta_1_temp: new theta matrices for the first subnetwork (list of arrays)
        theta_2_temp: new theta matrices for the second subnetwork (list of arrays)
        theta_relation_temp: new theta matrix for relation weights (array)
    """
    theta_1_temp = range(num_lay_1)
    if num_lay_1 == 1:
        theta_1_temp = (theta_1 - alpha*gradient_1) - M*moment_1 # Change weights in the first subnetwork
    else:
        for i in xrange(num_lay_1):
            theta_1_temp[i] = (theta_1[i] - alpha*gradient_1[i]) - M*moment_1[i] # Change weights in the first subnetwork
    theta_relation_temp = (theta_relation - alpha * gradient_rel) - M*moment_relation # Change relation weights
    theta_2_temp = range(num_lay_2 + 1)
    for i in xrange(num_lay_2 + 1):
        theta_2_temp[i] = (theta_2[i] - alpha * gradient_2[i]) - M * moment_2[i]  # Change weights in the second subnetwork
    # Accumulating moment
    moment_1 += gradient_1
    moment_2 += gradient_2
    moment_relation += gradient_rel

    return theta_1_temp, theta_2_temp, theta_relation_temp


def gradient_check(S, e, m, X, Y, input_relation, theta_1, theta_2, theta_relation,
                   num_lay_1, num_lay_2, R):
    """
    Computes the numerical gradient of the function around theta for every weight.
    Returns:
        ----
        numgrad_1: numerical estimation of the gradients in the first subnetwork
                (list of arrays)
        numgrad_2: numerical estimation of the gradients in the second subnetwork
                (list of arrays)
        numgrad_rel: numerical estimation of the gradients for the raletion
                weights (list of arrays)
    """
    # Estimation for the first subnetwork:
    numgrad_1 = range(num_lay_1)
    # One layer case
    if num_lay_1 == 1:
        numgrad_1 = np.zeros((np.shape(theta_1)))
        perturb = np.zeros((np.shape(theta_1)))
        for p in xrange(np.size(theta_1)):  # loop over all of the elements in the current weight matrix
            perturb.flat[p] = e  # change only one element per cycle, the others still equal zero
            th_ch_minus = theta_1 - perturb  # create weight matrices with one weight changed
            th_ch_plus = theta_1 + perturb
            # Compute "minus" error:
            [a_chm_1, a_chm_2] = forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                     input_relation, th_ch_minus, theta_2,
                                                     theta_relation)      # perform forward propagation with new weight
            j_ch_minus = compute_cost_function(m, a_chm_2, th_ch_minus, theta_2,
                                               theta_relation, num_lay_1, num_lay_2, R,
                                               Y)     # compute  new cost function
            # Compute "plus" error:
            [a_chp_1, a_chp_2] = forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                     input_relation, th_ch_plus, theta_2,
                                                     theta_relation)  # perform forward propagation with new weight
            j_ch_plus = compute_cost_function(m, a_chp_2, th_ch_plus, theta_2,
                                              theta_relation, num_lay_1, num_lay_2, R,
                                              Y)         # compute  new cost function
            numgrad_1.flat[p] = (j_ch_plus - j_ch_minus) / (2 * e)   # numerical estimation for one particular weight
            perturb.flat[p] = 0                                         # prepare "perturb" for the further computations
    # Multi layer case
    else:
        for i in xrange(num_lay_1):  # loop over the weight matrices in the first subnetwork
            numgrad_1[i] = np.zeros(
                (np.shape(theta_1[i])))  # structure which will contain estimation of the gradient for every weight
            perturb = np.zeros(
                (np.shape(theta_1[i])))  # this structure we will use to change weights in the original matrix
            for p in xrange(np.size(theta_1[i])):  # loop over all of the elements in the current weight matrix
                perturb.flat[p] = e  # change only one element per cycle, the others still equal zero
                th_ch_minus = theta_1[:]
                th_ch_plus = theta_1[:]
                th_ch_minus[i] = theta_1[i] - perturb  # create weight matrices with one weight changed
                th_ch_plus[i] = theta_1[i] + perturb
                # Compute "minus" error:
                [a_chm_1, a_chm_2] = forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                         input_relation, th_ch_minus, theta_2,
                                                         theta_relation)      # perform forward propagation with new weight
                j_ch_minus = compute_cost_function(m, a_chm_2, th_ch_minus, theta_2,
                                                   theta_relation, num_lay_1, num_lay_2, R,
                                                   Y)     # compute  new cost function
                # Compute "plus" error:
                [a_chp_1, a_chp_2] = forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                         input_relation, th_ch_plus, theta_2,
                                                         theta_relation)  # perform forward propagation with new weight
                j_ch_plus = compute_cost_function(m, a_chp_2, th_ch_plus, theta_2,
                                                  theta_relation, num_lay_1, num_lay_2, R,
                                                  Y)         # compute  new cost function
                numgrad_1[i].flat[p] = (j_ch_plus - j_ch_minus) / (2 * e)   # numerical estimation for one particular weight
                perturb.flat[p] = 0                                         # prepare "perturb" for the further computations

    # Estimation for the relation matrix:
    numgrad_rel = np.zeros((np.shape(theta_relation)))  # structure which will contain estimation
    # of the gradient for every weight
    perturb = np.zeros((np.shape(theta_relation)))  # this structure we will use to change weights
    # in the original matrix
    for p in xrange(np.size(theta_relation)):   # loop over all of the elements in the current weight matrix
        perturb.flat[p] = e                     # change only one element per cycle, the others still equal zero
        th_ch_minus = theta_relation - perturb  # create weight matrices with one weight changed
        th_ch_plus = theta_relation + perturb
        # Compute "minus" error:
        [a_chm_1, a_chm_2] = forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                 input_relation, theta_1, theta_2,
                                                 th_ch_minus)          # perform forward propagation with new weight
        j_ch_minus = compute_cost_function(m, a_chm_2, theta_1, theta_2,
                                           th_ch_minus, num_lay_1, num_lay_2, R,
                                           Y)                # compute  new cost function
        # Compute "plus" error:
        [a_chp_1, a_chp_2] = forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                 input_relation, theta_1, theta_2,
                                                 th_ch_plus)       # perform forward propagation with new weight
        j_ch_plus = compute_cost_function(m, a_chp_2, theta_1, theta_2,
                                          th_ch_plus, num_lay_1, num_lay_2, R,
                                          Y)             # compute  new cost function
        numgrad_rel.flat[p] = (j_ch_plus - j_ch_minus) / (2 * e)    # numerical estimation for one particular weight
        perturb.flat[p] = 0                                         # prepare "perturb" for the further computations

    # Estimation for the second subnetwork:
    numgrad_2 = range(num_lay_2 + 1)
    for i in xrange(num_lay_2 + 1):  # loop over the weight matrices in the second subnetwork
        numgrad_2[i] = np.zeros((np.shape(theta_2[i])))     # structure which will contain estimation
        # of the gradient for every weight
        perturb = np.zeros((np.shape(theta_2[i])))          # this structure we will use to change
        # weights in the original matrix
        for p in xrange(np.size(theta_2[i])):  # loop over all of the elements in the current weight matrix
            perturb.flat[p] = e  # change only one element per cycle, the others still equal zero
            th_ch_minus = theta_2[:]
            th_ch_plus = theta_2[:]
            th_ch_minus[i] = theta_2[i] - perturb  # create weight matrices with one weight changed
            th_ch_plus[i] = theta_2[i] + perturb
            # Compute "minus" error:
            [a_chm_1, a_chm_2] = forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                     input_relation, theta_1, th_ch_minus,
                                                     theta_relation)   # perform forward propagation with new weight
            j_ch_minus = compute_cost_function(m, a_chm_2, theta_1, th_ch_minus, # compute  new cost function
                                               theta_relation, num_lay_1, num_lay_2, R, Y)
            # Compute "plus" error:
            [a_chp_1, a_chp_2] = forward_propagation(S, m, num_lay_1, num_lay_2, X,
                                                     input_relation, theta_1, th_ch_plus,
                                                     theta_relation)   # perform forward propagation with new weight
            j_ch_plus = compute_cost_function(m, a_chp_2, theta_1, th_ch_plus, # compute  new cost function
                                              theta_relation, num_lay_1, num_lay_2, R, Y)
            numgrad_2[i].flat[p] = (j_ch_plus - j_ch_minus) / (2 * e)  # numerical estimation for one particular weight
            perturb.flat[p] = 0  # prepare "perturb" for the further computations

    return numgrad_1, numgrad_2, numgrad_rel


def verify_gradient(gradient_1, gradient_2, gradient_rel, numgrad_1, numgrad_2, numgrad_rel):
    """ Show maximum difference between grdients for every weight matrix"""
    diff_1 = range(len(gradient_1))
    for i in xrange(len(gradient_1)):
        diff_1[i] = np.max(np.abs(numgrad_1[i] - gradient_1[i]))
    diff_2 = range(len(gradient_2))
    for i in xrange(len(gradient_2)):
        diff_2[i] = np.max(np.abs(numgrad_2[i] - gradient_2[i]))
    diff_rel = np.max(np.abs(numgrad_rel - gradient_rel))
    print "differences in the first subnetwork: ",diff_1
    print "difference in the relation subnetwork: ",diff_rel
    print "differences in the second subnetwork: ",diff_2


def check_result(example, epoch, file, hidden_1, hidden_2, theta_history, S):
    """
    Takes one example from learning data(example) and learned weight matrices
    from theta_hisory according to the particular epoch of learning.
    Print output values of the network(res) with respect to attributes(attr_names)
    and original values(teacher).
    """
    # Import data
    [item, rel, attr] = complex_data_preparation(file)
    # Assign local variables
    [theta_1, theta_2, theta_relation] = theta_history[epoch]
    num_lay_1 = len(hidden_1)
    num_lay_2 = len(hidden_2)
    m = 1
    # Set data
    X = item[example]
    input_relation = rel[example]
    Y = attr[example]
    # Recognize item input
    if (X == [1,0,0,0]).all():
        object = 'Pozharniy'
    elif (X == [0,1,0,0]).all():
        object = 'Stas'
    elif (X == [0,0,1,0]).all():
        object = 'Vanya'
    elif (X == [0,0,0,1]).all():
        object = 'Sergey Sergeich'
    # Recognize relation input
    if (input_relation == [1,0,0,0]).all():
        relation = 'moget'
    elif (input_relation == [0,1,0,0]).all():
        relation = 'imeet'
    elif (input_relation == [0,0,1,0]).all():
        relation = 'yavlyaetsya'
    elif (input_relation == [0,0,0,1]).all():
        relation = 'kakoy'

    # Compute activations of every unit in the network.
    [a_1, a_2] = forward_propagation(S, m, num_lay_1, num_lay_2,
                                                    X.reshape(1,4), input_relation.reshape(1,4), theta_1,
                                                    theta_2, theta_relation)
    # List of attribute names
    attr_names = ['tushit pozhar', 'mit mashinu', 'halturit', 'formu', 'brandspoit',
    'raciyu', 'evgrafichem', 'komandirom', 'predprinimatelem', 'vesyoliy', 'pozhiloy', 'hitriy',
    'smotryet futbol', 'chinyit mashinu', 'strichsya', 'korotkuyu strizhku', 'vnedorozhnik',
    'voblu', 'naparnyikon vani', 'dalnoboyshikom', 'bolelshikom', 'dobriy', 'poryadochniy',
    'krugloliciy', 'prosit prosheniya', 'rabotat s policiyei', 'ugnat furu', 'svyaz s bandoy',
    'porvanuyu kurtku', 'leviye dengi', 'sinom direktora', 'voditelyem', 'vtyanutim v prestuplyeniye',
    'nayivniy', 'molodoy', 'doverchiviy', 'szhigat', 'igrat v billiard', 'ugrozhat', 'ochki',
    'krminalniy bisness', 'grubiy golos', 'prestupnikom', 'bisnessmenom', 'glavaryom',
    'opasniy', 'rasoblachyonniy', 'zhadniy']

    # transform arrays in lists
    res = []
    for i in xrange(np.size((a_2[-1]))):
        res.append(a_2[-1][0][i])
    teacher = []
    for i in xrange(np.size((Y))):
        teacher.append(Y[i])
    for i in xrange(len(attr_names)):
        print object+' '+relation+' '+attr_names[i]+':  '+str(res[i])+' ('+str(teacher[i])+')'



