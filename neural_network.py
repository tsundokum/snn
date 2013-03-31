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


def sigmoid(z):
    """Compute sigmoid function for given number z."""
    return 1 / (1.0 + np.exp(-z))


def sigmoid_gradient(z):
    """Compute partial derivative of sigmoid function with respect to value z"""
    q = sigmoid(z)
    return q * (1 - q)


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
    w_struct_1 = np.hstack((input_size, hidden_1))  # append input vector as the first layer
    theta_1 = range(num_lay_1)  # create list ready to fill with matrices of weights
    for i in xrange(num_lay_1):  # loop over the layers of the first subnetwork
        # matrix with random values from (-1 * epsilon) to (1 * epsilon)
        theta_1[i] = (np.random.rand(w_struct_1[i] + 1, w_struct_1[i + 1]) * 2 - 1) * epsilon

    # Second subnetwork
    w_struct_2 = np.hstack((hidden_1[-1], hidden_2, output_size))  # append input vector as the first layer
    theta_2 = range(num_lay_2 + 1)  # create list ready to fill with matrices of weights
    for i in xrange(num_lay_2 + 1):  # loop over the layers of the second subnetwork
        # matrix with random values from (-1 * epsilon) to (1 * epsilon)
        theta_2[i] = (np.random.rand(w_struct_2[i] + 1, w_struct_2[i + 1]) * 2 - 1) * epsilon

    # Intermediate subnetwork (relation)
    # matrix with random values from (-1 * epsilon) to (1 * epsilon)
    theta_relation = (np.random.rand(relation_in_size + 1, hidden_2[0]) * 2 - 1) * epsilon
    return theta_1, theta_2, theta_relation


def forward_propagation(m, num_lay_1, num_lay_2, X, input_relation, theta_1,
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
    for i in xrange(1, num_lay_1 + 1):  # loop over the first subnetwork
        z_1[i] = np.dot(a_1[i - 1], theta_1[i - 1])  # perform matrix multiplication to compute sum for every unit
        a_1[i] = np.hstack((np.ones((m, 1)), sigmoid(z_1[i])))  # compute sigmoid function and add bias units

    # Second subnetwork:
    z_2 = range(num_lay_2 + 1)
    a_2 = range(num_lay_2 + 1)
    rel_input_b = np.hstack((np.ones((m, 1)), input_relation))          # add bias term to the relation input,
    # a_1[-1] already have bias
    z_2[0] = np.dot(a_1[-1], theta_2[0]) + np.dot(rel_input_b, theta_relation)  # first layer in the second subnetwork
    a_2[0] = np.hstack((np.ones((m, 1)), sigmoid(z_2[0])))
    for i in xrange(1, num_lay_2 + 1):                   # loop over the other layers of the second subnetwork
        z_2[i] = np.dot(a_2[i - 1], theta_2[i])
        a_2[i] = np.hstack((np.ones((m, 1)), sigmoid(z_2[i])))
    a_2[-1] = a_2[-1][:, 1:]                             # remove bias unit from the last(output) layer
    return a_1, a_2


def compute_cost_function(m, a_2, theta_1, theta_2, theta_relation,
                          num_lay_1, num_lay_2, R, Y):
    """
    Compute average error with regularization.
    Returns:
        ----
        J: approximate error (float)
    """
    # Average cost
    cost = np.sum(-Y * np.log(a_2[-1]) - (1 - Y) * np.log(1 - a_2[-1])) / m
    # Regularization
    for i in xrange(num_lay_1):
        reg_1 = np.sum(theta_1[i][:, 1:] ** 2)
    reg_relation = np.sum(theta_relation[:, 1:] ** 2)
    for i in xrange(num_lay_2 + 1):
        reg_2 = np.sum(theta_2[i][:, 1:] ** 2)
    regularization = (reg_1 + reg_2 + reg_relation) * (R / (2 * m))
    J = cost + regularization
    return J


def back_propagation(m, a_1, a_2, input_relation, theta_1, theta_2, theta_relation,
                     num_lay_1, num_lay_2, R, Y):
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
    d_2 = range(num_lay_2 + 1)
    rel_input_b = np.hstack((np.ones((m, 1)), input_relation))
    d_2[-1] = a_2[-1] - Y
    for i in xrange(2, num_lay_2 + 1):
        d_2[-i] = np.dot(d_2[-i + 1], theta_2[-i + 1][1:, :].T) * \
                  sigmoid_gradient(np.dot(a_2[-i - 1], theta_2[-i]))
    d_2[0] = np.dot(d_2[1], theta_2[1][1:, :].T) * \
             (sigmoid_gradient(np.dot(a_1[-1], theta_2[0])) +
             sigmoid_gradient(np.dot(rel_input_b, theta_relation)))
    # Errors of neurons in th first subnetwork
    d_1 = range(num_lay_1)
    d_1[-1] = np.dot(d_2[0], theta_2[0][1:, :].T) * \
              sigmoid_gradient(np.dot(a_1[-2], theta_1[-1]))
    for i in xrange(2, num_lay_1 + 1):
        d_1[-i] = np.dot(d_1[-i + 1], theta_1[-i + 1][1:, :].T) * \
                  sigmoid_gradient(np.dot(a_1[-i - 1], theta_1[-i]))

    # Gradient with regularization for the weights in the first subnetwork
    grad_1 = range(num_lay_1)
    grad_reg_1 = range(num_lay_1)
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


def descent(theta_1, theta_2, theta_relation, grad_reg_1, grad_reg_2,
            rel_grad_reg, num_lay_1, num_lay_2, alpha):
    """
    Change matrices of weights according to the gradient.
    Returns:
        ----
        theta_1_temp: new theta matrices for the first subnetwork (list of arrays)
        theta_2_temp: new theta matrices for the second subnetwork (list of arrays)
        theta_relation_temp: new theta matrix for relation weights (array)
    """
    theta_1_temp = range(num_lay_1)
    for i in xrange(num_lay_1):
        theta_1_temp[i] = theta_1[i] - alpha * grad_reg_1[i]  # Change weights in the first subnetwork
    theta_relation_temp = theta_relation - alpha * rel_grad_reg  # Change relation weights
    theta_2_temp = range(num_lay_2 + 1)
    for i in xrange(num_lay_2 + 1):
        theta_2_temp[i] = theta_2[i] - alpha * grad_reg_2[i]  # Change weights in the second subnetwork
    return theta_1_temp, theta_2_temp, theta_relation_temp


def gradient_check(e, m, X, Y, input_relation, theta_1, theta_2, theta_relation,
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
            [a_chm_1, a_chm_2] = forward_propagation(m, num_lay_1, num_lay_2, X,
                                                     input_relation, th_ch_minus, theta_2,
                                                     theta_relation)      # perform forward propagation with new weight
            j_ch_minus = compute_cost_function(m, a_chm_2, th_ch_minus, theta_2,
                                               theta_relation, num_lay_1, num_lay_2, R,
                                               Y)     # compute  new cost function
            # Compute "plus" error:
            [a_chp_1, a_chp_2] = forward_propagation(m, num_lay_1, num_lay_2, X,
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
        [a_chm_1, a_chm_2] = forward_propagation(m, num_lay_1, num_lay_2, X,
                                                 input_relation, theta_1, theta_2,
                                                 th_ch_minus)          # perform forward propagation with new weight
        j_ch_minus = compute_cost_function(m, a_chm_2, theta_1, theta_2,
                                           th_ch_minus, num_lay_1, num_lay_2, R,
                                           Y)                # compute  new cost function
        # Compute "plus" error:
        [a_chp_1, a_chp_2] = forward_propagation(m, num_lay_1, num_lay_2, X,
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
            [a_chm_1, a_chm_2] = forward_propagation(m, num_lay_1, num_lay_2, X,
                                                     input_relation, theta_1, th_ch_minus,
                                                     theta_relation)   # perform forward propagation with new weight
            j_ch_minus = compute_cost_function(m, a_chm_2, theta_1, th_ch_minus, # compute  new cost function
                                               theta_relation, num_lay_1, num_lay_2, R, Y)
            # Compute "plus" error:
            [a_chp_1, a_chp_2] = forward_propagation(m, num_lay_1, num_lay_2, X,
                                                     input_relation, theta_1, th_ch_plus,
                                                     theta_relation)   # perform forward propagation with new weight
            j_ch_plus = compute_cost_function(m, a_chp_2, theta_1, th_ch_plus, # compute  new cost function
                                              theta_relation, num_lay_1, num_lay_2, R, Y)
            numgrad_2[i].flat[p] = (j_ch_plus - j_ch_minus) / (2 * e)  # numerical estimation for one particular weight
            perturb.flat[p] = 0  # prepare "perturb" for the further computations

    return numgrad_1, numgrad_2, numgrad_rel
