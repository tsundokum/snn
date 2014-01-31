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
import codecs
import numpy as np
import csv
import os
import xlrd


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

def sigmoid(z, S):
    """Compute sigmoid function for given number z."""
    return 1 / (1.0 + np.exp(-S*z))


def sigmoid_gradient(z, S):
    """Compute partial derivative of sigmoid function with respect to value z"""
    q = sigmoid(z, S)
    return S * q * (1 - q)


def data_preparation_xls(file_name, data_representation):
    """
    Prepare learning data from given xls-file.
    Takes file name and data_representation parameters written as strings.
    Returns matrixes in format: number of examples by number of dimensions.
    For complex representation:
    Every row in input matrix represents one learning example in which the
    corresponding number equals one and others are zero.
    (Matrix for attributes contains all information for the current input pair!)
    For separate representation:
    Every row in the matrix represents one learning example in which the
    corresponding number equals one and others are zero.
    """
    # read file
    rb = xlrd.open_workbook(file_name)
    sheet = rb.sheet_by_name('Base')  # take data from particular page

    relations = sheet.col_values(0)[1:]  # take all the relation names
    for r in xrange(len(relations)):
        if relations[r] != relations[r+1]: break  # countitng attributes
    # define dimensionality  of learning matrices
    num_of_attr = r + 1
    num_of_items = len(sheet.row_values(0)[2:])
    num_of_rel = len(relations) / num_of_attr
    # create list of all connection values
    val_list = sheet.col_values(2)[1:]
    for col in xrange(num_of_items-1):
        val_list.extend(sheet.col_values(col+3)[1:])
    #  create learning matrices with respect to the data_representation requirements
    if data_representation == 'complex':
        num_ex = num_of_items * num_of_rel  # number of examples
        item_matrix = np.zeros((num_ex, num_of_items))
        rel_matrix = np.zeros((num_ex, num_of_rel))
        attr_matrix = np.zeros((num_ex, num_of_attr))
        for ex in xrange(num_ex):       # fill the attribute matrix
            attr_matrix[ex, :] = val_list[num_of_attr*ex : num_of_attr*(ex+1)]
        for i in xrange(num_of_items):  # fill item learning matrix
            item_matrix[i*num_of_items:(i+1)*num_of_items, i] = 1
        for j in xrange(num_of_items):  # fill the relation learning matrix
            for i in xrange(num_of_rel):
                rel_matrix[j*num_of_items+i, i] = 1

        return item_matrix, rel_matrix, attr_matrix

    elif data_representation == 'separate':
        num_ex = num_of_items * num_of_rel * num_of_attr  # number of examples
        num_per_item = num_of_attr * num_of_rel
        data = np.zeros((num_ex, 4))
        # Fill the first column with numbers of items
        for i in xrange(num_of_items):
            data[i*num_per_item : (i+1)*num_per_item, 0] = i + 1
        # Fill the second column with numbers of relations
            tab = data[i*num_per_item : (i+1)*num_per_item]
            for j in xrange(num_of_rel):
                tab[j*num_of_attr : (j+1)*num_of_attr, 1] = j + 1
            data[i*num_per_item : (i+1)*num_per_item] =  tab
        # Fill the third column with numbers of attributes
        for i in xrange(num_of_items * num_of_rel):
            tab = data[i*num_of_attr : (i+1)*num_of_attr]
            for j in range(num_of_attr):
                tab[j, 2] = j + 1
            data[i*num_of_attr : (i+1)*num_of_attr] = tab
        # Fill the fourth column with connection value
        data[:, 3] = val_list

        # Create learning matrices.
        item_matrix = np.zeros((num_ex, num_of_items))
        for i in xrange(num_ex):
            item_matrix[i, data[i, 0]-1] = 1
        rel_matrix = np.zeros((num_ex, num_of_rel))
        for i in xrange(num_ex):
            rel_matrix[i, data[i, 1]-1] = 1
        attr_num = np.zeros((num_ex, 1), dtype = 'int8')
        attr_val = np.zeros((num_ex, 1))
        for i in xrange(num_ex):
            attr_num[i] = data[i, 2] - 1
            attr_val[i] = data[i, 3]
        attr_matrix = [attr_num, attr_val]

        return item_matrix, rel_matrix, attr_matrix


# Function to prepare full(complex) training examples
def complex_data_preparation(file):
    """
    Prepare learning data from given csv-file (comma separated).
    ((Represent data in 16 examples))
    Takes file name written as string.
    Returns matrixes in format: number of examples by number of dimensions.
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
def separate_data_preparation2(file):
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
        attr: learning matrix for attributes (array)

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


# Function to prepare partial(separate) training examples
def separate_data_preparation(file_name):
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
        attr: learning matrix for attributes (array)
    """
    # Extract data from file as list of strings
    table = []
    with codecs.open(file_name, 'rb', 'utf-8') as infile:
        for line in infile:
            row = [x.replace(u'\ufeff','').strip() for x in line.split(',')]
            table.append(row)

    m = len(table)

    items = [row[0] for row in table]
    uniq_items = sorted(set(items))
    item_indexi = [uniq_items.index(it) for it in items]
    item_matrix = np.zeros((m, len(uniq_items)), dtype="int")
    item_matrix[np.arange(m), item_indexi] = 1

    relations = [row[1] for row in table]
    uniq_relations = sorted(set(relations))
    relations_indexi = [uniq_relations.index(rel) for rel in relations]
    relation_matrix = np.zeros((m, len(uniq_relations)), dtype="int")
    relation_matrix[np.arange(m), relations_indexi] = 1

    attrs = [row[2] for row in table]
    uniq_attrs = sorted(set(attrs))
    attr_indexi = np.array([uniq_attrs.index(at) for at in attrs], dtype="int")[:, np.newaxis]

    attr_values = np.array([float(row[3]) for row in table])[:, np.newaxis]

    return item_matrix, relation_matrix, attr_indexi, attr_values


def big_data_preparation(file_dir):
    """ Takes directory with origin data files.
        Returns big data set with assembled origin data."""
    full_item = []
    full_rel = []
    full_attr = []
    for f in os.listdir(file_dir):
        if f[-3:] == 'csv':
            [item, rel, attr] = complex_data_preparation(file_dir+'\\'+f)
        elif f[-3:] == 'xls':
            [item, rel, attr] = data_preparation_xls(file_dir+'\\'+f, 'complex')
        full_item.append(item)
        full_rel.append(rel)
        full_attr.append(attr)
    full_item = np.vstack((full_item))
    full_rel = np.vstack((full_rel))
    full_attr = np.vstack((full_attr))

    return full_item, full_rel, full_attr


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
        z_1[1] = np.dot(a_1[0], theta_1)
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


def mean_squares(m, a_2, Y, data_representation):
    """ Compute least-squares cost function"""
    if (data_representation == 'complex') or (data_representation == 'large'):
        cost = np.sum((a_2[-1] - Y) ** 2) / m
    elif data_representation == 'separate':
        cost = 0
        for i in xrange(m):
            cost += (a_2[-1][i][Y[0][i]] - Y[1][i]) ** 2
        cost /= m
    return cost


def cross_entropy(m, a_2, theta_1, theta_2, theta_relation,
                  num_lay_1, num_lay_2, R, Y, data_representation):
    """ Compute cross-entropy cost function"""
    # Average cost
    if (data_representation == 'complex') or (data_representation == 'large'):
        cost = np.sum(-Y * np.log(a_2[-1]) - (1 - Y) * np.log(1 - a_2[-1])) / m
    elif data_representation == 'separate':
        cost = 0
        for i in xrange(m):
            cost_ex = -Y[1][i] * np.log(a_2[-1][i][Y[0][i]]) - (1 - Y[1][i]) * \
                      np.log(1 - a_2[-1][i][Y[0][i]])
            cost += cost_ex
        cost /= m
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


def compute_cost_function(cost_function, m, a_2, theta_1, theta_2, theta_relation,
                          num_lay_1, num_lay_2, R, Y, data_representation):
    """
    Compute average error.
    Returns:
        ----
        J: approximate error (float)
    """
    if cost_function == 'mean_squares':
        J = mean_squares(m, a_2, Y, data_representation)
    elif cost_function == 'cross_entropy':
        J = cross_entropy(m, a_2, theta_1, theta_2, theta_relation,
                          num_lay_1, num_lay_2, R, Y, data_representation)
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
    if (data_representation == 'complex') or (data_representation == 'large'):
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


def compute_jacobian(S, m, a_1, a_2, input_relation, theta_1, theta_2, theta_relation,
                     num_lay_1, num_lay_2, R, Y, data_representation):
    """
    Compute partial derivatives with respect to weigths "theta"
    according to every example in the training set.
    """
    if num_lay_1 == 1:  # condition for one-layer subnet
        jacobian_1 = np.zeros((m, np.size(theta_1)))
    else:
        jacobian_1 = []
        for theta in theta_1:
            jacobian_1.append(np.zeros((m, np.size(theta))))
    jacobian_2 = []
    for theta in theta_2:
        jacobian_2.append(np.zeros((m, np.size(theta))))
    jacobian_rel = np.zeros((m, np.size(theta_relation)))
    # loop over examples in the set
    for i in xrange(m):
        #  Take parameters of the given example
        a_1ch = []
        for a in a_1:   # loop over the layers
            a_1ch.append(a[i].reshape(1, np.size(a, 1)))
        a_2ch = []
        for a in a_2:   # loop over the layers
            a_2ch.append(a[i].reshape(1, np.size(a, 1)))
        in_rel = input_relation[i].reshape(1, np.size(input_relation, 1))
        if  data_representation == 'separate':
            y = [Y[0][i], Y[1][i]]
        else:
            y = Y[i].reshape(1, np.size(Y, 1))
        [gradient_1, gradient_2,
         gradient_rel] = back_propagation(S, 1, a_1ch, a_2ch, in_rel, theta_1,
                                                         theta_2, theta_relation, num_lay_1,
                                                         num_lay_2, R, y, data_representation)
        # filling jacobian matrices for the first subnet:
        if num_lay_1 == 1:
            jacobian_1[i, :] = gradient_1.flatten()
        else:
            for j in xrange(len(theta_1)):
                jacobian_1[j][i, :] = gradient_1[j].flatten()
        # for the second subnet
        for j in xrange(len(theta_2)):
            jacobian_2[j][i, :] = gradient_2[j].flatten()
        jacobian_rel[i, :] = gradient_rel.flatten()

    return jacobian_1, jacobian_2, jacobian_rel


def delta_LMA(jacobian, L):
    """ Computes weiths changes for according to
        Levenberg-Markwardt algorithm"""
    hessian = np.dot(jacobian.T,jacobian)
    diagonal = np.diag(hessian)
    invert = np.linalg.pinv(hessian + (L * diagonal))
    changes = np.dot(jacobian, invert)
    changes = np.mean(changes, 0)
    return changes


def LMA_optimisation(jacobian_1, jacobian_2, jacobian_rel, L, theta_1, theta_2, theta_relation):
    """
    Change weights according to the given jacobian-matrices
    """
    # Compute weights changes for the first subnet
    if  not isinstance(theta_1, list):
        delta_1 = delta_LMA(jacobian_1, L).reshape(np.shape(theta_1))  #  for one-layer subnet
        theta_1_temp = theta_1 - delta_1
    else:
        delta_1 = range(len(theta_1))
        theta_1_temp = range(len(theta_1))
        for j in range(len(theta_2)):
            delta_1[j] = delta_LMA(jacobian_1[j], L).reshape(np.shape(theta_1[j]))  # for multi-layer subnet
            theta_1_temp[j] = theta_1[j] - delta_1[j]
    # Compute weights changes for the second subnet
    delta_2 = range(len(theta_2))
    theta_2_temp = range(len(theta_2))
    for j in range(len(theta_2)):
        delta_2[j] = delta_LMA(jacobian_2[j], L).reshape(np.shape(theta_2[j]))
        theta_2_temp[j] = theta_2[j] - delta_2[j]
    # Changes for relation weights
    delta_rel = delta_LMA(jacobian_rel, L).reshape(np.shape(theta_relation))
    theta_relation_temp = theta_relation - delta_rel

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


def check_result(example, epoch, file_name, hidden_1, hidden_2, theta_history, S):
    """
    Takes one example from learning data(example) and learned weight matrices
    from theta_hisory according to the particular epoch of learning.
    Print output values of the network(res) with respect to attributes(attr_names)
    and original values(teacher).
    """
    # Import data
    if file_name[-4:] == '.xls':
        [item, rel, attr] = data_preparation_xls(file_name, 'complex')
    elif file_name[-4:] == '.csv':
        [item, rel, attr] = complex_data_preparation(file_name)
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
    check_result = []
    for i in xrange(np.size((Y))):
        teacher.append(Y[i])
    for i in xrange(len(attr_names)):
        check_result.append(object+' '+relation+' '+attr_names[i]+':  '+str(res[i])+' ('+str(teacher[i])+')')
    return '\n'.join(check_result)



# ANALYTICAL FUNCTIONS:

# Variance analysis
def wVar(theta):
    """ Compute sum of variances of all weigths for every unit in the layer"""
    disp = 0
    for row in theta[1:]:
        disp += np.var(row)
    return disp


def pair_combs(origList):
    """ Returns every pair combination, of the origList in newList"""
    newList = []
    for a in range(len(origList)):
        for b in range(len(origList)):
            if a != b and [origList[b], origList[a]] not in newList:
                newList.append([origList[a],origList[b]])
    return np.array(newList)


def wPairVAr(theta):
    """ Compute variance between every pair of weigths for evry unit in the layer"""
    for row in theta[1:]:    # Take all weights of every unit in the next layer
        pairCombs = pair_combs(row)  # find all the pair combinations of this weights
        variance = np.var(pairCombs, 1)  # compute variance for every pair combiantion
        totalVar = np.sum(variance)  # compute sum
        return totalVar


def actVar(S, hidden_1, hidden_2, file_name, theta_1, theta_2, theta_relation):
    """ Returns mean variance for every layer over all learning examples."""
    [item, rel, attr] = complex_data_preparation(file_name)
    input_size = np.size(item, 1)  # Item number
    relation_in_size = np.size(rel, 1)  # Relations number
    output_size = 48                    # Number of attributes
    num_lay_1 = len(hidden_1)           # Number of layers in the first subnetwork
    num_lay_2 = len(hidden_2)           # Number of layers in the second subnetwork
    batch_size = len(item)
    m = batch_size    # change "m" for batch_size
    [a_1, a_2] = forward_propagation(S, m, num_lay_1, num_lay_2, item,
                                                    rel, theta_1, theta_2, theta_relation)
    var = []
    for a in [a_1[1], a_2[0], a_2[1]]:
        row_var = []
        for row in a:
            row_var.append(np.var(row[1:]))
        var.append(np.mean(row_var))

    return var







