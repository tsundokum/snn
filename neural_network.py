#-------------------------------------------------------------------------------
# Name:        neural_network
# Purpose:
#
# Author:      11
#
# Created:     08.03.2013
# Copyright:   (c) 11 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
'''
...
'''
import numpy

def initialise_weights(input_size, hidden_1, hidden_2, relation_in_size,
         output_size, num_lay_1, num_lay_2, epsilon):
    '''
    Generate 3 sets of matrices of initial weights according to the given structure.

    Parameters
    ----
    input_size: Item number
    hidden_1: Struscture of the first subnetwork
    hidden_2: Structure of the second subnetwork
    relation_in_size: Relations number
    output_size: Number of properties
    epsilon: Limitation of  initial eights
    '''
    # First subnetwork
    w_struct_1 = numpy.hstack((input_size, hidden_1)) # Append input vector as the first layer
    theta_1 = range(num_lay_1)
    for i in range(num_lay_1):
        rand_sign = numpy.random.randint(-1, 2, (w_struct_1[i] + 1,
                w_struct_1[i + 1])) # Matrix for sign randomisation
        theta_1[i] = numpy.random.rand(w_struct_1[i] + 1, w_struct_1[i + 1]) # Matrix with random values
        theta_1[i] = theta_1[i] * rand_sign * epsilon # Sign randomaisation and value limaitation
    # Second subnetwork
    w_struct_2 = numpy.hstack((hidden_1[-1], hidden_2, output_size))
    theta_2 = range(num_lay_2 + 1)
    for i in range(num_lay_2 + 1):
        rand_sign = numpy.random.randint(-1, 2, (w_struct_2[i] + 1,
                w_struct_2[i + 1]))
        theta_2[i] = numpy.random.rand(w_struct_2[i] + 1, w_struct_2[i + 1])
        theta_2[i] = theta_2[i] * rand_sign * epsilon
    # Intermediate subnetwork (relation)
    rand_sign = numpy.random.randint(-1, 2, (relation_in_size + 1, hidden_2[0]))
    theta_relation = numpy.random.rand(relation_in_size + 1, hidden_2[0])
    theta_relation = theta_relation * rand_sign * epsilon
    return theta_1, theta_2, theta_relation



def sigmoid(z):
    '''Compute sigmoid function.'''
    g = 1.0 / (1.0+numpy.exp(-z));
    return g


def forward_propagation(m, num_lay_1, num_lay_2, X, input_relation, theta_1,
                 theta_2, theta_relation):
    '''Compute activations of every unit in the ntework.'''
    z_1 = range(num_lay_1 + 1)
    a_1 = range(num_lay_1 + 1)
    z_1[0] = X
    a_1[0] = numpy.hstack((numpy.ones((m,1)),z_1[0]))
    for i in range(1,num_lay_1 + 1):
        z_1[i] = numpy.dot(a_1[i-1], theta_1[i-1])
        a_1[i] = numpy.hstack((numpy.ones((m,1)), sigmoid(z_1[i])))
    z_2 = range(num_lay_2 + 1)
    a_2 = range(num_lay_2 + 1)
    # a_1[-1] already have bias
    rel_input_b = numpy.hstack((numpy.ones((m,1)), input_relation)) # add bias term to the relation input
    z_2[0] = numpy.dot(a_1[-1], theta_2[0]) + numpy.dot(rel_input_b, theta_relation) # attention biases!!
    a_2[0] = numpy.hstack((numpy.ones((m,1)), sigmoid(z_2[0])))
    for i in range(1,num_lay_2 + 1):  # changing count for z_2 and a_2
        z_2[i] = numpy.dot(a_2[i-1], theta_2[i])
        a_2[i] = numpy.hstack((numpy.ones((m,1)), sigmoid(z_2[i])))
    a_2[-1] = a_2[-1][:,1:]
    return a_1, a_2



def compute_cost_function(m, a_2, theta_1, theta_2, theta_relation,
        num_lay_1, num_lay_2, R, Y):
    ''' Compute average error with regularization'''
    cost = numpy.sum(-Y * numpy.log(a_2[-1]) - (1-Y) * numpy.log(1-a_2[-1])) / m # average cost
    for i in range(num_lay_1):
        reg_1 = numpy.sum(theta_1[i][:,1:] ** 2)
    reg_realtion = numpy.sum(theta_relation[:,1:]**2)
    for i in range(num_lay_2 + 1):
        reg_2 = numpy.sum(theta_2[i][:,1:] ** 2)
    regularization = (reg_1 + reg_2 + reg_realtion) * (R/(2*m))
    J = cost + regularization
    return J


def sigmoid_gradient(z):
    '''Compute partial derivative of sigmoid function with respect to value z'''
    q = sigmoid(z)
    g = q * (1-q)
    return g


def back_propagation(m, a_1, a_2, input_relation, theta_1, theta_2, theta_relation,
        num_lay_1, num_lay_2, R, Y):
    '''
    Compute error of every unit.
    Compute derivative of the cost function with respect to matrices theta.
    '''
    d_2 = range(num_lay_2 + 1)
    rel_input_b = numpy.hstack((numpy.ones((m,1)), input_relation))
    d_2[-1] = a_2[-1] - Y
    for i in range(2,num_lay_2 + 1):
        d_2[-i] = numpy.dot(d_2[-i + 1], theta_2[-i + 1][1:,:].T) * \
        sigmoid_gradient(numpy.dot(a_2[-i - 1], theta_2[-i]))
    d_2[0] = numpy.dot(d_2[1], theta_2[1][1:,:].T) * \
        sigmoid_gradient(numpy.dot(a_1[-1], theta_2[0]) + \
                numpy.dot(rel_input_b, theta_relation))
    d_1 = range(num_lay_1)
    d_1[-1] = numpy.dot(d_2[0], theta_2[0][1:,:].T) * \
        sigmoid_gradient(numpy.dot(a_1[-2], theta_1[-1]))
    for i in range(2,num_lay_2 + 1):
        d_1[-i] = numpy.dot(d_1[-i + 1], theta_1[-i + 1][1:,:].T) * \
        sigmoid_gradient(numpy.dot(a_1[-i - 1], theta_1[-i]))
    grad_1 = range(num_lay_1)
    grad_reg_1 = range(num_lay_1)
    for i in range(num_lay_1):
        grad_1[i] = numpy.dot(a_1[i].T, d_1[i]) / m
        grad_reg_1[i] = grad_1[i]/m + R*theta_1[i] / m
        grad_reg_1[i][0,:] = grad_1[i][0,:] / m
    # relation
    rel_grad = numpy.dot(rel_input_b.T, d_2[0])
    rel_grad_reg = rel_grad/m + R*theta_relation/m
    rel_grad_reg[0,:] = rel_grad[0,:] / m
    # second subset
    grad_2 = range(num_lay_2 + 1)
    grad_reg_2 = range(num_lay_2 + 1)
    grad_2[0] = numpy.dot(a_1[-1].T, d_2[0])
    grad_reg_2[0] = grad_2[0]/m + R*theta_2[0] / m
    grad_reg_2[0][0,:] = grad_2[0][0,:] / m
    for i in range(1, num_lay_2 + 1):
        grad_2[i] = numpy.dot(a_2[i-1].T, d_2[i])
        grad_reg_2[i] = grad_2[i]/m + R*theta_2[i]/m
        grad_reg_2[i][0,:] = grad_2[i][0,:]/m
    return grad_reg_1, grad_reg_2, rel_grad_reg

def descent(theta_1, theta_2, theta_relation, grad_reg_1, grad_reg_2,
        rel_grad_reg, num_lay_1, num_lay_2, alpha):
    '''Change matrices of weights according to the gradient.'''
    theta_1_temp = range(num_lay_1)
    for i in range(num_lay_1):
        theta_1_temp[i] = theta_1[i] - alpha*grad_reg_1[i] # Change weights in the first subnetwork
    theta_relation_temp = theta_relation - alpha*rel_grad_reg # Change relation weights
    theta_2_temp = range(num_lay_2)
    for i in range(num_lay_2):
        theta_2_temp[i] = theta_2[i] - alpha*grad_reg_2[i] # Change weights in the second subnetwork
    return theta_1_temp, theta_2_temp, theta_relation_temp



def gradient_check(theta, theta_1, theta_2, theta_relation):
    numgrad = numpy.zeros((numpy.shape(theta_1[1])))
    perturb = numpy.zeros((numpy.shape(theta_1[1])))
    e = 1e-4
    th_ch_minus = numpy.copy(theta_1)
    th_ch_plus = numpy.copy(theta_1)
    for p in range(numpy.size(theta_1[1])):
        perturb.flat[p] = e
        th_ch_minus[1] = theta_1[1] - perturb
        th_ch_plus[1] = theta_1[1] + perturb
        [a_chm_1, a_chm_2] = forward_propagation(m, num_lay_1, num_lay_2,
                  X, input_relation, th_ch_minus, theta_2, theta_relation)
        j_ch_minus = compute_cost_function(a_chm_2, th_ch_minus, theta_2, theta_relation,
        num_lay_1, num_lay_2, R, Y)
        [a_chp_1, a_chp_2] = forward_propagation(m, num_lay_1, num_lay_2,
                  X, input_relation, th_ch_plus, theta_2, theta_relation)
        j_ch_plus = compute_cost_function(a_chp_2, th_ch_plus, theta_2, theta_relation,
        num_lay_1, num_lay_2, R, Y)
        numgrad.flat[p] = (j_ch_plus - j_ch_minus) / (2*e)
        perturb.flat[p] = 0
# print numgrad

def gradient_check(theta, theta_1, theta_2, theta_relation):
    numgrad = numpy.zeros((numpy.shape(theta_1[0])))
    perturb = numpy.zeros((numpy.shape(theta_1[0])))
    e = 1e-4
    th_ch_minus = numpy.copy(theta_1)
    th_ch_plus = numpy.copy(theta_1)
    for p in range(numpy.size(theta_1[0])):
        perturb.flat[p] = e
        th_ch_minus[0] = theta_1[0] - perturb
        th_ch_plus[0] = theta_1[0] + perturb
        [a_chm_1, a_chm_2] = forward_propagation(m, num_lay_1, num_lay_2,
                  X, input_relation, th_ch_minus, theta_2, theta_relation)
        j_ch_minus = compute_cost_function(a_chm_2, th_ch_minus, theta_2, theta_relation,
        num_lay_1, num_lay_2, R, Y)
        [a_chp_1, a_chp_2] = forward_propagation(m, num_lay_1, num_lay_2,
                  X, input_relation, th_ch_plus, theta_2, theta_relation)
        j_ch_plus = compute_cost_function(a_chp_2, th_ch_plus, theta_2, theta_relation,
        num_lay_1, num_lay_2, R, Y)
        numgrad.flat[p] = (j_ch_plus - j_ch_minus) / (2*e)
        perturb.flat[p] = 0
# print numgrad



def gradient_check(theta, theta_1, theta_2, theta_relation):
    numgrad_simple = numpy.zeros((numpy.shape(theta_2[0])))
    perturb = numpy.zeros((numpy.shape(theta_2[0])))
    e = 2
    th_ch_minus = numpy.copy(theta_2)
    for p in range(numpy.size(theta_2[0])):
        perturb.flat[p] = e
        th_ch_minus[0] = theta_2[0] - perturb
        [a_chm_1, a_chm_2] = forward_propagation(m, num_lay_1, num_lay_2, X, input_relation,
        theta_1, th_ch_minus, theta_relation)
        j_ch_minus = compute_cost_function(a_chm_2, theta_1, th_ch_minus, theta_relation,
        num_lay_1, num_lay_2, R, Y)
        numgrad_simple.flat[p] = (o - j_ch_minus) / e
        perturb.flat[p] = 0
# print numgrad_simple

