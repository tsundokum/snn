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
os.chdir('C:/')
import numpy
import neural_network


# Parameters:
input_size = 3  # Item number
hidden_1 = [3, 2]  # Struscture of the first subnetwork
relation_in_size = 3  # Relations number
hidden_2 = [3, 2]  # Structure of the second subnetwork
output_size = 3  # Number of properties
epsilon = 0.5  # Limitation of  initial eights
alpha = 0.1  # Learning rate
R = 0.0  # Coefficient of regularization
e = 1e-4  # value of weights cahanging in the gradien check function
number_of_epochs = 3
number_of_batches = 8

# Optional:
m = 3  # number of training examples



# BEGINNING


# Create test data set
X = numpy.arange(input_size * m).reshape(m, input_size)
input_relation = numpy.arange(relation_in_size * m).reshape(m, relation_in_size)
Y = numpy.ones((m * output_size))
for l in range(m * output_size):
    Y[l] = numpy.random.choice([0,1])
Y = Y.reshape(m, output_size)


# Usefull veriables:
m = len(X) # Batch size
num_lay_1 = len(hidden_1) # Number of layers in the first subnetwork
num_lay_2 = len(hidden_2) # Number of layers in the second subnetwork
J = numpy.arange(number_of_epochs * number_of_batches, dtype=float). \
        reshape(number_of_epochs, number_of_batches)  # this list will contain log of errors

# Data division (optional):
# ...

# Separate validation set:
#     training_data = ...
#     validation_data = ...

# Divide training data by number of batches
#     X = ...
#     Y = ...

# Data preprocessing:
# ...


#  Create 3 sets of matrices of initial weights according to the given structure.
[theta_1, theta_2, theta_relation] = neural_network.initialise_weights(input_size, hidden_1,
        hidden_2, relation_in_size, output_size, num_lay_1, num_lay_2, epsilon)

for epoch in range(number_of_epochs): # Beginning of epoch loop

    for batch in range(number_of_batches): # Beginning of batch loop

        # Compute activations of every unit in the ntework.
        [a_1, a_2] = neural_network.forward_propagation(m, num_lay_1, num_lay_2,
                X, input_relation, theta_1, theta_2, theta_relation)

        # Compute average error with regularization.
        J[epoch, batch] = neural_network.compute_cost_function(m, a_2, theta_1, theta_2,
                theta_relation, num_lay_1, num_lay_2, R, Y)

        # Compute derivative of the cost function with respect to matrices theta.
        [grad_reg_1, grad_reg_2, rel_grad_reg] = neural_network.back_propagation(m,
                a_1, a_2, input_relation, theta_1, theta_2, theta_relation,
                num_lay_1, num_lay_2, R, Y)

        # Computes the numerical gradient of the function around theta for every weight (optional).
        '''[numgrad_1, numgrad_2, numgrad_rel] = neural_network.gradient_check(e,
                m, X, Y, input_relation, theta_1, theta_2, theta_relation,
                num_lay_1, num_lay_2, R)'''

        # Change matrices of weights according to the gradient.
        [theta_1_temp, theta_2_temp, theta_relation_temp] = \
                neural_network.descent(theta_1, theta_2, theta_relation,
                grad_reg_1, grad_reg_2, rel_grad_reg, num_lay_1, num_lay_2, alpha)

        # Update current weight matrices
        theta_1 = theta_1_temp
        theta_relation = theta_relation_temp
        theta_2 = theta_2_temp



