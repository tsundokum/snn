#-------------------------------------------------------------------------------
# Name:        Neural Network Learning
# Purpose:
#
# Author:      11
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
input_size = 6 # Item number
hidden_1 = [5, 4] # Struscture of the first subnetwork
relation_in_size = 2 # Relations number
hidden_2 = [3, 2] # Structure of the second subnetwork
output_size = 4 # Number of properties
epsilon = 0.5 # Limitation of  initial eights
alpha = 0.1 # Learning rate
R = 0.3 # Coefficient of regularization
number_of_epochs = 3
number_of_batches = 8



# BEGINNING


# Create test data set
X = numpy.arange(18).reshape(3,6)
input_relation = numpy.arange(6).reshape(3, 2)
Y = numpy.ones((m * output_size))
for l in range(m * output_size):
    Y[l] = numpy.random.choice([0,1])
Y = Y.reshape(m, output_size)


# Usefull veriables:
m = len(X) # Batch size
num_lay_1 = len(hidden_1) # Number of layers in the first subnetwork
num_lay_2 = len(hidden_2) # Number of layers in the second subnetwork


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
        [a_1, a_2] = neural_network.forward_propagation(m, num_lay_1, num_lay_2, X,
                input_relation, theta_1, theta_2, theta_relation)

        J = neural_network.compute_cost_function(m, a_2, theta_1, theta_2, theta_relation,
        num_lay_1, num_lay_2, R, Y)

        [grad_reg_1, grad_reg_2, rel_grad_reg] = neural_network.back_propagation(m, a_1, a_2, input_relation, theta_1, theta_2, theta_relation,
        num_lay_1, num_lay_2, R, Y)

[grad_reg_1, grad_reg_2, rel_grad_reg] = neural_network.back_propagation(a_1, a_2, theta_1, theta_2, theta_relation,
        num_lay_1, num_lay_2, R, Y, m)

        [theta_1_temp, theta_2_temp, theta_relation_temp] = neural_network.descent(theta_1,
        theta_2, theta_relation, grad_reg_1, grad_reg_2, rel_grad_reg,
        num_lay_1, num_lay_2, alpha)

        theta_1 = theta_1_temp
        theta_relation = theta_relation_temp
        theta_2 = theta_2_temp






