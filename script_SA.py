#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      11
#
# Created:     20.09.2013

# Copyright:   (c) 11 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import csv
##import matplotlib.pyplot as pp
##import matplotlib as mpl
##from matplotlib import cm
##from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
import os
import pickle

os.chdir('c:\SNN\protoGUI')

import neural_network
import NN_learning

##np.random.seed(111)

# Parameters:
##hidden_1 = [3]  # Structure of the first subnetwork
##hidden_2 = [4]  # Structure of the second subnetwork
epsilon = 0.5  # Limitation of  initial weights
alpha = 1.5  # Learning rate
S = 1.5   # Slope of the sigmoid function
R = 0.0  # Coefficient of regularization
M = 0.0  # Momentum
e = 1e-4  # value of weights changing in the gradient check function
number_of_epochs = 2
number_of_batches = 1
data_proportion = 0.25
online_learning = '' # Set 'on' to turn on online learing (one example per iteration)
data_representation = 'separate'  # Representation of lerning data, 'complex' or 'separate'
cost_function = 'least_squares'  # could be 'lestt squares or 'cross entropy'
file = 'Learn_data/03.csv'


# NET STRUCTURE ANALYSIS
hidden_1_range = [2, 5]
hidden_2_range = [4, 10]
num_init = 7    # number of random initializations
file_name = 'full_SA/03'  # name to save results

# Prepare data from given file
[item, rel, attr, batch_size,
 number_of_batches, training_ex_idx,
 test_item_set, test_rel_set, test_attr_set] = NN_learning.Prepare_Learning(epsilon, number_of_epochs,
                                                                            number_of_batches, data_proportion,
                                                                            online_learning, data_representation, file)
# Perform analisys with test set
[SA, average_theta, best_iter] = NN_learning.SA(alpha, R, S, M, epsilon, batch_size, item, rel, attr, data_representation,
                                                data_proportion, cost_function, number_of_epochs, number_of_batches, training_ex_idx,
                                                test_item_set, test_rel_set, test_attr_set, hidden_1_range, hidden_2_range, num_init)

if not os.path.exists('full_SA'):
    os.mkdir('full_SA')

# save results
NN_learning.save_SA(file_name, 'test', best_iter, SA, average_theta)


# Perform analisys with training set only
data_proportion = 0
[SA, average_theta, best_iter] = NN_learning.SA(alpha, R, S, M, epsilon, batch_size, item, rel, attr, data_representation,
                                                data_proportion, cost_function, number_of_epochs, number_of_batches, training_ex_idx,
                                                test_item_set, test_rel_set, test_attr_set, hidden_1_range, hidden_2_range, num_init)
# save results
NN_learning.save_SA(file_name, 'training', best_iter, SA, average_theta)


### Load results
##SA_file = ''
##loaded_SA = open('01_SA_surf_(750.142857143)_test.pkl', 'r')
##J_SA = pickle.load(loaded_SA)
##
### Visualisation
##NN_learning.disp_struct_analysis(J_SA)
##
### indeces of the min error
##for row in xrange(np.size(J_SA, 0)):
##    for column in xrange(np.size(J_SA, 1)):
##        if J_SA[row, column] == np.min(J_SA):
##            break
##min_idx = [row, column]