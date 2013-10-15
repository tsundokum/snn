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
import pickle
import os

##np.random.seed(111)

# Parameters:
##hidden_1 = [3]  # Structure of the first subnetwork
##hidden_2 = [4]  # Structure of the second subnetwork
epsilon = 0.5  # Limitation of  initial weights
alpha = 0.3  # Learning rate
S = 3   # Slope of the sigmoid function
R = 0.0  # Coefficient of regularization
M = 0.0  # Momentum
e = 1e-4  # value of weights changing in the gradient check function
number_of_epochs = 600
number_of_batches = 1
data_proportion = 0.25
online_learning = 'on' # Set 'on' to turn on online learing (one example per iteration)
data_representation = 'complex'  # Representation of lerning data, 'complex' or 'separate'
cost_function = 'mean_squares'

# NET STRUCTURE ANALYSIS
hidden_1_range = [3, 7]
hidden_2_range = [3, 12]
num_init = 8    # number of random initializations




import neural_network
import NN_learning


for i in range(14):
    file = 'Learn_data/'+str(i+1)+'.csv'
    file_name = 'full_SA/'+str(i+1)
    data_proportion = 0.25
    print str(i+1)+'  test set size:'+str(data_proportion)
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

    print str(i+1)+'  test set size:'+str(data_proportion)
    # Perform analisys with training set only
    data_proportion = 0
    [SA, average_theta, best_iter] = NN_learning.SA(alpha, R, S, M, epsilon, batch_size, item, rel, attr, data_representation,
                                                    data_proportion, cost_function, number_of_epochs, number_of_batches, training_ex_idx,
                                                    test_item_set, test_rel_set, test_attr_set, hidden_1_range, hidden_2_range, num_init)
    # save results
    NN_learning.save_SA(file_name, 'training', best_iter, SA, average_theta)


