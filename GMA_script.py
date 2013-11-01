#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      11
#
# Created:     26.10.2013
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
os.chdir('c:\SNN\SA')

import neural_network
import NN_learning

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
number_of_epochs = 2
number_of_batches = 1
data_proportion = 0.0
online_learning = 'on' # Set 'on' to turn on online learing (one example per iteration)
data_representation = 'large'  # Representation of lerning data, 'complex' or 'separate'
cost_function = 'mean_squares'

hidden_1_range = [7, 7]
hidden_2_range = [12, 12]
num_init = 10    # number of random initializations

file_name = 'C:/SNN/SA/Learn_data'

# Save parameters for genral model learning
cfg = dict(epsilon=epsilon, alpha=alpha, S=S, R=R, M=M, number_of_epochs=number_of_epochs,
           number_of_batches=number_of_batches, data_proportion=data_proportion,
           online_learning=online_learning, cost_function=cost_function,
           file_name=file_name, hidden_1_range=hidden_1_range,
           hidden_2_range=hidden_2_range, num_init=num_init, )

NN_learning.save_cfg(cfg, 'GeneralModel', True)

# Build general model

# Prepare data from given file
[item, rel, attr, batch_size,
 number_of_batches, training_ex_idx,
 test_item_set, test_rel_set, test_attr_set] = NN_learning.Prepare_Learning(number_of_epochs,
                                                                            number_of_batches, data_proportion,
                                                                            online_learning, data_representation, file_name)
# Perform analisys with test set
[SA, average_theta, best_iter] = NN_learning.SA(alpha, R, S, M, epsilon, batch_size, item, rel, attr, data_representation,
                                                data_proportion, cost_function, number_of_epochs, number_of_batches, training_ex_idx,
                                                test_item_set, test_rel_set, test_attr_set, hidden_1_range, hidden_2_range, num_init)

# create dir to store results
if not os.path.exists('GMA_results'):
    os.mkdir('GMA_results')
# save GM theta matrix
f = open('GMA_results/GM_theta.pkl', 'wb')
pickle.dump(average_theta, f)
f.close()


# Perform specification for every probationer:

# set new parameters ...
data_representation = 'complex'
hidden_1 = [hidden_1_range[0]]
hidden_2 = [hidden_2_range[0]]

cfg = dict(epsilon=epsilon, alpha=alpha, S=S, R=R, M=M, number_of_epochs=number_of_epochs,
           number_of_batches=number_of_batches, data_proportion=data_proportion,
           online_learning=online_learning, cost_function=cost_function,
           file_name=file_name, hidden_1=hidden_1,
           hidden_2=hidden_2)

NN_learning.save_cfg(cfg, file_name, False)

theta_list = []  # list to store theta matrices for every probationer
for prob in os.listdir(file_name):
    # Prepare data from given file
    [item, rel, attr, batch_size,
     number_of_batches, training_ex_idx,
     test_item_set, test_rel_set, test_attr_set] = NN_learning.Prepare_Learning(number_of_epochs,
                                                                                number_of_batches, data_proportion,
                                                                                online_learning, data_representation, file_name+'/'+prob)

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
    [theta_1, theta_2, theta_relation] = [average_theta[0][0][0], average_theta[0][0][1], average_theta[0][0][2]]

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
            if data_representation == 'complex':
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

    # add learned weights in a list
    theta_list.append([theta_1, theta_2, theta_relation])

    print prob

    # save theta matrices
    f = open(os.getcwd()+'/GMA_results/'+prob+'_theta.pkl', 'wb')
    pickle.dump([theta_1, theta_2, theta_relation], f)
    f.close()

wVar_list = []  # list to store weights variance
wPairVar_list = []  # list to store weights variances computed over all pair combinations
actVar_list = []  # list to store actiavtion variance
for theta in theta_list:
    actVar_list.append(neural_network.actVar(S, hidden_1, hidden_2, file_name+'/01.csv',
                              theta[0], theta[1], theta[2]))
    wVar_sublist = []
    wPairVar_sublist = []
    # unpack all theta matrices in one list
    thetaSet = []
    if len(hidden_1) > 1:
        for t in xrange(theta[0]): thetaSet.append(t)
    else: thetaSet.append(theta[0])
    for t in theta[1]:
        thetaSet.append(t)
    thetaSet.append(theta[2])

    for th in thetaSet:
        wVar_sublist.append(neural_network.wVar(th))
        wPairVar_sublist.append(neural_network.wPairVAr(th))
    wVar_list.append(wVar_sublist)
    wPairVar_list.append(wPairVar_sublist)



num_prob = len(os.listdir(os.getcwd()+'/Learn_data'))
table = range(num_prob)

# fill the table
for r in xrange(num_prob):
    table[r] = []
    for v in wVar_list[r]: table[r].append(v)
    for v in wPairVar_list[r]: table[r].append(v)
    for v in actVar_list[r]: table[r].append(v)

with open(os.getcwd()+'/GMA_results/result_tableee.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(table)






