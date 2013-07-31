#-------------------------------------------------------------------------------
# Name:        NN_alnalylis
# Purpose:     The main body of the program.
#
# Author:      Ilya Pershin
#
# Created:     14.05.2013
# Copyright:   (c) 11 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
"""
Common:
  +  -Make large-set version(?)
  +  -Saving parameters
    -Prepare working version and readme
##  +  -GitHub
    -rewrite SNN as class

Optimisation:
    -SA: use simpler SNN-function(prepared), only training loops
    -change square for multiplication
    -PyPy, Psyco
    -Check
    -CUDA
    -Numexpr for multiprocessing perfomance


Questions:
    -Small training set/ artificial data
    -why dir changes
"""

import os
import numpy as np
import csv
import matplotlib.pyplot as pp
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

os.chdir('C:/SNN/complex_ver')
import neural_network
import NN_learning
os.chdir('C:/SNN/complex_ver')

#np.random.seed(111)

# Parameters:
hidden_1 = [7]  # Structure of the first subnetwork
hidden_2 = [7]  # Structure of the second subnetwork
epsilon = 0.1  # Limitation of  initial weights
alpha = 1  # Learning rate
S = 2   # Slope of the sigmoid function
R = 0.0  # Coefficient of regularization
M = 0.8  # Momentum
e = 1e-4  # value of weights changing in the gradient check function
epochs_count = 500
batches_count = 1
data_proportion = 0.3
online_learning = '' # Set 'on' to turn on online learing (one example per iteration)
data_representation = 'separate'  # Representation of lerning data, 'complex' or 'separate'
file = 'ilashevskaya.csv'

start = timer()
[J, J_test, theta_history, time_ext, time_int] = NN_learning.SNN(hidden_1, hidden_2,
                                                 epsilon, alpha, S, R, M, e, epochs_count,
                                                 batches_count, data_proportion, online_learning,
                                                 data_representation, file)
print timer() - start

# Visualization
NN_learning.disp_learning_dynamic(J, J_test)
print min(J), min(J_test)

# Check result of learning
example = 10
epoch = 199
neural_network.check_result(example, epoch, file, hidden_1, hidden_2, theta_history, S)

# NET STRUCTURE ANALYSIS

hidden_1_max = 20
hidden_2_max = 20
num_init = 10      # number of random initializations
overfitting = '' # set 'on' to consider overfitting effect

start = timer()

J_SA = NN_learning.Structure_Analysis(hidden_1_max, hidden_2_max, num_init, overfitting,
                                      hidden_1, hidden_2, epsilon, alpha, S, R, M, e, epochs_count,
                                      batches_count, data_proportion, online_learning, file)
print timer() - start
# Saving
np.savetxt('SA/StructAn[h1max='+str(hidden_1_max)+', '+ \
                             'h2max='+str(hidden_2_max)+', '+ \
                             'Ninit='+str(num_init)+']_'+file, J_SA, delimiter=',')

# Load SA-results
SA_file = 'SA/StructAn[h1max=2, h2max=3, Ninit=2]_ilashevskaya.csv'
loaded_J_SA = NN_learning.Load_SA_results(SA_file)

# Visualization
NN_learning.disp_struct_analysis(J_SA, hidden_1_max, hidden_2_max)







# Unprepared pieces of code (for the future upgrades)
### One layer dynamic
##hidden_2_max = 30
##J_neur = np.zeros((hidden_2_max, 1))
##for i in range(hidden_2_max):
##    hidden_2 = [i]
##
##    [J, J_test, theta_history] = NN_learning.SNN([20], hidden_2, epsilon, alpha, R, M, e, epochs_count, batches_count,
##                                            data_proportion, online_learning, file)
##    J_neur[i,0] = J_test[-1]
##
##num_iter = range(len(J_neur))
##pp.figure(1)
##pp.plot(num_iter, J_neur)
##pp.xlabel('Number of neurons')
##pp.ylabel('Error')
##pp.title('Hidden layer')
##pp.show()
##
### Dynamic of activations and weight values during learning
act_1 = range(len(theta_history))
act_2 = range(len(theta_history))
for i in range(len(theta_history)):
    iter = i
    [act_1[i], act_2[i]] = NN_learning.hidden_activation(4, 4, len(hidden_1), len(hidden_2), theta_history, iter, S)

for i in xrange(len(theta_history)):
    print  np.mean(theta_history[i][1][0])

for i in xrange(len(theta_history)):
    print np.mean(act_2[i][0][0][0])
##
##
##
### PCA dynamics
### Perform learning
##[J, J_test, theta_history] = NN_learning.SNN(hidden_1, hidden_2, epsilon, alpha,
##        R, M, e, epochs_count, batches_count, data_proportion, online_learning, file)
##
### Drowing hidden layer activations for every iteration
##
##item_num = 2
##rel_num = 2
##activation = NN_learning.hidden_activation(item_num, rel_num, len(hidden_1), len(hidden_2), theta_history)
##
##
### Euclidean distance
##for i in xrange(item_num):
##    for j in xrange(rel_num):
##
##dist_a2 = (activation[0][0]  - activation[1][0])**2
##dist_a22 = (activation[0][1]  - activation[1][1])**2
##iteration_number = range(len(dist_a2))
##
##dist_a2_red = NN_learning.PCA(dist_a2, 2)
##dist_a22_red = NN_learning.PCA(dist_a22, 2)
##pp.figure(1)
##pp.plot(iteration_number, dist_a2)
##pp.show()
##
##
### Appliing PCA to reduce dimensions
##activation_red = range(item_num)
##for i in xrange(item_num):
##    activation_red[i] = range(rel_num)
##for i in xrange(item_num):
##    for j in xrange(rel_num):
##        activation_red[i][j] = NN_learning.PCA(activation[i][j], 2)
##
### PCA visualisation
##num_iter = range(len(J))
##fig, ax = pp.subplots()
##ax.plot(dist_a2_red[:,0], dist_a2_red[:,1], lw=1, ls='-', marker='o', markersize=4) # half-transparant red
##ax.plot(dist_a22_red[:,0], dist_a22_red[:,1], lw=1, ls='-', marker='x', markersize=4)      # RGB hex code for a bluish color
##ax.plot(activation_red[1][0][:,0][0], activation_red[1][0][:,1][0], lw=1, ls='-', marker='o', markersize=4)
##ax.plot(activation_red[1][1][:,0][0], activation_red[1][1][:,1][0], lw=1, ls='-', marker='o', markersize=4)
##
##pp.show()
