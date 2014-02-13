#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        Neural Network Analysis
# Purpose:     Contains complex functions to analyse learning of Semantic
#              neural network.
#
# Author:      Ilia Pershin
#
# Created:     14.03.2013
# Copyright:   (c) 11 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
"""
Contains complex functions to analyse learning of Semantic neural network
...
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as pp
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
import pickle
import neural_network
import NN_learning


# STRUCTURE ANALYSIS

def save_SA(file_name, test, best_iter, SA, average_theta):
    # error surface
    f = open(file_name+'_SA_surf'+'_('+str(best_iter)+')'+'_'+test+'.pkl', 'wb')
    pickle.dump(SA, f)
    f.close()
    # theta matrices
    f = open(file_name+'_SA_theta'+'_('+str(best_iter)+')'+'_'+test+'.pkl', 'wb')
    pickle.dump(average_theta, f)
    f.close()


def StrAn(alpha, R, S, M, epsilon, batch_size, number_of_epochs, number_of_batches,
       data_representation, data_proportion, cost_function, exact_error_eval,
       train_set, test_set, hidden_1_range, hidden_2_range, num_init, self):
    """
    Computes efficiency of network with respect to number of neurons in every layer
    (Only for one-layer subnetworks).
    Takes range of number of neurons in every layer and number of random weight initializations.
    For every net structure learn network several times(num_init) with different random initial weights.
    Every time it takes only one value of test error (minimum).
    And then computes average error for particular net structure over all initializations.
    Doesn't take in consideration overfiting effect (takes only minimum error values, not last).
    If there is the a sample, takes values only from test sample.
        ----
        SA - matrix of errors accrording to the structure
        average_iter - best iteration averaged and rounded over trials
        average_theta - theta matrices for the best iteration averaged and rounded
    """

    # Prepare arrays to fill with error values
    hidden_1_max = (hidden_1_range[1] + 1) - hidden_1_range[0]  # maximum range of neurons in every subnet
    hidden_2_max = (hidden_2_range[1] + 1) - hidden_2_range[0]
    SA = np.zeros((hidden_1_max, hidden_2_max))
    average_theta = range(hidden_1_max)
    for i in xrange(len(average_theta)):
        average_theta[i] = range(hidden_2_max)
        for j in xrange(len(average_theta[i])):
            average_theta[i][j] =[0] * 3
            average_theta[i][j][1] = [0,0]
    performance_time = 0

    for i in range(hidden_2_range[0], hidden_2_range[1] + 1):      # Loop over the hidden layer
        hidden_2 = [i]                 # Set number of neurons in the second layer(hidden)

        for j in range(hidden_1_range[0], hidden_1_range[1] + 1):  # Loop over  the representaton layer
            start = timer()
            hidden_1 = [j]             # Set number of neurons in the first layer(representation)
            # epmpty lists for random init-s errors
            init_error = []
            theta_accumulator = [0] * 3
            theta_accumulator[1] = [0,0]
            Iterations = []
            # Loop over the random initializations
            for init in range(num_init):
                [J, J_test,
                 theta_history] = NN_learning.Learning(alpha, R, S, M,
                                                       hidden_1, hidden_2,
                                                       epsilon, batch_size,
                                                       data_representation,
                                                       data_proportion,
                                                       cost_function,
                                                       number_of_epochs,
                                                       number_of_batches,
                                                       train_set, test_set,
                                                       exact_error_eval)
                if data_proportion != 0:
                    best_iter = np.argmin(J_test)
                    init_error.append(np.min(J_test))
                    Iterations.append(best_iter)
                    theta_accumulator[0] += theta_history[best_iter][0]
                    theta_accumulator[1][0] += theta_history[best_iter][1][0]
                    theta_accumulator[1][1] += theta_history[best_iter][1][1]
                    theta_accumulator[2] += theta_history[best_iter][2]
                else:
                    init_error.append(J[-1])
                    theta_accumulator[0] += theta_history[-1][0]
                    theta_accumulator[1][0] += theta_history[-1][1][0]
                    theta_accumulator[1][1] += theta_history[-1][1][1]
                    theta_accumulator[2] += theta_history[-1][2]

            r_idx = hidden_1[0] - hidden_1_range[0]
            c_idx = hidden_2[0] - hidden_2_range[0]
            # average error value
            SA[r_idx, c_idx] = np.average(init_error)  # fill the surface of errors
            # average theta matrices
            average_theta[r_idx][c_idx][0] = theta_accumulator[0] / num_init
            average_theta[r_idx][c_idx][1][0] = theta_accumulator[1][0] / num_init
            average_theta[r_idx][c_idx][1][1] = theta_accumulator[1][1] / num_init
            average_theta[r_idx][c_idx][2] = theta_accumulator[2] / num_init
            time_per_loop = timer() - start
            # average number of the best iteration
            if data_proportion != 0:
                average_iter = round(np.average(Iterations))
            else:
                average_iter = number_of_epochs * number_of_batches

            # Time
            full_time = time_per_loop * (hidden_1_max * hidden_2_max)
            performance_time += time_per_loop
            remaining_time = full_time - performance_time
            hours = int(remaining_time / 3600)
            minutes = int((remaining_time - (hours * 3600)) / 60)
            seconds = remaining_time - (hours*3600) - (minutes*60)
            # Show progress
            if self:
                self.txtRemainingTime.SetValue('R: '+str(hidden_1[0])+'/'+str(hidden_1_range[1])+', '+
                                               'H: '+str(hidden_2[0])+'/'+str(hidden_2_range[1])+'; '+
                                               str(hours)+' h, '+
                                               str(minutes)+' min, '+
                                               str(round(seconds, 2))+' sec')

    return SA, average_iter, average_theta


def prep_SA(hidden_1_range, hidden_2_range, num_init, epsilon, alpha, S, R, M,
            number_of_epochs, number_of_batches, data_proportion, online_learning,
            data_representation, cost_function, exact_error_eval, file_name, self):
    """
    Struct analysis inncluding data preparation.
    """
    # prepare data form file
    [batch_size,
     number_of_batches,
     train_set,
     test_set] = NN_learning.Prepare_Learning(number_of_epochs, number_of_batches,
                                              data_proportion, online_learning,
                                              data_representation, file_name)
    # perform analysis
    [SA, best_iter,
     average_theta] = StrAn(alpha, R, S, M, epsilon, batch_size,
                                     number_of_epochs, number_of_batches,
                                     data_representation, data_proportion,
                                     cost_function, exact_error_eval,
                                     train_set, test_set, hidden_1_range,
                                     hidden_2_range, num_init, self)

    return SA, best_iter, average_theta


def full_SA(hidden_1_range, hidden_2_range, num_init, epsilon, alpha, S, R, M,
            number_of_epochs, number_of_batches, data_proportion,
            online_learning, data_representation, cost_function,
            exact_error_eval, file_dir, out_dir, train_eval, self):
    """
    Full function for the structure analysis. Can analyse all files in given
    directiory. Both with test and train-only modes.
    "f" - can be a path to a file or a path to a folder either
    "train_eval" - if True, performs additional analysis without test sample
    """
    if (data_proportion == 0) and (train_eval == True):  # exaption for wrong parameters
        train_eval = False
    if os.path.isfile(file_dir):  # if file given
        out_file = out_dir+f[f.rfind('\\'):][:-4]
        if train_eval == True:  # in case of additional train-only evaluation
            for d in [0, data_proportion]:  # preform SA without test and with test il loop
                data_proportion = d
                if self:
                    self.txtCurFile.SetValue(f[f.rfind('\\'):]+'  - test= '+str(data_proportion))
                [SA, best_iter,
                 average_theta] = prep_SA(hidden_1_range, hidden_2_range,
                                          num_init, epsilon, alpha, S, R, M,
                                          number_of_epochs, number_of_batches,
                                          data_proportion, online_learning,
                                          data_representation, cost_function,
                                          exact_error_eval, f, self)
                # save results
                save_SA(out_file, 'test='+str(data_proportion), best_iter, SA, average_theta)
        else:
            if self:
                self.txtCurFile.SetValue(f[f.rfind('\\'):]+'  - test= '+str(data_proportion))
            [SA, best_iter,
             average_theta] = prep_SA(hidden_1_range, hidden_2_range,
                                      num_init, epsilon, alpha, S, R, M,
                                      number_of_epochs, number_of_batches,
                                      data_proportion, online_learning,
                                      data_representation, cost_function,
                                      exact_error_eval, f, self)
            # save results
            save_SA(out_file, 'test='+str(data_proportion), best_iter, SA, average_theta)
    # if given directory
    else:
        for f in os.listdir(file_dir):  # loop over files in directory
            out_file = out_dir+'\\'+f[:-4]
            if train_eval == True:
                for d in [0, data_proportion]:  # preform SA without test and with test il loop
                    data_proportion = d
                    if self:
                        self.txtCurFile.SetValue(f+'  - test= '+str(data_proportion))
                    [SA, best_iter,
                     average_theta] = prep_SA(hidden_1_range, hidden_2_range,
                                              num_init, epsilon, alpha, S, R, M,
                                              number_of_epochs, number_of_batches,
                                              data_proportion, online_learning,
                                              data_representation, cost_function,
                                              exact_error_eval, file_dir+'\\'+f,
                                              self)
                    # save results
                    save_SA(out_file, 'test='+str(data_proportion), best_iter, SA, average_theta)
            else:
                if self:
                    self.txtCurFile.SetValue(f+'  - test= '+str(data_proportion))
                [SA, best_iter,
                 average_theta] = prep_SA(hidden_1_range, hidden_2_range,
                                          num_init, epsilon, alpha, S, R, M,
                                          number_of_epochs, number_of_batches,
                                          data_proportion, online_learning,
                                          data_representation, cost_function,
                                          exact_error_eval, file_dir+'\\'+f, self)
                # save results
                save_SA(out_file, 'test='+str(data_proportion), best_iter, SA, average_theta)


def fill_table_SA(dir):
    """
    Function to fill the final data in the table.
    Takes path to the directory written as string.
    dir may consist train, test or both data.
    """
    # separate files in the categories
    files = os.listdir(dir)
    (surf_train, surf_test, theta_train, theta_test) = ([], [], [], [])
    for f in files:
        if 'SA' in f:
            if f[-5] == '0':
                surf_train.append(f)
            else:
                surf_test.append(f)
        else:
            if f[-5] == '0':
                theta_train.append(f)
            else:
                theta_test.append(f)
    train = (surf_train, theta_train)
    test = (surf_test, theta_test)

    for t in (train, test):
        if t[0] != 0:
            num_prob = len(t[0])
            table = range(num_prob)
            for l in range(num_prob):
                table[l] = []
            # loop over probationers
            for i in xrange(num_prob):
                # Load surface
                surf_file = dir+'\\'+t[0][i]
                loaded_SA = open(surf_file, 'r')
                SA_surface = pickle.load(loaded_SA)
                loaded_SA.close()
                # indeces of the min error
                for q in xrange(np.size(SA_surface, 0)):
                    for w in xrange(np.size(SA_surface, 1)):
                        if SA_surface[q, w] == np.min(SA_surface):
                            min_idx = [q, w]

                # Load theta matrices
                th_file = dir+'\\'+t[1][i]
                loaded_SA = open(th_file, 'r')
                SA_theta = pickle.load(loaded_SA)
                loaded_SA.close()

                # Find theta variance
                theta = SA_theta[min_idx[0]][min_idx[1]] # weight matrices for particular structure
                theta_collect = np.hstack((theta[0].flat, theta[1][0].flat, theta[1][1].flat, theta[2].flat))
                theta_variance = np.var(theta_collect)

                # Fill the table
                prob_name = t[0][i][:t[0][i].find('_')]
                table[i].append(prob_name)
                number_of_neurons = (min_idx[0] + 3) + (min_idx[1] + 3)
                table[i].append(number_of_neurons)
                num_repr = min_idx[0] + 3
                table[i].append(num_repr)
                num_hid = min_idx[1] + 3
                table[i].append(num_hid)
                table[i].append(np.min(SA_surface))
                best_iteration = float(surf_file[surf_file.find('(')+1:surf_file.find(')')])  # Take an iteration
                table[i].append(best_iteration)
                table[i].append(theta_variance)

            if t[0][0][-5] == '0':
                T = 'train'
            else:
                T = 'test'
            with open(dir+'\\SA_table_'+T+'.csv', "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(table)


# GENARAL MODEL ANALYSIS

def GMA(epsilon, alpha, S, R, M, number_of_epochs, number_of_batches,
        data_proportion, online_learning, data_representation,
        cost_function, exact_error_eval, hidden_1_range, hidden_2_range,
        num_init, file_dir, out_dir):
    """

    """
    # perform learning of the general model
    [SA, best_iter,
     average_theta] = prep_SA(hidden_1_range, hidden_2_range,
                              num_init,
                              epsilon, alpha, S, R, M,
                              number_of_epochs, number_of_batches,
                              data_proportion, online_learning,
                              'large', cost_function,
                              exact_error_eval, file_dir)
    # save theta matrices
    f = open(out_dir+'\\GMA_average_theta', 'wb')
    pickle.dump(average_theta, f)
    f.close()

    hidden_1 = [hidden_1_range[0]]
    hidden_2 = [hidden_2_range[0]]
    data_representation = 'complex'
    theta_list = []  # list to store theta matrices for every probationer
    for prob in os.listdir(file_dir):
        # prepare data form file
        [batch_size,
         number_of_batches,
         train_set,
         test_set] = NN_learning.Prepare_Learning(number_of_epochs, number_of_batches,
                                                  data_proportion, online_learning,
                                                  'complex', file_dir+'\\'+prob)
        [J, J_test,
         theta_history] = NN_learning.Learning(alpha, R, S, M, hidden_1, hidden_2,
                                   epsilon, batch_size, data_representation,
                                   data_proportion, cost_function,
                                   number_of_epochs, number_of_batches,
                                   train_set, test_set, exact_error_eval)

        theta_list.append(theta_history[-1])  # add learned weights in a list
        del J, J_test, theta_history  # free memory

        # save theta matrices
        f = open(out_dir+'/GMA_'+prob[:-4]+'_theta.pkl', 'wb')
        pickle.dump(theta_list[-1], f)
        f.close()

    return theta_list


def fill_table_GMA(theta_list, example_file, S, hidden_1, hidden_2, out_dir, file_dir):
    """
    """
    wVar_list = []  # list to store weights variance
    wPairVar_list = []  # list to store weights variances computed over all pair combinations
    actVar_list = []  # list to store actiavtion variance
    for theta in theta_list:

        actVar_list.append(neural_network.actVar(S, hidden_1, hidden_2, example_file,
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

    num_prob = len(theta_list)
    table = range(num_prob)

    # fill the table
    for r in xrange(num_prob):
        table[r] = []
        table[r].append(os.listdir(file_dir)[r][:-4])
        for v in wVar_list[r]: table[r].append(v)
        for v in wPairVar_list[r]: table[r].append(v)
        for v in actVar_list[r]: table[r].append(v)

    with open(out_dir+'\\GMA_table.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(table)



