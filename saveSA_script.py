#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      11
#
# Created:     14.10.2013
# Copyright:   (c) 11 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
import pickle
import numpy as np
import csv

def fill_table(stDir, thDir, num_prob, T):
    table = range(num_prob)
    for l in range(num_prob):
        table[l] = []

    for i in xrange(num_prob):
        # Load surface
        surf = os.listdir(stDir)[i]
        surf_file = stDir +'/'+ surf
        loaded_SA = open(surf_file, 'r')
        SA_surface = pickle.load(loaded_SA)
        loaded_SA.close()

        # indeces of the min error
        for q in xrange(np.size(SA_surface, 0)):
            for w in xrange(np.size(SA_surface, 1)):
                if SA_surface[q, w] == np.min(SA_surface):
                    min_idx = [q, w]
    ##    print min_idx
    ##
    ##    # Visualisation
    ##    NN_learning.disp_struct_analysis(SA_surface)
    ##    print np.min(SA_surface)

        # Load theta matrices
        th = os.listdir(thDir)[i]
        th_file = thDir+'/'+th
        loaded_SA = open(th_file, 'r')
        SA_theta = pickle.load(loaded_SA)
        loaded_SA.close()

        # Find theta variance
        theta = SA_theta[min_idx[0]][min_idx[1]] # weight matrices for particular structure
        theta_collect = np.hstack((theta[0].flat, theta[1][0].flat, theta[1][1].flat, theta[2].flat))
        theta_variance = np.var(theta_collect)

        # Fill the table
        table[i].append(i + 1)
        number_of_neurons = (min_idx[0] + 3) + (min_idx[1] + 3)
        table[i].append(number_of_neurons)
        num_repr = min_idx[0] + 3
        table[i].append(num_repr)
        num_hid = min_idx[1] + 3
        table[i].append(num_hid)
        table[i].append(np.min(SA_surface))
        file = os.listdir(stDir)[i]  # Take a file name value
        best_iteration = float(file[file.find('(')+1:file.find(')')])  # Take an iteration
        table[i].append(best_iteration)
        table[i].append(theta_variance)

    with open('table_'+str(T)+'.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(table)


num_prob = 5
# table for the test values
stDir = 'c:/SNN/SA/SA_surface_test'
thDir = 'c:/SNN/SA/SA_theta_test'
fill_table(stDir, thDir, num_prob, 'test')

# table for the train values
stDir = 'c:/SNN/SA/SA_surface_train'
thDir = 'c:/SNN/SA/SA_theta_train'
fill_table(stDir, thDir, num_prob, 'train')

print 'OK!'

