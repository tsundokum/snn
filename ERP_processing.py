#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import csv
import glob
import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model
from sklearn.preprocessing import normalize as norm


def repl_dots(f1, f2):
    # replace '.'
##    if ~f2:
##        f2 =f_
    f1 = open(f1, 'r')
    f2 = open(f2, 'w')
    for line in f1:
        f2.write(line.replace('.', ','))
    f1.close()
    f2.close()

def conc_types(st_types_file):
    """load list of stimulus types"""
    st_types = pd.read_csv(st_types_file, header=None, encoding='cp1251').fillna('')
    st_conc = range(len(st_types))
    for i, r in st_types.iterrows():
        st_conc[i] = [r[0] + ' ' + r[1] + ' ' + r[2], r[3]] # concatenate triads
    return pd.DataFrame(st_conc)

st_types_file = "c:\\SNN\\epr_data_by_10_14\\preprocessing\\st_types.csv"
st_conc = conc_types(st_types_file)

# version for infotable (codes, real names, types of stimuli) using
def recognize_types_info(path_dist, infotable):
    """recognize types of triads"""
    names = []
    types = []
    dists = []
    for dist_file in os.listdir(path_dist):
        if dist_file.endswith('.csv'):
            print dist_file
            distances = pd.read_csv(path_dist + dist_file, header=None, encoding='utf-8').fillna('')
            for i, r in distances.iterrows():
                pos = st_conc[0].str.contains(r[0] + ' ' + r[1] + ' ' + r[2]) # find this triad
                n_type = int(st_conc[1][pos]) - 128
                name = infotable.Name[infotable.Code == dist_file[0:dist_file.find('_')]]
                names.append(name.values[0])
                types.append(n_type)
                dists.append(int(r[3]))
    estimated_triads = pd.DataFrame({'names': pd.Series(names),
                                     'types': np.array(types),
                                     'distances': np.array(dists)},
                                     columns=['names', 'types', 'distances'])
    return estimated_triads

# read info table
infotable = pd.read_csv("c:\\SNN\\epr_data_by_10_14\\preprocessing\\infoTable.csv", sep=',', encoding='cp1251')
path_dist = 'c:\\SNN\\epr_data_by_10_14\\dist_renamed\\'
estimated_triads = recognize_types_info(path_dist, infotable)
estimated_triads.to_csv("c:\\SNN\\epr_data_by_10_14\\preprocessing\\estimated_triads.csv", index=False, encoding='cp1251')
estimated_triads = pd.read_csv("c:\\SNN\\epr_data_by_10_14\\preprocessing\\estimated_triads.csv", encoding='cp1251')

# version for proper file names
def recognize_types(path_dist):
    """recognize types of triads"""
    names = []
    types = []
    dists = []
    for dist_file in os.listdir(path_dist):
        if dist_file.endswith('.csv'):
            print dist_file
        distances = pd.read_csv(path_dist + dist_file, header=None, encoding='utf-8').fillna('')
        for i, r in distances.iterrows():
            pos = st_conc[0].str.contains(r[0] + ' ' + r[1] + ' ' + r[2]) # find this triad
            n_type = int(st_conc[1][pos]) - 128
            names.append(dist_file[0:dist_file.rfind(' ')])
            types.append(n_type)
            dists.append(int(r[3]))
    estimated_triads = pd.DataFrame({'names': pd.Series(names),
                                     'types': np.array(types),
                                     'distances': np.array(dists)},
                                     columns=['names', 'types', 'distances'])
    return estimated_triads

path_dist = 'c:\\SNN\\epr_data_by_10_14\\dist_renamed\\'
estimated_triads = recognize_types(path_dist)
estimated_triads.to_csv("c:\\SNN\\epr_data_by_10_14\\preprocessing\\estimated_triads.csv", index=False, encoding='cp1251')
estimated_triads = pd.read_csv("c:\\SNN\\epr_data_by_10_14\\preprocessing\\estimated_triads.csv", encoding='cp1251')


# read erp data excluding bio and pg channels (1000, 0, 50)
erps = pd.read_csv("c:\\SNN\\epr_data_by_10_14\\preprocessing\\result.csv", \
                   sep=';', decimal=',', encoding='cp1251').ix[:,:-81]

# read erp data excluding bio and pg channels (1100, -100, 50)
erps = pd.read_csv("c:\\SNN\\epr_data_by_10_14\\preprocessing\\result(1100,-100,50).csv", \
                   sep=';', decimal=',', encoding='cp1251').ix[:,:-41]


def exclude_prob(erps, name_num):
    # exclude probationer
    names = erps.Name.unique()
    print 'excluding '+names[name_num]
    erps = erps[~(erps.Name == names[name_num])]
    erps.to_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp.csv", index=False, sep=';')
    repl_dots("c:\\SNN\\ERP_processing\\ERPdata\\erp_avg_excl.csv", \
              "c:\\SNN\\ERP_processing\\ERPdata\\erp_avg_excl(,).csv")
    names = erps.Name.unique()
    for n in names: print n
    return erps

##test1 = pd.read_csv("c:\\SNN\\epr_data_by_10_14\\result1.csv", sep=';', decimal=',')
##test2 = pd.read_csv("c:\\SNN\\epr_data_by_10_14\\result2.csv", sep=';', decimal=',')
####import matplotlib.pyplot as pp
##test_f1 = np.array(test1.ix[0,4:-2], dtype='float')
##test_f2 = np.array(test2.ix[0,4:-2], dtype='float')
##print test_f1.shape, test_f2.shape
##
##disp_learning_dynamic(test_f1, test_f2)
##


nchan = 19
nstep = 22
##def ProcessERP(nchan, nstep, erps, subtr_mean):
##    """..."""
# get unique names and types of stimulus
types = np.sort(erps.TypStim.unique())
names = erps.Name.unique()
#  add real distances (averaging)
erp_processed = pd.DataFrame()
for n in names:
    print n
    erp_n = erps[erps.Name == n]
    dist_by_name = estimated_triads[estimated_triads.names == n]
    for t in types:
        print t
        erp_n_t = erp_n.ix[erp_n.TypStim == t, 4:]
        dist = int(dist_by_name.distances[dist_by_name.types == t])
        processed_n_t = pd.DataFrame()
        # subtract average on every channel
        for ch in range(nchan):
            erp_n_t_ch = erp_n_t.iloc[:,nstep*ch:((ch+1)*nstep)]  # components per cahnnel
            subtrahend = erp_n_t_ch.iloc[:,:2].mean(1)  # take an average of two previous components
            differ = erp_n_t_ch.iloc[:,2:].subtract(subtrahend, axis=0)  # subtraction
            processed_n_t = pd.concat([processed_n_t, differ], axis=1)
        ##       avg_erp = pd.DataFrame(np.mean(erp_n_t)).T  # averaging
##        avg_erp = pd.DataFrame(processed_n_t)  # withount averaging
        avg_erp = processed_n_t  # withount averaging
        avg_erp['Name'] = pd.Series(n, index=avg_erp.index)
        avg_erp['TypStim'] = pd.Series(t, index=avg_erp.index)
        avg_erp['Dist'] = pd.Series(dist, index=avg_erp.index)
        erp_processed = pd.concat((erp_processed, avg_erp))
# replace columns
cols = ['Name'] + ['TypStim'] + ['Dist'] + list(erps.columns[4:])
erp_processed_ord = erp_processed[cols]
erp_processed_ord.to_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_dist.csv", index=False, decimal=',', sep=';')
erps = pd.read_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_dist.csv", decimal='.', sep=';')
repl_dots("c:\\SNN\\ERP_processing\\ERPdata\\erp_dist.csv", "c:\\SNN\\ERP_processing\\ERPdata\\erp_dist(,).csv")

# JUNK
for n in names:
    if sum(erps.Name == n) < 300:
        print n



# WITHOUT AVERAGING
# get unique names and types of stimulus
types = np.sort(erps.TypStim.unique())
names = erps.Name.unique()
#  add real distances (averaging)
erp_processed = pd.DataFrame()
for n in names:
    print n
    erp_n = erps[erps.Name == n]
    dist_by_name = estimated_triads[estimated_triads.names == n]
    for t in types:
        print t
        erp_n_t = erp_n.ix[erp_n.TypStim == t, 4:]
        dist = int(dist_by_name.distances[dist_by_name.types == t])
        ##       avg_erp = pd.DataFrame(np.mean(erp_n_t)).T  # averaging
        avg_erp = pd.DataFrame(erp_n_t)  # withount averaging
        avg_erp['Name'] = pd.Series(n, index=avg_erp.index)
        avg_erp['TypStim'] = pd.Series(t, index=avg_erp.index)
        avg_erp['Dist'] = pd.Series(dist, index=avg_erp.index)
        erp_processed = pd.concat((erp_processed, avg_erp))



# add group and known\unknown type
groups = {u'???????': 1,
          u'?????????': 0,
          u'?????': 1,
          u'???????????': 0,
          u'???????': 0,
          u'??????????': 0,
          u'???????': 1}

erps['group'] = 0  # add new column for group type
names = erps.Name.unique()
for name in names:
    group = groups[name.decode('cp1251')]
    erps['group'][erps['Name'] == name] = group


types = erps['TypStim'].unique()
erps['known'] = 0 # add new column

unknown_first_gr = (erps['group'] == 0) * (erps['TypStim'] >= types[-31])
erps.known[unknown_first_gr] = 0
erps.known[~unknown_first_gr] = 1

unknown_second_gr = (erps['group'] == 1) * (erps['TypStim'] >= 37) * (erps['TypStim'] <= 72)
erps.known[unknown_first_gr] = 0
erps.known[~unknown_first_gr] = 1

erps = erps[list(erps.columns[:3]) + list(erps.columns[-2:]) + list(erps.columns[3:-2])]
erps.to_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_dist_known.csv", index=False, sep=';')
repl_dots("c:\\SNN\\ERP_processing\\ERPdata\\erp_dist_known.csv", "c:\\SNN\\ERP_processing\\ERPdata\\erp_dist_known(,).csv")

erps = pd.read_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_dist_known(,).csv", decimal=',', sep=';')




# take late ERP components
chan_names = erps.columns[3:]
nof_comp = 20  # number of components
num_ch = len(chan_names) / nof_comp
late_comp = []
for n_ch in xrange(num_ch):
    counter = n_ch*nof_comp
    late_comp += list(chan_names)[counter:counter+nof_comp][-6:]
erps = erps[list(erps.columns[:3])+late_comp]  # remove components
erps.to_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_avg_excl_late.csv", index=False, decimal=',', sep=';')
repl_dots("c:\\SNN\\ERP_processing\\ERPdata\\erp_avg_excl_late.csv", "c:\\SNN\\ERP_processing\\ERPdata\\erp_avg_excl_late(,).csv")

# normalize
def normalize(erps, save=):
    """Normalize data for every probationer"""
    chan_names = erps.columns[3:]
    for n in names:
        erp_by_name = erps[erps.Name == n]
        print n
        for ch in chan_names:
            col = erp_by_name[ch]
            m = col.mean()
            sd = col.std()
            erp_by_name[ch] = (col - m) / sd
            print ch
        erps[erps.Name == n] = erp_by_name
        return erps
    if save:
        erps.to_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_norm.csv", index=False, decimal=',', sep=';')
        repl_dots("c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_norm.csv", "c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_norm(,).csv")
##erps = pd.read_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_norm.csv", decimal='.', sep=';')

##def remove_outliers(erps, n_sig):
n_sig = 3
# remove outliers (set NaN)
chan_names = erps.columns[5:]
for n in names:
    erp_by_name = erps[erps.Name == n]
    print n
    for ch in chan_names:
        m = erp_by_name[ch].mean()
        bound = erp_by_name[ch].std() * n_sig
        erp_by_name[ch][np.abs(erp_by_name[ch] - m) >= bound] = np.nan
        print ch
    erps[erps.Name == n] = erp_by_name

erps.to_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_known_rm.csv", index=False, sep=';')
repl_dots("c:\\SNN\\ERP_processing\\ERPdata\\erp_known_rm.csv", "c:\\SNN\\ERP_processing\\ERPdata\\erp_known_rm(,).csv")


# Averaging
types = np.sort(erps.TypStim.unique())
names = erps.Name.unique()
erp_avg = pd.DataFrame()
for n in names:
    print n
    erp_by_name = erps[erps.Name == n]
    for t in types:
       print t
       erp_by_name_type = erp_by_name.ix[erp_by_name.TypStim == t, 2:]
       avg = pd.DataFrame(np.mean(erp_by_name_type)).T
       avg['Name'] = pd.Series(n, index=avg.index)
       avg['TypStim'] = pd.Series(t, index=avg.index)
       erp_avg = pd.concat((erp_avg, avg))
# replace columns
cols = ['Name'] + ['TypStim'] + list(erps.columns[2:])
erps = erp_avg[cols]

erps.to_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_norm_avg.csv", index=False, sep=';')
repl_dots("c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_norm_avg.csv", "c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_norm_avg(,).csv")


# Subtract means per channel
erps = pd.read_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_rm.csv", decimal='.', sep=';')
# remove outliers (set NaN)
chan_names = erps.columns[3:]
nof_comp = 20  # number of components
num_ch = len(chan_names) / nof_comp
for n in names:
    erp_by_name = erps[erps.Name == n]
    print n
    for n_ch in xrange(num_ch):
        print n_ch
        counter = n_ch*nof_comp
        comp_per_chan = list(chan_names)[counter:counter+nof_comp]
        m = erp_by_name[comp_per_chan].mean().mean()
        sub = erp_by_name[comp_per_chan] - m
        erps[erps.Name == n][comp_per_chan] = sub

erps.to_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_chsub.csv", index=False, sep=';')
repl_dots("c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_chsub.csv", "c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_chsub(,).csv")



### compute correllations
##chan_names = erps.columns[3:]
##correl = {}
##erp_corrs = pd.DataFrame()
##for n in names:
##    erp_by_name = erps[erps.Name == n]
##    erp_by_name.ix[:,3:] = norm(erp_by_name.ix[:,3:], axis=0)  # normalization
##    for i in xrange(len(chan_names)):
##        erp_by_name[chan_names[i]] = np.isnan(erp_by_name[chan_names[i]])
##        corr_coef = erp_by_name['Dist'].corr(erp_by_name[chan_names[i]])
##        correl[chan_names[i]] = corr_coef
##        print corr_coef
##    erp_corrs = pd.concat((erp_corrs, pd.DataFrame(correl, index=[n])))
##
##erp_corrs.to_csv("c:\\SNN\\ERP_processing\\erp_correl_norm.csv", index=True, decimal=',', sep=';')
##
##
##erp_coef = pd.DataFrame()
##for n in names:
##    print n
##    erp_by_name = erps[erps.Name == n]
##    lr = linear_model.LinearRegression()
##    lr.fit(norm(erp_by_name.ix[:,3:], axis=0), erp_by_name.Dist)
##    coef = pd.DataFrame(lr.coef_.reshape(1,420), columns=chan_names, index=[n])
##    erp_coef = pd.concat((erp_coef, coef))
##
##erp_coef.to_csv("c:\\SNN\\ERP_processing\\erp_coeff_norm.csv", index=True, decimal=',', sep=';')

estimated_triads = pd.read_csv("c:\\SNN\\ERP_processing\\estimated_triads.csv")
st_types = pd.read_csv("c:\\SNN\\ERP_processing\\st_types.csv", header=None, encoding='cp1251').fillna('')

# neural encoding
def encode_data(data):
    """Take pandas series and encode every unique value"""
    encoded_data = data.copy()
    uniq_elements = data.unique()
    for i in xrange(len(uniq_elements)):
        encoded_data[data == uniq_elements[i]] = i
    return encoded_data

encoded_triads = pd.DataFrame()
encoded_triads['item'] = encode_data(st_types[0])
encoded_triads['relation'] = encode_data(st_types[1])
encoded_triads['attribute'] = encode_data(st_types[2])
encoded_triads['TypStim'] = st_types[3] - 128

encoded_triads.to_csv("c:\\SNN\\ERP_processing\\encoded_triads.csv",
                      index=False, header=True, sep=',')


# Create train files
predict = pd.read_csv("c:\\SNN\\ERP_processing\\ERPdata\\erp_rm_norm_Predict_good.txt",
                      decimal=',', sep=';')
encoded_triads = pd.read_csv("c:\\SNN\\ERP_processing\\encoded_triads.csv", sep=',')
uniq_names = predict.ix[:,0].unique()
path = "c:\\SNN\\ERP_processing\\train_files\\"

def make_train_csv(type_target, encoded_triads,  path):
    train_table = pd.merge(right=type_target, left=encoded_triads, on='TypStim')
    del train_table['TypStim']
    train_table.to_csv(path + n + '.csv', index=False, header=False, sep=',')

for n in uniq_names:
    type_target_byname = predict[predict.ix[:,0] == n][['TypStim', 'Predicted']]
    # scale
    minimum = type_target_byname.Predicted.min()
    maximum = type_target_byname.Predicted.max()
    scale = maximum - minimum
    type_target_byname.Predicted = (type_target_byname.Predicted - minimum) / scale
    # save
    make_train_csv(type_target_byname, encoded_triads, path)






###
direct = "c:\\SNN\\ERP_processing\\distances\\"
##file_name = os.listdir(direct)[1]
for file_name in os.listdir(direct):
    print file_name
    dist = pd.read_csv(direct+file_name, header=None, encoding='utf-8')
    print dist


estimated_triads = pd.read_csv("c:\\SNN\\ERP_processing\\estimated_triads.csv", encoding='cp1251')
stimulus = pd.read_csv("c:\\SNN\\epr_data_by_10_14\\stimuls.csv", encoding='cp1251', header=None)

names = estimated_triads.names.unique()
for n in names:
    type_dist = estimated_triads.ix[estimated_triads.names == n,[1,2]]
##    round_dist = (dist / 2.5).round()
##    type_dist.set_index([range(len(type_dist))])
    type_dist.index = range(len(type_dist))
    dist = type_dist.distances
    round_dist = type_dist.distances.copy()
    round_dist[dist>5] = 2
    round_dist[dist==0] = 0
    round_dist[(dist>0) & (dist<=5)] = 1
    type_dist.distances = round_dist
    dist_bystim = range(len(stimulus))
    for i in range(len(stimulus)):
        dist_bystim[i] = int(type_dist.ix[type_dist.types == stimulus.ix[i,0], 1])

    filename = "c:\\SNN\\ERP_processing\\dist_by_stim\\" + n +".txt"
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for val in dist_bystim:
            writer.writerow([val])
    print filename



