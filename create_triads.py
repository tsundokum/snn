#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      11
#
# Created:     25.04.2014
# Copyright:   (c) 11 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python
#-*- coding: utf-8 -*-

from pymorphy import get_morph
import sys
import codecs
import re
import csv
import random


##[path_item, path_rel, path_attr] = [path_item_it, path_rel, path_attr_it]

def create_triads(path_item, path_rel, path_attr):
    dicts = "c:\\Python27\\Lib\\site-packages\\pymorphy\\ru.sqlite-json\\"
    morph = get_morph(dicts)

    # read items
    with open(path_item) as f:
        items = f.readlines()
    # read relations
    with open(path_rel) as f:
        relations = f.readlines()
    # read attributes
    with open(path_attr) as f:
        attributes = f.readlines()

    # split attributes according to different parts of speech
    attrsN, attrsV, attrsAdj, attrsIs = [[],[],[],[]]
    for at in attributes:
        if 'N' in at: attrsN.append(re.split(',', at)[0].decode('cp1251').lower())
        if 'V' in at: attrsV.append(re.split(',', at)[0].decode('cp1251').lower())
        if 'Adj' in at: attrsAdj.append(re.split(',', at)[0].decode('cp1251').lower())
        if 'Is' in at: attrsIs.append(re.split(',', at)[0].decode('cp1251').lower())

    # assemble triads
    triads = []
    for it in items:
        it = it.replace('\n', '').decode('cp1251')
        for rel in relations:
            rel = rel.replace('\n', '').decode('cp1251')
            if rel == u'может':
                for attr in attrsV: triads.append([it, rel, attr])
            if rel == u'имеет':
                for attr in attrsN: triads.append([it, rel, attr])
            if rel == u'является':
                for attr in attrsIs: triads.append([it, rel, attr])
            if u'как' in rel:
                for attr in attrsAdj: triads.append([it, '', attr])

    # test
    for triad in triads:
        print triad[0] + ', ' + triad[1] + ', ' + triad[2]

    return triads


path = "c:\\SNN\\txts\\triads\\"
path_item_it = path + 'items_it.txt'
path_item_pers = path + 'items_pers.txt'
path_rel_it = path + 'relations_it.txt'
path_rel_pers = path + 'relations_pers.txt'
path_attr_it = path + 'attributes_items.txt'
path_attr_pers = path + 'attributes_pers.txt'

triads_it = create_triads(path_item_it, path_rel_it, path_attr_it)
triads_pers = create_triads(path_item_pers, path_rel_pers, path_attr_pers)
triads_all = triads_it + triads_pers
print len(triads_all)

for i in xrange(len(triads_all)):
    triads_all[i].append(str(128 + i))

# wirte csv
with open(path  + 'all_triads.csv', 'wb') as triads_file:
  file_writer = csv.writer(triads_file)
  for i in xrange(len(triads_all)):
      file_writer.writerow([tr.encode('cp1251') for tr in triads_all[i]])




# load prepared triads
triads_all = []
with open(path  + 'all_triads.csv', 'rb') as f:
    for row in f.readlines():
        triads_all.append(row.replace('\r\n', '').decode('cp1251').split(','))

# multiply and permute triads
traids_mult = triads_all * 10
random.shuffle(traids_mult)

# load model sample
with open(path  + 'model_sample.csv') as f:
    model_sample = f.readlines()
    model_head = model_sample[0].decode('cp1251').split(';')
    model_sample = model_sample[1:]

table = []
table.append(model_head)
for i in xrange(len(model_sample)):
    if i >= len(traids_mult):
        break
    line = model_sample[i].decode('cp1251').split(';')
    if int(line[model_head.index('TypStim')]) >= 128:
        line[model_head.index('NumStim')] = str(6 + i)
        line[model_head.index('NameStim')] = traids_mult[i][0] + ' ' + traids_mult[i][1] + ' ' + traids_mult[i][2]
        line[model_head.index('TypStim')] = str(traids_mult[i][3])
        line[model_head.index('Wait')] = str(random.randint(300, 500))
        line[model_head.index('Stimul')] = ' /~/~/~/~/~/~/~/~/~/~/~' + traids_mult[i][2]
        line[model_head.index('PrimeStim')] = ' /~/~/~/~/~/~/~/~/~/~/~' + traids_mult[i][0] + ' ' + traids_mult[i][1]
    table.append(line)
# append ending row
table.append(model_sample[-1].decode('cp1251').split(';'))

# wirte csv
with open(path  + 'stim_table.csv', 'wb') as table_file:
  file_writer = csv.writer(table_file, delimiter=';')
  for i in xrange(len(table)):
      file_writer.writerow([tr.encode('cp1251') for tr in table[i]])


### check content
###!/usr/bin/env python
### -*- coding: utf-8 -*-
##path = "c:\\SNN\\txts\\"
##words_output_file = 'resources\\test-distances.csv'
##
##with open(path + '\\' + words_output_file) as f:
##    for row in f.readlines():
##        print row.decode('utf-8')



