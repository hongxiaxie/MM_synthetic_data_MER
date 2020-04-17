import numpy as np
import csv
import os
import pandas as pd
import pickle
import ast
"""
modified to meet the requirments of torch_geometric 
"""
#file = 'CASME/CASME2-coding-20190701.xlsx'
#df = pd.read_excel(file).drop(['Unnamed: 2','Unnamed: 6'], axis=1)


file='synthetic_label.xlsx'
df = pd.read_excel(file)
#
# build adj matrix of au (35 different au)
import itertools
from collections import Counter
# calculate occurance of each AU
au_list = [str(i) for i in df['Action Units'].values.tolist()]
au_list = [str(i).split("+") for i in au_list]

au_list_ =  [str(s).replace('L','') for s in au_list]# remove "L","R"
au_list_ =  [str(s).replace('R','') for s in au_list_]# remove "L","R"

#print("len of au_list",len(au_list_)) #255
for i in range(len(au_list_)):
    au_list_[i] = ast.literal_eval(au_list_[i])
    au_list_[i] = [int(i) for i in au_list_[i]]


au_nums = dict(Counter(i for i in list(itertools.chain.from_iterable(au_list_))))
#print(len(au_nums)) # 18 in synthetic data
#print(au_nums.keys())
#print(au_nums.values()) # the number of occurance




# create adj matrix (co-occurance)
au_keys = [i for i in (au_nums.keys())] # AU keys
au_adj = np.zeros((18,18))
#con_ = []
for i in au_list_:
    if len(i)!=1:
        # list all combination 
        com = list(itertools.combinations(i,2))
        #print("com:",com)
        #con_.append(com)
        for j in com:
            # create symmetric matrix
            #print(au_keys.index(j[0]))
            #print(au_keys.index(j[1]))
            au_adj[au_keys.index(j[0])][au_keys.index(j[1])] = 1     
            au_adj[au_keys.index(j[1])][au_keys.index(j[0])] = 1
#print(len(con_)) 
#l2 = []
#[l2.append(i) for i in con_ if not i in l2]
#print (len(l2)) 
#print(l2)        
#casme_adj = {}
#casme_adj['nums'] = np.array([i for i in (au_nums.values())])

#casme_adj = np.empty((19,19)) 
#casme_adj= np.array(au_adj)
#print(type(au_adj))
#print(au_adj)


file = open('synthetic_adj_18_new.pkl', 'wb')
pickle.dump(au_adj, file)
file.close()
print("finished create synthetic adj_file")
assert 0
#create au one hot encoded
au_one_hot = np.zeros((19,19))
for i in range(19):
    au_one_hot[i][i] = 1

file = open('au_one_hot_19.pkl', 'wb')
pickle.dump(au_one_hot, file)
file.close()
