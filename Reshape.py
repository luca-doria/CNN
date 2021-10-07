#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from scipy.interpolate import griddata
import time

## Read data ##

t0 = time.time()

df = pd.read_csv('ar39_8_raw.csv', sep=';', decimal='.', header=None)
df.dropna(axis='columns', inplace=True)
df.columns=['Event','PMT','Q','T','costh','phi','x','y','z']

df_all = pd.read_csv('All_pmts.csv', sep=';', decimal='.', header=None)
df_all.dropna(axis='columns', inplace=True)
df_all.columns=['PMT','costheta','phi']
df_all = df_all.sort_values(by=['PMT'])

## Read data from 16x16 matrix of PMTs ##

with open("PMT_spatial_grid.txt") as f:
    lis = [line.split() for line in f]
    single_lis = [int(item) for sublist in lis for item in sublist]

#######################

## Extract charge ##

for e in range(5000):

    df_event = df.loc[df['Event'] == e]
    df_event=df_event[["PMT", "Q", "costh", "phi"]]
    
    ## Normalize Q ##
    
    Q_column=df_event["Q"]
    max_Q=Q_column.max()
    
    df_event['Q']=df_event['Q'].apply(lambda x: x/max_Q)
    
    df_event = df_event.sort_values(by=['PMT'])
    
    ## Search for PMTs ##
    
    PMT_column = df_event.loc[:,'PMT']
    PMT = PMT_column.values
    
    Q_column = df_event.loc[:,'Q']
    Q = Q_column.values
    
    costh_column = df_event.loc[:,'costh']
    costh = costh_column.values
    
    phi_column = df_event.loc[:,"phi"]
    phi = phi_column.values

    
    ## All PMTs ##
    
    costheta_all_column = df_all.loc[:,'costheta']
    costheta_all = costheta_all_column.values
    
    phi_all_column = df_all.loc[:,"phi"]
    phi_all = phi_all_column.values

    #############
    
    mix = [PMT, Q, costh, phi]
    
    for i in range(0, 255):
        if i not in PMT:
            mix[0] = np.append(mix[0], i)
            mix[1] = np.append(mix[1], 0)
            mix[2] = np.append(mix[2], costheta_all[i])
            mix[3] = np.append(mix[3], phi_all[i])
    
    mix[0] = np.append(mix[0], 255)
    mix[1] = np.append(mix[1], 0)
    
    mix_df = pd.DataFrame(mix)
    mix_df = mix_df.transpose()
    mix_df.columns = ['PMT','Q', 'costh', 'phi']
    
    mix_df.sort_values(by=["PMT"] ,inplace=True)
    mix_df.reset_index(drop=True, inplace=True)
    
    # print(mix_df.head(15))
    
    Q_array=[]
    
    for i in single_lis:
        a = mix_df.loc[mix_df['PMT'] == i]
        Q_array.append(a.iloc[0]['Q'])
    
    #print(Q_array)
    
    # Q_column = mix_df.loc[:,'Q']
    # Q = Q_column.values
    Q_round=np.around(Q_array, decimals = 4)
      
    img_array = np.split(Q_round, 16)   # if I am not mistaken this is the only new part of the code
    n = 8
    img_array = np.kron(img_array, np.ones((n,n))) # kroneker product function to make bigger matrices
    
    #print(img_array)
    # plt.imshow(img_array)
    # plt.savefig("Event3.png")
    # plt.show()
    
    with open('Inception_data.dat', 'a') as f:
        f.writelines(' '.join(str(elem) for elem in row) + '\n' for row in img_array)

##################    

## Extracting coordinates ##

# i=0

# df_event=df[["Event", "x", "y", "z"]]
# coor=[]
   
# for row in df.iterrows():
#     ind=df_event.loc[df_event["Event"]==i].index[0]
#     event_row = df_event.iloc[ind]
#     x=str(round(event_row["x"], 2))
#     y=str(round(event_row["y"], 2))
#     z=str(round(event_row["z"], 2))
#     xyz=[x, y, z]
#     i+=1
    
#     with open('All data/All_data_coordinates.dat', 'a') as f:
#         f.writelines(' '.join(elem for elem in xyz) + '\n')

###########################

t1 = time.time()
total = t1-t0
print(total)

## This is how I split training and test data

with open('Inception_data.dat','r') as original, open('Inception_test.dat','w') as new:
    original_lines = original.readlines()
    new.writelines(original_lines[1920000:]) 
