import numpy as np
import random
import pandas as pd
import argparse
import pickle
import copy
 
argparser = argparse.ArgumentParser()
argparser.add_argument("--pes_file", type = str) #input is the energy profile
 
with open('./'+pes_file+'.pkl','rb') as f:
    data = pickle.load(f)
data = data.sort_values(by=['idx_x','idx_y','idx_z'])
#set outside to a very high energy
data.loc[~((data['idx_inner_pore']==1)&(data['idx_overlap']==0)), 'energy'] = 200.0 
pes = data['energy'].values.reshape(41,41,80)
pes *=1.689 #kcal/mol to kT
 
num_traj = 1000
state = np.zeros((num_traj,3),dtype=np.int) #initialize the particle position
state[:,0] = 20 #start of x is center
state[:,1] = 20 #start of y is center
pos_z_max = 79 #the end of z; adsorption boundary
 
idx = state[:,0],state[:,1],state[:,2]
 
energy = pes[idx]
 
time_max = 10**6
freq_record = 100
 
time_record = np.zeros(num_traj,dtype=np.int)
#record traj with a given frequency
traj_record = np.zeros((int(time_max/freq_record),num_traj,3),dtype=np.int) 
 
for i in range(time_max):
    state_delta = 2*np.random.randint(2, size=num_traj)-1
    direction_delta = np.random.randint(3, size=num_traj)
    idx = (np.arange(num_traj),direction_delta) #the direction of the move are randomly chosen
    state_new = copy.deepcopy(state)
    state_new[idx] += state_delta
    condition1 = state_new[:,2]>0 #require particle inside
    condition2 = state_new[:,2]<=pos_z_max
    condition = condition1.astype(int)+condition2.astype(int)
    condition_flag = (condition==2)
    state_new = state_new*condition_flag[..., np.newaxis] + state*(1-condition_flag)[..., np.newaxis] #new state
 
    idx = state_new[:,0],state_new[:,1],state_new[:,2]
    energy_new = pes[idx] #energy of new state
 
    energy_delta = energy_new-energy
    accept_prob = np.exp(-energy_delta)
    #state will be accepted based on Metropolis acceptance ratio
    accept_flag = np.random.random(num_traj)<accept_prob 
    #update energy
    state = state_new*accept_flag[..., np.newaxis] + state*(1-accept_flag)[..., np.newaxis] 
    idx = state[:,0],state[:,1],state[:,2]
    energy = pes[idx] #update state
    time_record += (state[:,2]==pos_z_max)*i*(time_record==0) #record fist-passage time
    if i%freq_record==0:
        print (i)
        traj_record[int(i/freq_record)] = state
 
np.save('./mfpt_xy/%s_time_record.npy'%(pes_file), time_record)
np.save('./mfpt_xy/%s_traj_record.npy'%(pes_file), traj_record)

