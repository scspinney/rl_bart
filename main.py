import numpy as np
import glob as glob
import os
from maxent import *
import matplotlib.pyplot as plt

maindir = '/Users/sean/Projects/rl_bart'
year=1

# number of trajectories to consider
N=200

# get all participant data and consider them as a single expert

# trajectories
traj_paths = os.path.join(maindir,'data',f'V{year}')
trajectories = [np.load(p,allow_pickle=True) for p in glob.glob(os.path.join(traj_paths,'**','traj.npy'))]

#trajectories = np.load(traj_path,allow_pickle=True)
#trajectories = trajectories[1:N]

# feature matrix
# fmat_path = os.path.join(maindir,'data',f'fmat_Y{year}.npy')
# feature_matrix = np.load(fmat_path)

fmat_paths = os.path.join(maindir,'data',f'V{year}')
feature_matrices = [np.load(p,allow_pickle=True) for p in glob.glob(os.path.join(fmat_paths,'**','fmat.npy'))]

# normalize features
#feature_matrix[:,0] = feature_matrix[:,0]/max(feature_matrix[:,0])
#feature_matrix[:,-1] = feature_matrix[:,-1]/max(feature_matrix[:,-1])



# transition prob: they are all the same so just pick one
Tprob = np.load(os.path.join(maindir,'data',f'transition_prob_Y{year}.npy'))



reward_weights = maxent_irl(maindir, year, feature_matrices, Tprob, gamma=0.9, trajectories=trajectories, lr=0.01,n_iters=40, use_prior=False)

np.save(os.path.join(maindir,'data','results',f'rewards_weights_{year}.npy'), reward_weights, allow_pickle=True)

