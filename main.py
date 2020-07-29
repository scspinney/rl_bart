import numpy as np
import glob as glob
import os
from maxent import *
from likelihood import *
from plots import *
import matplotlib.pyplot as plt

maindir = '/Users/sean/Projects/rl_bart/'
year=1


lr_decay=1

N=68

popPoints = [64,105,39,96,88,21,121,10,64,32,64,101,26,34,47,121,64,95,75,13,64,112,30,88,9,64,91,17,115,50]

# get all participant data and consider them as a single expert

# trajectories
traj_paths = os.path.join(maindir,'data',f'V{year}')
trajectories = [np.load(p,allow_pickle=True) for p in np.sort(glob.glob(os.path.join(traj_paths,'**','traj.npy')))]

#trajectories = np.load(traj_path,allow_pickle=True)
#trajectories = trajectories[1:N]

# feature matrices
fmat_paths = os.path.join(maindir,'data',f'V{year}')
feature_matrices = [np.load(p,allow_pickle=True) for p in np.sort(glob.glob(os.path.join(fmat_paths,'**','fmat.npy')))]


# transition prob: they are all the same so just pick one
Tprob = np.load(os.path.join(maindir,'data',f'transition_prob_Y{year}.npy'))

#TODO: temp
Tprob = Tprob[:-2,:,:-2]

#TODO: temporary for debugging 12th feature
trajectories = trajectories[:N]
feature_matrices = feature_matrices[:N]

N_EXPERTS = len(feature_matrices)
N_TRIAL, N_STATES, N_FEAT = np.shape(feature_matrices[0])
N_STATES -= 2

#weights = maxent_irl(maindir, year, feature_matrices, Tprob, gamma=1, trajectories=trajectories, lr=1E-4,lr_decay=lr_decay,n_iters=100,n_epochs=2,popPoints=popPoints, use_prior=False)

### likelihood section

weights=[ 0.5878049,   0.25019523,  0.6986219,  -0.0080535,  0.34347032,  0.67330279,
            0.69810915,  0.30025994,  0.6172051,   0.51155751, -0.31407854]



#TODO: not working
obs_exp_rewards, avg_save_state = get_stats(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,trajectories)



### PLOTTING ###

plot_reward_landscape(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,weights,feature_matrices,obs_exp_rewards,avg_save_state,'line')

# load gradients
gradients = np.load('results/gradients_V1.npy')
plot_gradients(gradients)

#ll = likelihood(trajectories, feature_matrices, weights, discount=1, Tprob=Tprob)

#print(ll)







