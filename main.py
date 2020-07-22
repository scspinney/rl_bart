import numpy as np
import glob as glob
import os
from maxent import *
from likelihood import *
import matplotlib.pyplot as plt

maindir = '/data/neuroventure/behavioral/nback_and_bart/rl_bart'
year=2


lr_decay=1


popPoints = [64,105,39,96,88,21,121,10,64,32,64,101,26,34,47,121,64,95,75,13,64,112,30,88,9,64,91,17,115,50]

# get all participant data and consider them as a single expert

# trajectories
traj_paths = os.path.join(maindir,'data',f'V{year}')
trajectories = [np.load(p,allow_pickle=True) for p in glob.glob(os.path.join(traj_paths,'**','traj.npy'))]

#trajectories = np.load(traj_path,allow_pickle=True)
#trajectories = trajectories[1:N]

# feature matrices
fmat_paths = os.path.join(maindir,'data',f'V{year}')
feature_matrices = [np.load(p,allow_pickle=True) for p in glob.glob(os.path.join(fmat_paths,'**','fmat.npy'))]


# transition prob: they are all the same so just pick one
Tprob = np.load(os.path.join(maindir,'data',f'transition_prob_Y{year}.npy'))

#TODO: temp
Tprob = Tprob[:-2,:,:-2]

reward_weights = maxent_irl(maindir, year, feature_matrices, Tprob, gamma=1, trajectories=trajectories, lr=0.01,lr_decay=lr_decay,n_iters=60,popPoints=popPoints, use_prior=False)

### likelihood section

weights = [14.5589288,  -10.91125129, 134.64405279,  -2.04189656,  60.60180254,
   5.08663998,  27.57215712,  11.22929793,  58.56049503,  69.74105026,
  -6.38408481]


ll = likelihood(trajectories, feature_matrices, weights, discount=0.95, Tprob=Tprob)

print(ll)




