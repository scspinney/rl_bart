from utils import load_data
from maxent import get_stats
from likelihood import likelihood
from plots import *
import numpy as np

maindir = '/data/neuroventure/behavioral/nback_and_bart/rl_bart'
year=2
N=148


## Load data

feature_matrices, Tprob, trajectories = load_data(maindir,year,N)

N_EXPERTS = len(feature_matrices)
N_TRIAL, N_STATES, N_FEAT = np.shape(feature_matrices[0])
N_STATES -= 2

gradients = np.load(f'results/gradients_V{year}.npy')
weights = np.load(f'results/theta_V{year}.npy')


obs_exp_rewards, avg_save_state = get_stats(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,trajectories)

### PLOTTING ###

plot_reward_landscape(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,weights,feature_matrices,obs_exp_rewards,avg_save_state,'line')

plot_gradients(gradients)

ll = likelihood(trajectories, feature_matrices, weights, discount=1, Tprob=Tprob)


