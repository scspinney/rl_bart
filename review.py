from utils import load_data
from maxent import get_stats
from likelihood import likelihood
from plots import *
import numpy as np

np.set_printoptions(suppress=True)

maindir = '/Users/sean/Projects/rl_bart'
year=2
N=138


## Load data

feature_matrices, Tprob, trajectories = load_data(maindir,year,N)

N_EXPERTS = len(feature_matrices)
N_TRIAL, N_STATES, N_FEAT = np.shape(feature_matrices[0])
N_STATES -= 2

gradients = np.load(f'results/gradients_V{year}.npy')
weights = np.load(f'results/theta_V{year}.npy')

# pass the average weights over the 30 trajectories, over the N experts
avg_weights = weights.mean(axis=1).mean(axis=1)[-1]

obs_exp_rewards, avg_save_state = get_stats(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,trajectories)

### PLOTTING ###

#plot_reward_landscape(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,avg_weights,feature_matrices,obs_exp_rewards,avg_save_state,'line',clobber=True)
#plot_gradients(gradients)
plot_weights(weights,'line')

#avg_LL = likelihood(N_TRIAL,trajectories, feature_matrices, avg_weights, discount=1, Tprob=Tprob)

#print(f"Average Log Likelihood on training demonstrations: N = {N_EXPERTS*N_TRIAL} demonstrations, LL = {avg_LL}")


