import numpy as np
import pandas as pd
from plots import plot_weights_bar
from utils import *
from maxent import get_stats
from plots import plot_reward_landscape

# get the feature matrices and trajectories from the human experts
maindir = '/data/neuroventure/behavioral/nback_and_bart/rl_bart'
year=2

feature_matrices, Tprob, trajectories = load_data(maindir,year)
ai_players = {'human': 'results/theta_V2_N138_E100_LR0.0001_LRD1_S100.npy',
              'qlearner': 'results/QL_ai_theta_V2_N138_E100_LR0.0001_LRD1_S100.npy',
              'alwayspump': 'results/AlwaysPump_ai_theta_V2_N138_E100_LR0.0001_LRD1_S100.npy',
              'random': 'results/Random_ai_theta_V2_N138_E100_LR0.0001_LRD1_S100.npy'}


# get weights
all_weights = process_weights(ai_players['human'],'human')

for p,f in ai_players.items():
    all_weights.append(process_weights(f,p))

# bar plots of the weights
plot_weights_bar(all_weights)

# reward landscape plots
N_EXPERTS = 138
N_TRIAL = 30
N_STATES = 128
N_FEAT = 11

obs_exp_rewards, avg_save_state = get_stats(N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,trajectories)

for p,f in ai_players.items():
    weights = np.load(f)
    weights = weights[-1].mean(axis=(0,1))

    plot_reward_landscape(p,N_EXPERTS,N_TRIAL,N_STATES,N_FEAT,weights,feature_matrices,obs_exp_rewards,avg_save_state)
